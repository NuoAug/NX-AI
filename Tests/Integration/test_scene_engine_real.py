"""阶段 3 真机软件闭环：真实语音 → 意图 → 取图 → OCR/VLM → 语音播报。

覆盖方案阶段 3（L105–109）的软件层：单图读字、价格确定性解析、看图描述；
相机路径单独用例（设备被占用/不可用则 skip）。llama-server 未就绪时整体 skip。

运行：
    source Tools/target_env.sh && trun "python -m pytest -q Tests/Integration/test_scene_engine_real.py -s"
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable

import pytest

from aieyes.audio.replay_source import ReplaySource
from aieyes.audio.vad import EnergyVad
from aieyes.audio.wav import write_wav
from aieyes.chat import ChatSession
from aieyes.config import AppConfig
from aieyes.events import EventBus
from aieyes.interaction.engine import InteractionEngine
from aieyes.logging_setup import JsonlEvidenceLogger
from aieyes.scene import CameraFrameProvider, FileFrameProvider, FramePolicy, SceneService
from aieyes.speech.asr import StreamingAsr
from aieyes.speech.frontend import SpeechFrontend
from aieyes.speech.kws import SherpaKws
from aieyes.speech.tts import PiperTts
from aieyes.vision.ocr import OcrEngine
from aieyes.vision.vlm import VlmClient

# 复用回放测试里的 RecordingSink 与证据读取工具（同一目录，pytest 直接可导入）
from test_replay_voice_chain import (  # noqa: E402
    RecordingSink,
    _find,
    _read_records,
    _silence,
    _wait_until,
)

pytestmark = [pytest.mark.target, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "Tests" / "Fixtures"
CONFIG = ROOT / "Config" / "target.json"


class UnusedChat:
    """场景意图不应走聊天模型；若被调用说明意图路由错了。"""

    def ask(self, text: str):  # pragma: no cover - 失败路径
        raise AssertionError(f"场景意图不应提交聊天模型: {text}")


def _require_server(cfg: AppConfig) -> None:
    client = VlmClient(cfg.llm, cfg.vlm)
    try:
        ok = client.health()
    finally:
        client.close()
    if not ok:
        pytest.skip(f"llama-server 未就绪（{cfg.llm.base_url}），先运行 Tools/serve_llama.sh")


def _build_scene(cfg: AppConfig, provider) -> SceneService:
    return SceneService(
        vlm=VlmClient(cfg.llm, cfg.vlm),
        ocr=OcrEngine(cfg.ocr),
        provider=provider,
    )


def _make_question_wav(tts: PiperTts, text: str, path: Path) -> Path:
    audio = tts.synth(text)
    write_wav(path, audio.pcm, audio.sample_rate)
    return path


def _run_scene_scenario(
    cfg: AppConfig,
    segments: list,
    scene: SceneService,
    evidence_path: Path,
    wait: Callable[[list[dict]], bool],
    timeout_s: float,
) -> tuple[bool, list[dict], RecordingSink]:
    """装配真实语音链路 + 真实场景服务跑一次回放场景。"""
    source = ReplaySource(segments, realtime=True)
    frontend = SpeechFrontend(
        source,
        kws=SherpaKws(cfg.kws),
        vad=EnergyVad(cfg.vad),
        asr=StreamingAsr(cfg.asr),
    )
    sink = RecordingSink()
    tts = PiperTts(cfg.tts)
    evidence = JsonlEvidenceLogger(evidence_path)

    async def _scenario() -> bool:
        bus = EventBus()
        engine = InteractionEngine(
            cfg=cfg,
            bus=bus,
            frontend=frontend,
            sink=sink,
            tts=tts,
            chat=ChatSession(UnusedChat()),
            scene=scene,
            evidence=evidence,
        )
        sink.start()
        await engine.start()
        # 引擎就绪（进入待机、问候音已预合成）后再开始采集，避免启动竞态
        frontend.start()
        try:
            return await _wait_until(
                lambda: wait(_read_records(evidence_path)), timeout_s=timeout_s
            )
        finally:
            await engine.stop()
            frontend.stop()
            sink.close()
            tts.close()

    ok = asyncio.run(_scenario())
    return ok, _read_records(evidence_path), sink


def _dump(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)


@pytest.fixture()
def cfg() -> AppConfig:
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = AppConfig.from_file(CONFIG)
    if not Path(config.models.root).is_dir():
        pytest.skip(f"目标机模型目录不存在: {config.models.root}")
    config.audio.backend = "replay"
    config.audio.replay_realtime = True
    return config


def _wake_path() -> Path:
    wake = FIXTURES / "audio" / "wake" / "tts_xiaobing_nihao.wav"
    if not wake.is_file():
        pytest.skip("缺少唤醒素材 tts_xiaobing_nihao.wav")
    return wake


def test_price_question_reads_number_from_image(cfg: AppConfig, tmp_path: Path) -> None:
    """价格意图 → 文件取图 → OCR 确定性解析 → 播报含“12.50”的价格答案。"""
    _require_server(cfg)
    image = FIXTURES / "images" / "text_cn_price.png"
    if not image.is_file():
        pytest.skip(f"缺少测试图: {image}")
    tts = PiperTts(cfg.tts)
    try:
        question = _make_question_wav(tts, "这个多少钱？", tmp_path / "q_price.wav")
    finally:
        tts.close()

    provider = FileFrameProvider(image)
    scene = _build_scene(cfg, provider)
    evidence = tmp_path / "price.jsonl"
    try:
        ok, records, sink = _run_scene_scenario(
            cfg,
            [_wake_path(), _silence(2500), question, _silence(1800)],
            scene,
            evidence,
            wait=lambda recs: bool(_find(recs, "answer", "spoken")),
            timeout_s=60.0,
        )
    finally:
        scene.close()
        provider.close()

    assert ok, f"60s 内没有完成价格问答，证据:\n{_dump(records)}"
    assert _find(records, "intent", "price"), f"意图不是价格: {_dump(records)}"
    scene_answer = _find(records, "scene", "answer")
    assert scene_answer, f"缺少场景回答记录: {_dump(records)}"
    rec = scene_answer[-1]
    assert rec["intent"] in ("price", "read_text"), rec
    assert "ocr" in str(rec["sources"]), f"价格应来自 OCR: {rec}"
    answers = _find(records, "answer", "spoken")
    assert answers and answers[-1].get("played") is True, f"回答没有完整播完: {answers}"
    spoken = str(answers[-1].get("text", ""))
    assert "12" in spoken, f"播报里没有价格数字: {spoken}"
    assert sink.records, "没有任何播放记录"
    print(
        "price:",
        {
            "intent": rec["intent"],
            "sources": rec["sources"],
            "answer": spoken,
            "scene_ms": rec["ms"],
        },
    )


def test_look_question_describes_image(cfg: AppConfig, tmp_path: Path) -> None:
    """“前面有什么东西” → 文件取图 → VLM 描述 → 播报。"""
    _require_server(cfg)
    image = FIXTURES / "images" / "scene_red_square.png"
    if not image.is_file():
        pytest.skip(f"缺少测试图: {image}")
    tts = PiperTts(cfg.tts)
    try:
        question = _make_question_wav(tts, "看看前面有什么？", tmp_path / "q_look.wav")
    finally:
        tts.close()

    provider = FileFrameProvider(image)
    scene = _build_scene(cfg, provider)
    evidence = tmp_path / "look.jsonl"
    try:
        ok, records, sink = _run_scene_scenario(
            cfg,
            [_wake_path(), _silence(2500), question, _silence(1800)],
            scene,
            evidence,
            wait=lambda recs: bool(_find(recs, "answer", "spoken")),
            timeout_s=60.0,
        )
    finally:
        scene.close()
        provider.close()

    assert ok, f"60s 内没有完成看图描述，证据:\n{_dump(records)}"
    scene_answer = _find(records, "scene", "answer")
    assert scene_answer, f"缺少场景回答记录: {_dump(records)}"
    rec = scene_answer[-1]
    assert rec["intent"] in ("describe", "find"), rec
    assert rec["chars"] > 0
    answers = _find(records, "answer", "spoken")
    assert answers and answers[-1].get("played") is True, f"回答没有完整播完: {answers}"
    print(
        "look:",
        {
            "intent": rec["intent"],
            "sources": rec["sources"],
            "answer": answers[-1].get("text"),
            "scene_ms": rec["ms"],
        },
    )


def test_camera_look_real(cfg: AppConfig, tmp_path: Path) -> None:
    """真实相机路径：/dev/video* 取当前一帧并描述；设备被占用/不可用则 skip。"""
    _require_server(cfg)
    # 现场可能偏暗/模糊：允许降级取帧（光照不足时也要给出受限回答，而不是直接失败）。
    # 降级信息会体现在 scene.answer 的 degraded 里。
    provider = CameraFrameProvider(policy=FramePolicy(require_clear=False))
    try:
        frame = provider.acquire(purpose="vqa")
    except Exception as exc:  # noqa: BLE001 - 相机不可用属于环境限制
        provider.close()
        pytest.skip(f"相机不可用: {exc}")
    quality = getattr(provider.last_acquisition, "quality", None)
    scene = _build_scene(cfg, provider)
    tts = PiperTts(cfg.tts)
    try:
        question = _make_question_wav(tts, "看看前面有什么？", tmp_path / "q_camera.wav")
    finally:
        tts.close()
    evidence = tmp_path / "camera.jsonl"
    try:
        ok, records, _sink = _run_scene_scenario(
            cfg,
            [_wake_path(), _silence(2500), question, _silence(1800)],
            scene,
            evidence,
            wait=lambda recs: bool(_find(recs, "answer", "spoken")),
            timeout_s=60.0,
        )
    finally:
        scene.close()
        provider.close()

    assert ok, f"60s 内没有完成相机看图描述，证据:\n{_dump(records)}"
    scene_answer = _find(records, "scene", "answer")
    assert scene_answer, f"缺少场景回答记录: {_dump(records)}"
    rec = scene_answer[-1]
    assert rec["frame_id"], f"缺少取图来源: {rec}"
    answers = _find(records, "answer", "spoken")
    assert answers and answers[-1].get("played") is True, f"回答没有完整播完: {answers}"
    utt = _find(records, "utterance", "final")
    assert utt, "没有识别到问题"
    print(
        "camera:",
        {
            "frame_id": rec["frame_id"],
            "answer": answers[-1].get("text"),
            "preview_frame": frame.frame_id,
            "quality": str(quality),
            "degraded": rec["degraded"],
            "scene_ms": rec["ms"],
        },
    )
