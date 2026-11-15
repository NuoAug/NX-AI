"""阶段 4 真机软件闭环：真实语音 → 双图比较 → 语音播报；打断后“继续”复用已取图。

覆盖方案阶段 4 的软件层：A/B 两位置取图与计数、按项目（价格）同口径比较、
完成即释放引用；被打断的任务保留已取帧，说“继续”时不重复取图。

运行：
    source Tools/target_env.sh && trun "python -m pytest -q Tests/Integration/test_compare_engine_real.py -s"
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Callable

import pytest

from aieyes.audio.replay_source import ReplaySource
from aieyes.audio.vad import EnergyVad
from aieyes.chat import ChatSession
from aieyes.config import AppConfig
from aieyes.events import EventBus
from aieyes.imagetasks import ImageTaskStore
from aieyes.interaction.engine import InteractionEngine
from aieyes.logging_setup import JsonlEvidenceLogger
from aieyes.scene import FileFrameProvider, SceneService
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
    """价格比较不应提交聊天模型；若被调用说明意图路由错了。"""

    def ask(self, text: str):  # pragma: no cover - 失败路径
        raise AssertionError(f"比较意图不应提交聊天模型: {text}")


@pytest.fixture()
def config() -> AppConfig:
    return _load_cfg()


def _load_cfg() -> AppConfig:
    import logging as _logging

    _logging.basicConfig(level=_logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = AppConfig.from_file(CONFIG)
    if not Path(config.models.root).is_dir():
        pytest.skip(f"目标机模型目录不存在: {config.models.root}")
    config.audio.backend = "replay"
    config.audio.replay_realtime = True
    return config


def _require_server(config: AppConfig) -> None:
    client = VlmClient(config.llm, config.vlm)
    try:
        ok = client.health()
    finally:
        client.close()
    if not ok:
        pytest.skip(f"llama-server 未就绪（{config.llm.base_url}），先运行 Tools/serve_llama.sh")


def _wake_path() -> Path:
    wake = FIXTURES / "audio" / "wake" / "tts_xiaobing_nihao.wav"
    if not wake.is_file():
        pytest.skip("缺少唤醒素材 tts_xiaobing_nihao.wav")
    return wake


def _compare_images() -> list[Path]:
    a = FIXTURES / "scene" / "compare_price_a.png"
    b = FIXTURES / "scene" / "compare_price_b.png"
    if not (a.is_file() and b.is_file()):
        pytest.skip(f"缺少比较图: {a} / {b}")
    return [a, b]


def _question_fixture(name: str) -> Path:
    """固定的问句音频（合成一次并做过 ASR 自检，见 Tools/make_compare_audio_fixtures.py）。"""
    path = FIXTURES / "audio" / "questions" / f"{name}.wav"
    if not path.is_file():
        pytest.skip(f"缺少问句素材 {path}（先跑 Tools/make_compare_audio_fixtures.py）")
    return path


def _run_compare_scenario(
    config: AppConfig,
    segments: list,
    provider: FileFrameProvider,
    evidence_path: Path,
    wait: Callable[[list[dict]], bool],
    timeout_s: float,
    prepare: Callable[[InteractionEngine], None] | None = None,
) -> tuple[bool, list[dict], RecordingSink]:
    source = ReplaySource(segments, realtime=True)
    frontend = SpeechFrontend(
        source,
        kws=SherpaKws(config.kws),
        vad=EnergyVad(config.vad),
        asr=StreamingAsr(config.asr),
    )
    sink = RecordingSink()
    tts = PiperTts(config.tts)
    evidence = JsonlEvidenceLogger(evidence_path)

    async def _scenario() -> bool:
        bus = EventBus()
        engine = InteractionEngine(
            cfg=config,
            bus=bus,
            frontend=frontend,
            sink=sink,
            tts=tts,
            chat=ChatSession(UnusedChat()),
            scene=SceneService(vlm=VlmClient(config.llm, config.vlm), ocr=OcrEngine(config.ocr), provider=provider),
            evidence=evidence,
            image_tasks=ImageTaskStore(),
        )
        if prepare is not None:
            prepare(engine)
        sink.start()
        await engine.start()
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
            engine.scene.close()

    ok = asyncio.run(_scenario())
    return ok, _read_records(evidence_path), sink


def _dump(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)


def test_compare_two_price_labels(config: AppConfig, tmp_path: Path) -> None:
    """“这两个哪个贵？” → A/B 各取一张 → OCR 比较价格 → 播报差异。"""
    _require_server(config)
    question = _question_fixture("compare_two_which_expensive")

    provider = FileFrameProvider(_compare_images())
    evidence = tmp_path / "compare.jsonl"
    try:
        ok, records, sink = _run_compare_scenario(
            config,
            [_wake_path(), _silence(2500), question, _silence(2500)],
            provider,
            evidence,
            wait=lambda recs: bool(_find(recs, "answer", "spoken")),
            timeout_s=60.0,
        )
    finally:
        provider.close()

    assert ok, f"60s 内没有完成双图比较，证据:\n{_dump(records)}"
    compare_answer = _find(records, "compare", "answer")
    assert compare_answer, f"缺少比较回答记录: {_dump(records)}"
    rec = compare_answer[-1]
    assert "价格" in str(rec["common"]), f"比较项不是价格: {rec}"
    assert rec["a_frame"] and rec["b_frame"], rec
    captures = _find(records, "compare", "capture")
    assert len(captures) == 2, f"A/B 应各取图一次: {captures}"
    answers = _find(records, "answer", "spoken")
    assert answers and answers[-1].get("played") is True, f"回答没有完整播完: {answers}"
    spoken = str(answers[-1].get("text", ""))
    assert "12元5角" in spoken and "9元9角" in spoken, f"播报中缺少两边价格: {spoken}"
    assert not _find(records, "llm", "done"), "价格比较不应提交大模型"
    print("compare:", {"answer": spoken, "compare_ms": rec["ms"], "captures": len(captures)})


def test_resume_after_pause_reuses_captured_frames(config: AppConfig, tmp_path: Path) -> None:
    """被打断的 A/B 任务保留已取帧：说“继续”时不应重新取图。"""
    _require_server(config)
    resume = _question_fixture("resume_compare")

    provider = FileFrameProvider(_compare_images())
    prepared: dict[str, object] = {}

    def prepare(engine: InteractionEngine) -> None:
        task = engine.image_tasks.open_compare("比较一下这两个的价格")
        frames = [provider.acquire("vqa"), provider.acquire("vqa")]
        task.frames["A"] = frames[0]
        task.frames["B"] = frames[1]
        engine.image_tasks.mark(task, "paused", reason="barge_in")
        prepared["task"] = task

    evidence = tmp_path / "resume.jsonl"
    try:
        ok, records, sink = _run_compare_scenario(
            config,
            [_wake_path(), _silence(2500), resume, _silence(2500)],
            provider,
            evidence,
            wait=lambda recs: bool(_find(recs, "answer", "spoken")),
            timeout_s=60.0,
            prepare=prepare,
        )
    finally:
        provider.close()

    assert ok, f"60s 内没有完成恢复比较，证据:\n{_dump(records)}"
    assert _find(records, "compare", "resume"), f"缺少恢复记录: {_dump(records)}"
    assert not _find(records, "compare", "capture"), "复用已缓存帧时不应重新取图"
    answers = _find(records, "answer", "spoken")
    assert answers and answers[-1].get("played") is True, f"回答没有完整播完: {answers}"
    spoken = str(answers[-1].get("text", ""))
    assert "12元5角" in spoken and "9元9角" in spoken, f"播报中缺少两边价格: {spoken}"
    task = prepared["task"]
    assert getattr(task, "state", "") == "answered", getattr(task, "state", "")
    print("resume:", {"answer": spoken})
