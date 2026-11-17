"""阶段 1 真机软件闭环：回放素材 + 真实 llama-server（qwen35）。

覆盖方案阶段 1 的三条路径（`Docs/DeepSeek-V4.1-Flash1017_…含完整方案.md` L93–97）：

1. **只唤醒**：只有唤醒词、没有提问时，本地模型提交次数必须为零，T01 超时后回到待机；
2. **播完立即提问**：唤醒音播完后紧接真人问题，回答来自本地模型真实输出（非替身文本）；
3. **唤醒词与问题同句**：一句“小冰，一加一等于几？”里同时含唤醒词与问题，只受理一次、
   只提交一次模型请求。

前置：目标板 llama-server 已启动（`Tools/serve_llama.sh`）。未就绪时本文件整体 skip。
运行：
    source Tools/target_env.sh && trun "python -m pytest -q Tests/Integration/test_replay_real_llm.py -s"
"""

from __future__ import annotations

import asyncio
import json
import logging
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
from aieyes.speech.asr import StreamingAsr
from aieyes.speech.frontend import SpeechFrontend
from aieyes.speech.kws import SherpaKws
from aieyes.speech.tts import PiperTts
from aieyes.vision.vlm import VlmClient

# 复用阶段 1 回放测试里的 RecordingSink 与证据读取工具（同一目录，pytest 直接可导入）
from test_replay_voice_chain import (  # noqa: E402
    RecordingSink,
    _find,
    _read_records,
    _silence,
    _wait_until,
)

pytestmark = [pytest.mark.target, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]


def _resolve(*parts: str) -> Path:
    """兼容本机 Tools/Tests/Config 与目标板 tools/tests/config 的大小写。"""
    current = ROOT
    for part in parts:
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue
        matches = (
            [p for p in current.iterdir() if p.name.lower() == part.lower()]
            if current.is_dir()
            else []
        )
        if matches:
            current = matches[0]
            continue
        return candidate
    return current


FIXTURES = _resolve("Tests", "Fixtures", "audio")
CONFIG = _resolve("Config", "target.json")

SYSTEM_PROMPT = "你是戴在用户眼前的智能眼镜助手，用中文简短口语化回答，一般不超过两句话。"


class CountingVlm:
    """统计模型提交次数与 prompt 的 VLM 包装。"""

    def __init__(self, inner: VlmClient) -> None:
        self.inner = inner
        self.calls = 0
        self.prompts: list[str] = []

    def complete_stream(
        self,
        prompt: str,
        *,
        images=(),
        max_tokens: int | None = None,
        timeout_s: float | None = None,
        temperature: float | None = None,
    ):
        """真实产品路径是流式生成（阶段 7）；这里同样统计一次提交。"""
        self.calls += 1
        self.prompts.append(prompt)
        yield from self.inner.complete_stream(
            prompt,
            images=images,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            temperature=temperature,
        )

    def health(self, timeout_s: float | None = None) -> bool:
        return self.inner.health(timeout_s)

    def close(self) -> None:
        self.inner.close()

    def complete(
        self,
        prompt: str,
        *,
        images=(),
        max_tokens: int = 512,
        timeout_s: float = 120.0,
        temperature: float | None = None,
    ) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.inner.complete(
            prompt,
            images=images,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            temperature=temperature,
        )


def _require_server(cfg: AppConfig) -> VlmClient:
    client = VlmClient(cfg.llm, cfg.vlm)
    if not client.health():
        client.close()
        pytest.skip(f"llama-server 未就绪（{cfg.llm.base_url}），先运行 Tools/serve_llama.sh")
    # 验收口径是“普通热态聊天”：先跑一次极短生成，把模型/视觉投影加载
    # 与首帧分配放到计时之外；否则冷启动会被算进被测首声，导致样本抖动。
    try:
        client.complete(
            "你好",
            max_tokens=1,
            timeout_s=max(10.0, float(cfg.llm.request_timeout_s)),
        )
    except Exception:
        pass
    return client


def _make_playlist(cfg: AppConfig, segments: list) -> list:
    """把路径/静音片段展开成 ReplaySource 需要的段列表。"""
    return list(segments)


def _run_scenario(
    cfg: AppConfig,
    segments: list,
    chat: ChatSession,
    evidence_path: Path,
    wait: Callable[[list[dict]], bool],
    timeout_s: float,
) -> tuple[bool, list[dict], RecordingSink]:
    """装配真实组件跑一次回放场景，返回（是否满足条件, 证据记录, sink）。"""
    source = ReplaySource(_make_playlist(cfg, segments), realtime=True)
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
            cfg=cfg, bus=bus, frontend=frontend, sink=sink, tts=tts, chat=chat, evidence=evidence
        )
        sink.start()
        await engine.start()
        # 引擎就绪（进入待机、问候音预合成）后再开始采集，避免启动竞态
        frontend.start()
        try:
            return await _wait_until(lambda: wait(_read_records(evidence_path)), timeout_s=timeout_s)
        finally:
            await engine.stop()
            frontend.stop()
            sink.close()

    ok = asyncio.run(_scenario())
    return ok, _read_records(evidence_path), sink


def _dump(records: list[dict]) -> str:
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)


@pytest.fixture()
def cfg() -> AppConfig:
    import logging as _logging

    _logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    config = AppConfig.from_file(CONFIG)
    if not Path(config.models.root).is_dir():
        pytest.skip(f"目标机模型目录不存在: {config.models.root}")
    config.audio.backend = "replay"
    config.audio.replay_realtime = True
    return config


def test_wake_only_zero_model_submit(cfg: AppConfig, tmp_path: Path) -> None:
    """路径 1：只唤醒、不说话 → 模型提交次数为 0，T01 超时回到待机。"""
    client = _require_server(cfg)
    vlm = CountingVlm(client)
    wake = FIXTURES / "wake" / "tts_xiaobing_nihao.wav"
    if not wake.is_file():
        pytest.skip("缺少唤醒素材 tts_xiaobing_nihao.wav")

    # 唤醒音 1.1s + 13s 静音。T01 默认预算是 10s 等待 + 最多 3×15s 延长（全走到约 55s），
    # 本用例只验证“只唤醒不提交模型、最终回待机”的语义，故缩短计时预算；
    # 真实 10s/15s 预算由 config.TimersConfig 与单元测试覆盖。
    cfg.timers.t01_initial_wait_ms = 2000
    cfg.timers.t01_remind_at_ms = 700
    cfg.timers.t01_extend_ms = 600
    cfg.timers.t01_max_extend = 1
    segments = [wake, _silence(13000)]
    evidence = tmp_path / "wake_only.jsonl"

    ok, records, sink = _run_scenario(
        cfg,
        segments,
        ChatSession(vlm, system_prompt=SYSTEM_PROMPT, max_tokens=96),
        evidence,
        wait=lambda recs: bool(_find(recs, "t01", "idle")),
        timeout_s=40.0,
    )
    assert ok, f"没有回到待机，证据:\n{_dump(records)}"
    assert vlm.calls == 0, f"只唤醒却提交了模型请求: {vlm.calls} 次 / {vlm.prompts}"
    assert not _find(records, "answer", "spoken"), "只唤醒却播报了回答"
    wake_recs = _find(records, "wake", "detected")
    assert wake_recs, "没有唤醒记录"
    assert _find(records, "t01", "remind") or _find(records, "t01", "idle"), "缺少 T01 计时记录"
    print("wake_only:", [r.get("category") + "." + str(r.get("event")) for r in records])


def test_wake_then_question_real_answer(cfg: AppConfig, tmp_path: Path) -> None:
    """路径 2：唤醒 → 问候播完 → 真人提问 → 真实模型回答并完整播报。"""
    client = _require_server(cfg)
    vlm = CountingVlm(client)
    wake = FIXTURES / "wake" / "tts_xiaobing_nihao.wav"
    questions = sorted((FIXTURES / "questions").glob("*询问前方*"))
    if not wake.is_file() or not questions:
        pytest.skip("缺少回放素材（wake/questions）")
    question = questions[0]
    segments = [wake, _silence(2500), question, _silence(1800)]
    evidence = tmp_path / "wake_question.jsonl"

    ok, records, sink = _run_scenario(
        cfg,
        segments,
        ChatSession(vlm, system_prompt=SYSTEM_PROMPT, max_tokens=96, timeout_s=60.0),
        evidence,
        wait=lambda recs: bool(_find(recs, "answer", "spoken")),
        timeout_s=45.0,
    )
    assert ok, f"45s 内没有完成真实问答，证据:\n{_dump(records)}"
    assert vlm.calls == 1, f"应只提交一次模型请求，实际 {vlm.calls} 次"
    answers = _find(records, "answer", "spoken")
    assert answers and answers[-1].get("played") is True, f"回答没有完整播完: {answers}"
    utt = _find(records, "utterance", "final")
    assert utt and len(str(utt[-1].get("text", ""))) >= 2, f"识别结果异常: {utt}"
    llm_done = _find(records, "llm", "done")
    assert llm_done, "缺少真实模型调用记录"
    greet = [r for r in _find(records, "latency", "first_audio") if r.get("speech_kind") == "greeting"]
    assert greet and greet[0]["ms"] < 500, f"问候首声 {greet} 超过 500ms 目标"
    answer_first = [r for r in _find(records, "latency", "first_audio") if r.get("speech_kind") == "answer"]
    assert answer_first, "缺少回答首声时延记录"
    # 流式播报会按短语多次开口：首声取第一次（阶段 7 目标 ≤1s），末次仅用于观察
    assert answer_first[0]["ms"] < 1000, f"回答首声 {answer_first} 超过 1s 目标"
    # 播放顺序：问候在回答之前
    assert len(sink.records) >= 2, f"播放记录不足: {[r['piece_id'] for r in sink.records]}"
    print(
        "wake_question:",
        {
            "utterance": utt[-1].get("text"),
            "answer_first_audio_ms": answer_first[0]["ms"],
            "answer_phrases": len(answer_first),
            "llm_chars": llm_done[-1].get("chars"),
            "played": answers[-1].get("played"),
        },
    )


def test_same_sentence_wake_and_question(cfg: AppConfig, tmp_path: Path) -> None:
    """路径 3：一句“小冰，一加一等于几？”内含唤醒词与问题 → 只受理一次、只提交一次。"""
    client = _require_server(cfg)
    vlm = CountingVlm(client)
    # 用固定夹具而不是运行时合成：Piper 的随机时长会让同一句话每次声学不同，
    # 两字唤醒词在部分合成结果上会漏检（阶段 1 报告已列为风险）。
    wav_path = _resolve(
        "Tests", "Fixtures", "audio", "wake", "same_sentence.wav"
    )

    segments = [wav_path, _silence(1800)]
    evidence = tmp_path / "same_sentence.jsonl"

    ok, records, sink = _run_scenario(
        cfg,
        segments,
        ChatSession(vlm, system_prompt=SYSTEM_PROMPT, max_tokens=96, timeout_s=60.0),
        evidence,
        wait=lambda recs: bool(_find(recs, "answer", "spoken")),
        timeout_s=40.0,
    )
    assert ok, f"同句路径没有完成问答，证据:\n{_dump(records)}"
    assert _find(records, "wake", "detected"), "同句里没有检测到唤醒词"
    assert vlm.calls == 1, f"同句只应提交一次模型请求，实际 {vlm.calls} 次: {vlm.prompts}"
    utt = _find(records, "utterance", "final")
    assert utt and utt[-1].get("text"), f"同句没有识别结果: {utt}"
    answers = _find(records, "answer", "spoken")
    assert answers and answers[-1].get("played") is True, f"回答没有完整播完: {answers}"
    print(
        "same_sentence:",
        {
            "utterance": utt[-1].get("text"),
            "prompt_chars": len(vlm.prompts[0]) if vlm.prompts else 0,
            "played": answers[-1].get("played"),
        },
    )
