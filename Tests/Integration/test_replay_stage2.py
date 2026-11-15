"""阶段 2 软件级验证：回放中真人插话打断播报，停声时延按语音起点计。

链路全部是真实组件：ReplaySource（唤醒 + 问题 1 + 与回答重叠的问题 2）→ SherpaKws →
EnergyVad → StreamingAsr → InteractionEngine → PiperTts → RecordingSink（实时播放、可取消）。
模型用 FakeVlm 固定一段较长回答，保证第二段人声落在播报期间；这验证的是
“插话停声 + 取消证据 + 旧流程让位”的软件行为，不是真机声学。

在目标板上跑：
    source Tools/target_env.sh && trun "python -m pytest -q Tests/Integration/test_replay_stage2.py -s"
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from aieyes.audio.replay_source import ReplaySource
from aieyes.audio.vad import EnergyVad
from aieyes.chat import ChatSession
from aieyes.config import AppConfig
from aieyes.events import EventBus
from aieyes.interaction.engine import InteractionEngine
from aieyes.logging_setup import JsonlEvidenceLogger
from aieyes.speech.asr import StreamingAsr
from aieyes.speech.frontend import SpeechFrontend
from aieyes.speech.kws import SherpaKws
from aieyes.speech.tts import PiperTts

# 复用阶段 1 回放测试的播放替身与证据工具（同一目录，pytest 可直接导入）
from test_replay_voice_chain import (  # noqa: E402
    FakeVlm,
    RecordingSink,
    _find,
    _read_records,
    _silence,
    _wait_until,
)

pytestmark = [pytest.mark.target, pytest.mark.slow]

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "Tests" / "Fixtures" / "audio"
CONFIG = ROOT / "Config" / "target.json"

LONG_REPLY = "前方有一张桌子，桌上有一盏台灯，旁边还有一把椅子，你可以先坐下来慢慢看。"


def test_barge_in_during_playback_stops_audio(tmp_path: Path) -> None:
    cfg = AppConfig.from_file(CONFIG)
    if not Path(cfg.models.root).is_dir():
        pytest.skip(f"目标机模型目录不存在: {cfg.models.root}")
    wake = FIXTURES / "wake" / "tts_xiaobing_nihao.wav"
    questions = sorted((FIXTURES / "questions").glob("*询问前方*"))
    if not wake.is_file() or len(questions) < 2:
        pytest.skip("缺少回放素材（wake/questions×2）")

    cfg.audio.backend = "replay"
    cfg.audio.replay_realtime = True
    cfg.audio.replay_dir = str(FIXTURES / "questions")

    # 时间线：唤醒 → 问候 → 问题 1 → 2600ms 后问题 2（落在回答播报期间）→ 尾部静音
    segments = [wake, _silence(2500), questions[0], _silence(2600), questions[1], _silence(3000)]
    source = ReplaySource(segments, realtime=True)
    kws = SherpaKws(cfg.kws)
    vad = EnergyVad(cfg.vad)
    asr = StreamingAsr(cfg.asr)
    tts = PiperTts(cfg.tts)
    sink = RecordingSink()
    frontend = SpeechFrontend(source, kws=kws, vad=vad, asr=asr)
    vlm = FakeVlm(LONG_REPLY)
    chat = ChatSession(vlm, system_prompt="你是戴在用户眼前的智能眼镜助手，回答简短口语化。", max_tokens=64)
    evidence_path = tmp_path / "stage2.jsonl"
    evidence = JsonlEvidenceLogger(evidence_path)
    engine: InteractionEngine | None = None

    async def _scenario() -> bool:
        nonlocal engine
        bus = EventBus()
        engine = InteractionEngine(
            cfg=cfg, bus=bus, frontend=frontend, sink=sink, tts=tts, chat=chat, evidence=evidence
        )
        sink.start()
        await engine.start()
        frontend.start()
        try:
            return await _wait_until(
                lambda: len(_find(_read_records(evidence_path), "answer", "spoken")) >= 2,
                timeout_s=60.0,
            )
        finally:
            await engine.stop()
            frontend.stop()
            sink.close()

    ok = asyncio.run(_scenario())
    records = _read_records(evidence_path)
    if not ok:
        print("=== engine.state:", engine.state if engine else None)
        for r in records:
            print(r)
    assert ok, f"60s 内没有完成两轮问答，证据: {[(r.get('category'), r.get('event')) for r in records]}"

    barge = _find(records, "barge_in", "silenced")
    assert barge, f"没有记录到插话停声: {[(r.get('category'), r.get('event')) for r in records]}"
    # 停声时延从 VAD 语音起点算起（T 计时口径），目标 ≤ 300ms
    assert barge[0]["ms"] < 300, f"插话停声 {barge[0]['ms']}ms 超过 300ms 目标"
    cancel_paths = _find(records, "cancel", "path")
    assert any(r.get("reason") == "barge_in" for r in cancel_paths), "缺少插话取消的量化证据"
    path = [r for r in cancel_paths if r.get("reason") == "barge_in"][0]
    assert "played_ms" in path and "silenced_ms" in path and "old_task_ms" in path
    # 插话时第一轮回答已播出一部分（played_ms>0 说明打断发生在播报中），播放出口收到取消
    assert path["played_ms"] > 0, "插话发生时回答应已经播出了一部分"
    assert sink.cancel_calls >= 1, "播放出口没有收到取消"

    # 插话后引擎回到倾听，第二段人声仍被识别并正常回答
    answers = _find(records, "answer", "spoken")
    assert answers and answers[-1].get("played") is True, f"插话后第二问没有得到完整回答: {answers}"
    assert vlm.calls == 2, f"两轮问答应各提交一次模型，实际 {vlm.calls} 次"
