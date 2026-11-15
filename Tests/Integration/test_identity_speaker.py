"""阶段 5 真机软件闭环：真实 campplus 声纹核验接入交互状态机。

覆盖方案阶段 5（L117–120）软件层：本段 VAD 音频 → 本地声纹核验 → 身份记录；
免唤醒候选与资料权限在单测里用假身份覆盖，本文件验证真实模型与真实语音链路。

运行：
    source Tools/target_env.sh && trun "python -m pytest -q Tests/Integration/test_identity_speaker.py -s"
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from aieyes.audio.replay_source import ReplaySource
from aieyes.audio.vad import EnergyVad
from aieyes.audio.wav import read_wav
from aieyes.chat import ChatSession
from aieyes.config import AppConfig
from aieyes.events import EventBus
from aieyes.identity import IdentityManager, IdentityStatus, SpeakerGate
from aieyes.interaction.engine import InteractionEngine
from aieyes.logging_setup import JsonlEvidenceLogger
from aieyes.speech.asr import StreamingAsr
from aieyes.speech.frontend import SpeechFrontend
from aieyes.speech.kws import SherpaKws
from aieyes.speech.tts import PiperTts

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
WAKE = FIXTURES / "audio" / "wake" / "tts_xiaobing_nihao.wav"
QUESTIONS = FIXTURES / "audio" / "questions"
SPEAKERS = FIXTURES / "audio" / "speaker"


class FakeVlm:
    def __init__(self, reply: str = "好，我记住了。") -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt, *, images=(), max_tokens=512, timeout_s=120.0, temperature=None):
        self.prompts.append(prompt)
        return self.reply


def _cfg() -> AppConfig:
    return AppConfig.from_file(CONFIG)


def _require_speaker_model(cfg: AppConfig) -> None:
    onnx = cfg.models.resolve(cfg.models.speaker_onnx)
    if not onnx.is_file():
        pytest.skip(f"缺少声纹模型: {onnx}")


def _require_wavs(*paths: Path) -> None:
    for path in paths:
        if not path.is_file():
            pytest.skip(f"缺少音频夹具: {path}")


def _speaker_files(prefix: str) -> list[Path]:
    return sorted(SPEAKERS.glob(f"{prefix}_*.wav"))


# ---------------------------------------------------------------------------
# 1) 真实声纹门：本人通过、他人被拒
# ---------------------------------------------------------------------------
def test_real_speaker_gate_verifies_owner_and_rejects_others() -> None:
    cfg = _cfg()
    _require_speaker_model(cfg)
    gate = SpeakerGate(cfg=cfg.speaker, models_root=cfg.models.root)
    try:
        files = _speaker_files("he")
        _require_wavs(*files)
        enrolled = gate.enroll_files(files[:3])
        assert enrolled >= 1, "本人样本应能注册"
        assert gate.enrolled is True

        owner, _ = read_wav(files[0], target_rate=16000)
        owner_result = gate.verify(owner)
        assert owner_result.status is IdentityStatus.VERIFIED, owner_result
        assert owner_result.score >= cfg.speaker.threshold
        assert owner_result.profile_version.startswith("campplus-zh-cn-16k@v1")

        others = {}
        for who in ("me", "she"):
            paths = _speaker_files(who)
            _require_wavs(*paths)
            samples, _sr = read_wav(paths[0], target_rate=16000)
            result = gate.verify(samples)
            others[who] = round(result.score, 4)
            assert result.status is IdentityStatus.MISMATCH, (who, result)
        print(
            "speaker_gate",
            {
                "owner_score": round(owner_result.score, 4),
                "others": others,
                "threshold": cfg.speaker.threshold,
                "profile": owner_result.profile_version,
            },
        )
    finally:
        gate.close()


# ---------------------------------------------------------------------------
# 2) 真实语音链路：唤醒 → 提问 → 引擎记录真实声纹结论
# ---------------------------------------------------------------------------
def test_engine_records_real_speaker_verification(tmp_path: Path) -> None:
    cfg = _cfg()
    _require_speaker_model(cfg)
    question_wav = QUESTIONS / "01-he-测试自身智能状态.wav"
    _require_wavs(WAKE, question_wav)

    gate = SpeakerGate(cfg=cfg.speaker, models_root=cfg.models.root)
    gate.enroll_files(_speaker_files("he")[:3])
    identity = IdentityManager(gate, cfg.timers, topic_window_ms=cfg.timers.t21_topic_window_ms)
    evidence_path = tmp_path / "identity.jsonl"

    async def scenario() -> None:
        sink = RecordingSink()
        tts = PiperTts(cfg.tts)
        segments = [WAKE, _silence(2500), question_wav, _silence(1800)]
        source = ReplaySource(segments, sample_rate=cfg.audio.sample_rate, realtime=True)
        frontend = SpeechFrontend(
            source,
            kws=SherpaKws(cfg.kws),
            vad=EnergyVad(cfg.vad),
            asr=StreamingAsr(cfg.asr),
        )
        bus = EventBus()
        engine = InteractionEngine(
            cfg=cfg,
            bus=bus,
            frontend=frontend,
            sink=sink,
            tts=tts,
            chat=ChatSession(FakeVlm()),
            identity=identity,
            evidence=JsonlEvidenceLogger(evidence_path),
        )
        sink.start()
        await engine.start()
        frontend.start()
        try:
            ok = await _wait_until(
                lambda: bool(_find(_read_records(evidence_path), "identity", "verify")),
                timeout_s=45.0,
            )
            assert ok, "45s 内没有身份核验记录"
        finally:
            frontend.stop()
            await engine.stop()
            sink.close()
            tts.close()

    asyncio.run(scenario())
    verify = _find(_read_records(evidence_path), "identity", "verify")[0]
    assert verify["status"] == "verified", verify
    assert verify["score"] >= cfg.speaker.threshold, verify
    assert "profile" in verify and verify["profile"].startswith("campplus")
    print("identity_verify", verify)


# ---------------------------------------------------------------------------
# 3) 话题窗口 + 候选准入（用真实核验结果驱动 IdentityManager）
# ---------------------------------------------------------------------------
def test_real_verification_drives_topic_candidate_decision() -> None:
    cfg = _cfg()
    _require_speaker_model(cfg)
    gate = SpeakerGate(cfg=cfg.speaker, models_root=cfg.models.root)
    try:
        gate.enroll_files(_speaker_files("he")[:3])
        manager = IdentityManager(gate, cfg.timers, topic_window_ms=cfg.timers.t21_topic_window_ms)
        owner, _ = read_wav(_speaker_files("he")[0], target_rate=16000)
        result = manager.verify(owner)
        manager.note(result)
        assert manager.access_ok() is True
        manager.open_topic(now_ms=1000)
        decision = manager.decide_candidate(
            result, "这个多少钱？", directed=True, now_ms=2000
        )
        assert decision.action == "accept", decision
        other, _ = read_wav(_speaker_files("me")[0], target_rate=16000)
        other_result = manager.verify(other)
        denied = manager.decide_candidate(other_result, "这个多少钱？", directed=True, now_ms=3000)
        assert denied.action == "ignore" and denied.reason == "identity_mismatch", denied
    finally:
        gate.close()
