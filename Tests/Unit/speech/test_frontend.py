"""端到端：ReplaySource → 唤醒（合成“小冰”）→ 静音 → 问题 wav → utterance。"""

from __future__ import annotations

import threading

import numpy as np
import pytest

from speech_helpers import FIXTURES, has_cjk

from aieyes.audio.replay_source import ReplaySource
from aieyes.audio.vad import EnergyVad
from aieyes.audio.wav import read_wav
from aieyes.config import VadConfig
from aieyes.interfaces import SAMPLE_RATE
from aieyes.speech.asr import StreamingAsr
from aieyes.speech.frontend import SpeechFrontend
from aieyes.speech.kws import SherpaKws


def test_frontend_e2e_wake_then_utterance(models):
    wake_wav = FIXTURES / "wake" / "tts_xiaobing.wav"
    question_wav = models.resolve(models.asr_dir) / "test_wavs" / "0.wav"
    if not wake_wav.is_file() or not question_wav.is_file():
        pytest.skip("缺少 e2e wav 素材")

    wake, _ = read_wav(wake_wav, target_rate=SAMPLE_RATE)
    question, _ = read_wav(question_wav, target_rate=SAMPLE_RATE)
    silence = lambda ms: np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)  # noqa: E731

    source = ReplaySource(
        [wake, silence(900), question, silence(1000)],
        sample_rate=SAMPLE_RATE,
        realtime=False,
    )
    kws = SherpaKws(models=models)
    asr = StreamingAsr(models=models)
    frontend = SpeechFrontend(source, kws=kws, vad=EnergyVad(VadConfig()), asr=asr, preroll_ms=300)

    events: list[tuple] = []
    done = threading.Event()

    def on_wake(keyword: str, t_ms: int) -> None:
        events.append(("wake", keyword, int(t_ms)))
        frontend.set_listening(True)

    def on_speech_start(t_ms: int) -> None:
        events.append(("speech_start", int(t_ms)))

    def on_utterance(utterance) -> None:
        events.append(("utterance", utterance))
        done.set()

    frontend.on_wake = on_wake
    frontend.on_speech_start = on_speech_start
    frontend.on_utterance = on_utterance

    try:
        frontend.start()
        assert source.wait(120.0), "回放未在 120s 内结束"
        assert done.wait(30.0), f"未在超时内收到 utterance，事件={events}"
    finally:
        frontend.stop()
        kws.close()
        asr.close()

    kinds = [event[0] for event in events]
    assert "wake" in kinds, kinds
    assert "speech_start" in kinds, kinds
    assert "utterance" in kinds, kinds
    assert kinds.index("wake") < kinds.index("speech_start") < kinds.index("utterance"), kinds
    assert len([kind for kind in kinds if kind == "wake"]) == 1

    wake_events = [event for event in events if event[0] == "wake"]
    assert "小冰" in wake_events[0][1]

    utterance = [event[1] for event in events if event[0] == "utterance"][0]
    assert utterance.text.strip(), utterance
    assert has_cjk(utterance.text), utterance.text
    assert utterance.t_audio_end_ms >= utterance.t_audio_start_ms
    if getattr(source, "realtime", True):
        assert utterance.t_final_ms >= utterance.t_audio_end_ms
    print(
        f"[frontend] events={kinds} text={utterance.text!r} "
        f"audio=[{utterance.t_audio_start_ms},{utterance.t_audio_end_ms}]ms "
        f"final={utterance.t_final_ms}ms"
    )
