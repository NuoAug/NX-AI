"""能量 VAD 单测（合成语音/静音）。"""

from __future__ import annotations

import numpy as np

from audio_helpers import frames, silence, tone

from aieyes.audio.vad import EnergyVad
from aieyes.config import VadConfig
from aieyes.interfaces import SAMPLE_RATE


def _run(vad: EnergyVad, samples: np.ndarray):
    segs = []
    for frame in frames(samples):
        seg = vad.accept(frame)
        if seg is not None:
            segs.append(seg)
    return segs


def test_detects_burst_and_ends_on_silence():
    cfg = VadConfig(threshold=0.5, min_speech_ms=200, min_silence_ms=500, preroll_ms=300)
    vad = EnergyVad(cfg)
    speech = tone(400, amp=0.2)  # RMS≈0.14 >> 门限
    samples = np.concatenate([silence(400), speech, silence(800)])
    segs = _run(vad, samples)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.cause == "vad_end"
    # preroll 300ms：段起点最多提前 300ms
    assert 100 <= seg.t_start_ms <= 400
    assert seg.t_end_ms >= 800
    expected_ms = len(seg.samples) * 1000 / SAMPLE_RATE
    assert abs(expected_ms - (seg.t_end_ms - seg.t_start_ms)) < 25
    assert vad.in_speech is False


def test_click_too_short_is_rejected():
    cfg = VadConfig(threshold=0.5, min_speech_ms=200, min_silence_ms=500)
    vad = EnergyVad(cfg)
    samples = np.concatenate([silence(300), tone(60, amp=0.3), silence(600)])
    assert _run(vad, samples) == []
    assert vad.in_speech is False


def test_max_utterance_cut():
    cfg = VadConfig(
        threshold=0.5,
        min_speech_ms=200,
        min_silence_ms=500,
        max_utterance_ms=600,
        preroll_ms=100,
    )
    vad = EnergyVad(cfg)
    segs = _run(vad, tone(1500, amp=0.2))
    assert len(segs) >= 1
    assert segs[0].cause == "max_len"
    assert (segs[0].t_end_ms - segs[0].t_start_ms) <= 800


def test_reset_clears_state():
    cfg = VadConfig(threshold=0.5, min_speech_ms=200, min_silence_ms=500)
    vad = EnergyVad(cfg)
    for frame in frames(tone(400, amp=0.2)):
        vad.accept(frame)
    assert vad.in_speech is True
    vad.reset()
    assert vad.in_speech is False
    assert vad.accept(frames(silence(200))[0]) is None


def test_threshold_maps_to_open_level():
    low = EnergyVad(VadConfig(threshold=0.5))
    high = EnergyVad(VadConfig(threshold=2.0))
    assert abs(low.open_level - 0.01) < 1e-9
    assert high.open_level > low.open_level
    assert low.close_level < low.open_level
