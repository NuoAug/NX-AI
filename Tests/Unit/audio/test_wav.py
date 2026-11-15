"""WAV 读写 / 重采样 / 拼接单测。"""

from __future__ import annotations

import numpy as np
import pytest

from audio_helpers import tone

from aieyes.audio.wav import (
    WavError,
    concat_segments,
    read_wav,
    resample_linear,
    write_wav,
)
from aieyes.interfaces import SAMPLE_RATE


def test_pcm16_roundtrip(tmp_path):
    x = tone(500, amp=0.5)
    path = tmp_path / "tone16.wav"
    write_wav(path, x)
    y, sr = read_wav(path)
    assert sr == SAMPLE_RATE
    assert y.dtype == np.float32
    assert y.shape == x.shape
    assert float(np.max(np.abs(y - x))) < 1e-3


def test_float32_roundtrip(tmp_path):
    x = tone(200, amp=0.3)
    path = tmp_path / "tone32.wav"
    write_wav(path, x, subtype="float32")
    y, sr = read_wav(path)
    assert sr == SAMPLE_RATE
    assert y.shape == x.shape
    assert float(np.max(np.abs(y - x))) < 1e-6


def test_resample_linear_length():
    x = np.ones(22050, dtype=np.float32)
    y = resample_linear(x, 22050, SAMPLE_RATE)
    assert y.shape == (SAMPLE_RATE,)
    assert float(np.max(np.abs(y - 1.0))) < 1e-3


def test_concat_segments_with_gap():
    a = tone(100)
    b = tone(100, freq=880.0)
    joined = concat_segments([a, b], gap_ms=100)
    assert joined.shape == (len(a) + len(b) + 1600,)
    assert float(np.max(np.abs(joined[: len(a)] - a))) < 1e-6
    assert float(np.max(np.abs(joined[len(a) : len(a) + 1600]))) == 0.0


def test_bad_wav_raises(tmp_path):
    path = tmp_path / "bad.wav"
    path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt ")
    with pytest.raises(WavError):
        read_wav(path)
