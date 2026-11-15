"""audio 单测共用的小工具（合成信号 / 切帧）。"""

from __future__ import annotations

import numpy as np

from aieyes.interfaces import SAMPLE_RATE, AudioFrame

FRAME_SAMPLES = SAMPLE_RATE * 20 // 1000  # 320


def silence(ms: int, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(int(round(ms * sample_rate / 1000)), dtype=np.float32)


def tone(ms: int, freq: float = 440.0, amp: float = 0.2, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    n = int(round(ms * sample_rate / 1000))
    t = np.arange(n, dtype=np.float64) / sample_rate
    return (amp * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)


def frames(
    samples: np.ndarray,
    *,
    sample_rate: int = SAMPLE_RATE,
    frame_ms: int = 20,
    t0_ms: int = 0,
) -> list[AudioFrame]:
    """把连续音频切成定长帧（末帧补零到整帧）。"""
    n = sample_rate * frame_ms // 1000
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    out: list[AudioFrame] = []
    seq = 0
    for start in range(0, len(samples), n):
        chunk = samples[start : start + n]
        if chunk.size < n:
            chunk = np.pad(chunk, (0, n - chunk.size))
        out.append(
            AudioFrame(
                samples=chunk.copy(),
                t_start_ms=int(t0_ms + seq * frame_ms),
                seq=seq,
            )
        )
        seq += 1
    return out
