"""speech 单测辅助：把 wav 切成 AudioFrame、收集 KWS 命中。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aieyes.audio.wav import read_wav
from aieyes.interfaces import SAMPLE_RATE, AudioFrame

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "Tests" / "Fixtures" / "audio"
FRAME = 320


def frames_from_samples(samples: np.ndarray) -> list[AudioFrame]:
    samples = np.asarray(samples, dtype=np.float32).reshape(-1)
    out: list[AudioFrame] = []
    for i in range(0, len(samples), FRAME):
        chunk = samples[i : i + FRAME]
        if chunk.size < FRAME:
            chunk = np.pad(chunk, (0, FRAME - chunk.size))
        out.append(
            AudioFrame(samples=chunk.copy(), t_start_ms=int(i * 1000 / SAMPLE_RATE), seq=i // FRAME)
        )
    return out


def frames_from_wav(path, *, pad_ms: int = 0) -> list[AudioFrame]:
    samples, _ = read_wav(path, target_rate=SAMPLE_RATE)
    if pad_ms:
        samples = np.concatenate(
            [samples, np.zeros(int(SAMPLE_RATE * pad_ms / 1000), dtype=np.float32)]
        )
    return frames_from_samples(samples)


def collect_hits(kws, frames: list[AudioFrame]) -> list[tuple[str, int | None]]:
    hits: list[tuple[str, int | None]] = []
    for frame in frames:
        keyword = kws.accept(frame)
        if keyword:
            hits.append((keyword, frame.t_start_ms))
    tail = kws.flush()
    if tail:
        hits.append((tail, None))
    return hits


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text or "")
