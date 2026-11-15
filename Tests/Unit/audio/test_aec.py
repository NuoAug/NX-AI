"""NLMS AEC 单测：合成“参考信号 + 延迟衰减回声 + 噪声”验证 ERLE。"""

from __future__ import annotations

import numpy as np

from aieyes.audio.aec import NlmsAec
from aieyes.interfaces import SAMPLE_RATE


def _synthetic_echo(
    *,
    seed: int = 7,
    n_frames: int = 400,
    block: int = 320,
    delay: int = 640,
    gain: float = 0.5,
    noise: float = 0.002,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ref = rng.standard_normal((n_frames, block)).astype(np.float32)
    ref *= 0.2 / float(np.sqrt(np.mean(ref**2)))  # 参考信号 RMS=0.2
    flat = ref.reshape(-1)
    echo = np.zeros_like(flat)
    echo[delay:] = gain * flat[:-delay]
    mic = echo.reshape(n_frames, block) + noise * rng.standard_normal((n_frames, block)).astype(
        np.float32
    )
    return mic.astype(np.float32), ref


def test_erle_at_least_10db():
    mic, ref = _synthetic_echo()
    aec = NlmsAec(filter_ms=200, mu=0.6)
    out = aec.process_frames(mic, ref)
    assert out.shape == mic.shape
    assert np.all(np.isfinite(out))
    erle = aec.erle_db
    assert erle >= 10.0, f"ERLE={erle:.2f} dB，参考信号/回声模型可能未收敛"


def test_without_reference_output_passthrough():
    aec = NlmsAec()
    x = np.random.default_rng(1).standard_normal(320).astype(np.float32)
    out = aec.process(x, None)
    assert np.allclose(out, x, atol=1e-6)


def test_streaming_matches_frames_processing():
    mic, ref = _synthetic_echo(seed=11, n_frames=120)
    aec_a = NlmsAec()
    aec_b = NlmsAec()
    out_a = aec_a.process_frames(mic, ref)
    out_b = np.stack([aec_b.process(mic[i], ref[i]) for i in range(mic.shape[0])])
    assert np.allclose(out_a, out_b, atol=1e-6)


def test_reset_clears_erle_and_weights():
    mic, ref = _synthetic_echo(n_frames=60)
    aec = NlmsAec()
    aec.process_frames(mic, ref)
    assert aec.erle_db > 0.0
    aec.reset()
    assert aec.erle_db == 0.0
    assert float(np.max(np.abs(aec.weights))) == 0.0
