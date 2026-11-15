"""WAV 读写与重采样工具。

约定与 ``interfaces.SAMPLE_RATE`` 一致：内部一律 ``float32``、mono、[-1, 1]。
读取侧兼容 PCM16 / PCM24 / PCM32 / IEEE float32（模型自带的 test_wavs 多是
PCM16），写入侧默认 PCM16（兼容性最好），也可写 float32。
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import BinaryIO

import numpy as np

SAMPLE_RATE = 16_000

_WAVE_FORMAT_PCM = 0x0001
_WAVE_FORMAT_IEEE_FLOAT = 0x0003
_WAVE_FORMAT_EXTENSIBLE = 0xFFFE


class WavError(ValueError):
    """WAV 文件格式错误。"""


def _read_fmt(fmt: bytes) -> tuple[int, int, int, int, int]:
    """返回 (format_tag, channels, sample_rate, bits, block_align)。"""
    if len(fmt) < 16:
        raise WavError(f"fmt chunk too short: {len(fmt)}")
    tag, channels, rate, _byte_rate, block_align, bits = struct.unpack("<HHIIHH", fmt[:16])
    if tag == _WAVE_FORMAT_EXTENSIBLE and len(fmt) >= 40:
        # SubFormat GUID 的前两个字节与 format tag 相同
        tag = struct.unpack("<H", fmt[24:26])[0]
    if channels <= 0 or rate <= 0 or bits <= 0:
        raise WavError(f"invalid fmt: tag={tag} ch={channels} rate={rate} bits={bits}")
    return tag, channels, rate, bits, block_align


def _decode_pcm(data: bytes, tag: int, bits: int, channels: int) -> np.ndarray:
    if tag == _WAVE_FORMAT_IEEE_FLOAT:
        if bits == 32:
            arr = np.frombuffer(data, dtype="<f4")
        elif bits == 64:
            arr = np.frombuffer(data, dtype="<f8").astype(np.float32)
        else:
            raise WavError(f"unsupported float bits: {bits}")
        return arr.copy()
    if tag != _WAVE_FORMAT_PCM:
        raise WavError(f"unsupported wav format tag: {tag}")

    if bits == 8:  # unsigned
        arr = (np.frombuffer(data, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        return arr
    if bits == 16:
        return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0
    if bits == 24:
        raw = np.frombuffer(data, dtype=np.uint8)
        usable = (raw.size // 3) * 3
        tri = raw[:usable].reshape(-1, 3).astype(np.int32)
        val = tri[:, 0] | (tri[:, 1] << 8) | (tri[:, 2] << 16)
        val = np.where(val >= 1 << 23, val - (1 << 24), val)
        return val.astype(np.float32) / float(1 << 23)
    if bits == 32:
        return np.frombuffer(data, dtype="<i4").astype(np.float32) / 2147483648.0
    raise WavError(f"unsupported pcm bits: {bits}")


def read_wav(
    path: str | Path | BinaryIO,
    *,
    target_rate: int | None = SAMPLE_RATE,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """读取 WAV，返回 ``(float32 [-1,1] 数组, 原始采样率)``。

    多声道默认按算术平均混成单声道；``target_rate`` 非 None 且与文件不同
    时做线性重采样。返回值第二项是**重采样后**的采样率。
    """
    if isinstance(path, (str, Path)):
        raw = Path(path).read_bytes()
    else:
        raw = path.read()

    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise WavError("not a RIFF/WAVE file")

    pos = 12
    fmt: bytes | None = None
    data: bytes | None = None
    while pos + 8 <= len(raw):
        cid = raw[pos : pos + 4]
        (size,) = struct.unpack("<I", raw[pos + 4 : pos + 8])
        body = raw[pos + 8 : pos + 8 + size]
        if cid == b"fmt ":
            fmt = body
        elif cid == b"data":
            data = body
            break
        pos += 8 + size + (size & 1)
    if fmt is None:
        raise WavError("missing fmt chunk")
    if data is None:
        raise WavError("missing data chunk")

    tag, channels, rate, bits, _block_align = _read_fmt(fmt)
    samples = _decode_pcm(data, tag, bits, channels).reshape(-1, channels)
    if mono and channels > 1:
        samples = samples.mean(axis=1)
    elif not mono:
        pass
    samples = np.ascontiguousarray(samples, dtype=np.float32).reshape(-1)

    rate_out = rate
    if target_rate is not None and target_rate != rate:
        samples = resample_linear(samples, rate, target_rate)
        rate_out = target_rate
    return samples, rate_out


def write_wav(
    path: str | Path,
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    subtype: str = "pcm16",
) -> None:
    """写单声道 WAV。``subtype`` 为 ``pcm16`` 或 ``float32``。"""
    pcm = np.asarray(samples, dtype=np.float32).reshape(-1)
    if subtype == "float32":
        tag, bits = _WAVE_FORMAT_IEEE_FLOAT, 32
        payload = np.clip(pcm, -1.0, 1.0).astype("<f4").tobytes()
    elif subtype == "pcm16":
        tag, bits = _WAVE_FORMAT_PCM, 16
        payload = np.clip(pcm * 32767.0, -32768, 32767).astype("<i2").tobytes()
    else:
        raise WavError(f"unsupported subtype: {subtype}")

    channels = 1
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    fmt = struct.pack(
        "<HHIIHH", tag, channels, sample_rate, byte_rate, block_align, bits
    )
    chunks = b"fmt " + struct.pack("<I", len(fmt)) + fmt
    chunks += b"data" + struct.pack("<I", len(payload)) + payload
    header = b"RIFF" + struct.pack("<I", 4 + len(chunks)) + b"WAVE"
    Path(path).write_bytes(header + chunks)


def resample_linear(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """线性插值重采样（语音回放/测试足够；不追求抗混叠）。"""
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if src_rate == dst_rate or x.size == 0:
        return x.copy()
    n_out = int(round(x.size * dst_rate / float(src_rate)))
    if n_out <= 0:
        return np.zeros(0, dtype=np.float32)
    xp = np.arange(x.size, dtype=np.float64)
    xs = np.linspace(0.0, x.size - 1, n_out, dtype=np.float64)
    return np.interp(xs, xp, x).astype(np.float32)


def concat_segments(
    segments: list[np.ndarray], *, gap_ms: int = 0, sample_rate: int = SAMPLE_RATE
) -> np.ndarray:
    """把多段音频用静音拼接起来（测试/回放用）。"""
    gap = np.zeros(int(round(gap_ms * sample_rate / 1000)), dtype=np.float32)
    parts: list[np.ndarray] = []
    for i, seg in enumerate(segments):
        if i:
            parts.append(gap)
        parts.append(np.asarray(seg, dtype=np.float32).reshape(-1))
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)
