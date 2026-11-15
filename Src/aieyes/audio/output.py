"""AudioSink：模块内唯一播放出口。

设计要点（对应验收项）：

* **可信游标** ``played_ms()``：按“已交给设备回调的样本数”计，再减去设备缓冲
  延迟（``stream.latency``），比按墙钟估算更可靠；用于取消/续播与 T12 计时。
* **线程安全**：``play/cancel/stop/played_ms/reference_frame`` 都可在 asyncio
  线程调用，PortAudio 回调线程只持锁搬运样本，不做分配以外的重活。
* **AEC 参考**：``reference_frame()`` 返回最近一帧真正写入设备的音频。
* **统计**：预抢占（overrun）、补零（underrun）、设备 xrun 计数，用于诊断
  “播报卡顿/被插话”。
* 无设备环境可注入 ``stream_factory`` 做单测；TTS 的 22050 Hz 输出在这里
  线性重采样到设备采样率。
"""

from __future__ import annotations

import threading
from typing import Callable

import numpy as np

from aieyes.clock import mono_ms
from aieyes.errors import AudioDeviceError
from aieyes.events import PlaybackFinished, PlaybackStarted
from aieyes.interfaces import SAMPLE_RATE, PlaybackHandle

from .wav import resample_linear

StreamFactory = Callable[[Callable, int, int], object]


def _open_output_stream(callback: Callable, sample_rate: int, block: int, device) -> object:
    try:
        import sounddevice as sd  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - 依赖系统库
        raise AudioDeviceError(
            f"sounddevice 不可用（缺少 libportaudio2？）: {exc}"
        ) from exc
    try:
        return sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=block,
            device=device,
            callback=callback,
        )
    except Exception as exc:
        raise AudioDeviceError(f"打开输出设备失败 device={device!r}: {exc}") from exc


class AudioSink:
    """单通道播放器；同一时刻最多一段音频，新的 ``play`` 会抢占旧的。"""

    def __init__(
        self,
        *,
        device: str | int | None = None,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 20,
        bus=None,
        stream_factory: StreamFactory | None = None,
        output_gain_db: float = 0.0,
        latency_ms: int | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.block = max(1, self.sample_rate * int(frame_ms) // 1000)
        self.device = device
        self.bus = bus
        self._factory: StreamFactory | None = stream_factory
        self._gain = float(10.0 ** (float(output_gain_db) / 20.0))
        self._latency_ms = int(latency_ms or 0)

        self._lock = threading.RLock()
        self._stream = None
        self._pcm = np.zeros(0, dtype=np.float32)
        self._pos = 0
        self._playing = False
        self._piece_id = ""
        self._text = ""
        self._finished = threading.Event()
        self._finished.set()
        self._last_ref: np.ndarray | None = None
        self._piece_counter = 0
        self._stats = {
            "frames_out": 0,
            "pieces_started": 0,
            "pieces_finished": 0,
            "cancelled": 0,
            "preempted": 0,
            "xruns": 0,
            "underrun_frames": 0,
        }

    # ------------------------------------------------------------------ 生命周期
    def start(self) -> None:
        with self._lock:
            if self._stream is not None:
                return
            if self._factory is not None:
                factory = self._factory
            else:
                factory = lambda cb, sr, blk: _open_output_stream(cb, sr, blk, self.device)  # noqa: E731
            stream = factory(self._callback, self.sample_rate, self.block)
            try:
                stream.start()
            except Exception as exc:
                raise AudioDeviceError(f"启动输出流失败: {exc}") from exc
            self._stream = stream
            latency = getattr(stream, "latency", None)
            if self._latency_ms == 0 and isinstance(latency, (int, float)) and latency > 0:
                self._latency_ms = int(round(float(latency) * 1000))

    def stop(self) -> None:
        """停止当前播放并关闭设备（可再次 ``start``/``play``）。"""
        self.cancel()
        with self._lock:
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

    def close(self) -> None:
        self.stop()

    # ------------------------------------------------------------------ 播放
    def play(
        self,
        pcm: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        *,
        piece_id: str = "",
        text: str = "",
    ) -> PlaybackHandle:
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if sample_rate != self.sample_rate and samples.size:
            samples = resample_linear(samples, int(sample_rate), self.sample_rate)
        if self._gain != 1.0:
            samples = samples * self._gain
        self.start()

        with self._lock:
            preempt_evt = self._finish_locked(cancelled=True) if self._playing else None
            if preempt_evt is not None:
                self._stats["preempted"] += 1

            self._piece_counter += 1
            pid = piece_id or f"piece-{self._piece_counter}"
            finished = threading.Event()
            self._pcm = samples
            self._pos = 0
            self._piece_id = pid
            self._text = text
            self._last_ref = None
            self._finished = finished
            self._playing = samples.size > 0
            self._stats["pieces_started"] += 1
            total_ms = int(round(samples.size * 1000 / self.sample_rate))
            if not self._playing:
                finished.set()
            post_started = PlaybackStarted(piece_id=pid, t_ms=mono_ms(), text=text)
            preempt_payload = (
                PlaybackFinished(
                    piece_id=preempt_evt[0],
                    t_ms=mono_ms(),
                    played_ms=preempt_evt[1],
                    total_ms=preempt_evt[2],
                    cancelled=True,
                    text=preempt_evt[3],
                )
                if preempt_evt is not None
                else None
            )
        if preempt_payload is not None:
            self._post(preempt_payload)
        self._post(post_started)

        return PlaybackHandle(
            piece_id=pid,
            total_ms=total_ms,
            text=text,
            _cancel=lambda: self.cancel(pid),
            _wait=finished.wait,
        )

    def cancel(self, piece_id: str | None = None) -> bool:
        """取消当前（或指定 id 且仍为当前）片段；返回是否真的取消。"""
        with self._lock:
            if piece_id is not None and piece_id != self._piece_id:
                return False
            if not self._playing:
                return False
            finished = self._finish_locked(cancelled=True)
            self._stats["cancelled"] += 1
            pid, played, total, text = finished
            payload = PlaybackFinished(
                piece_id=pid,
                t_ms=mono_ms(),
                played_ms=played,
                total_ms=total,
                cancelled=True,
                text=text,
            )
        self._post(payload)
        return True

    # ------------------------------------------------------------------ 游标/参考
    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def played_ms(self) -> int:
        with self._lock:
            return self._played_ms_locked()

    def reference_frame(self) -> np.ndarray | None:
        with self._lock:
            if not self._playing or self._last_ref is None:
                return None
            return np.array(self._last_ref, dtype=np.float32, copy=True)

    def stats(self) -> dict:
        with self._lock:
            return dict(self._stats)

    @property
    def latency_ms(self) -> int:
        return self._latency_ms

    # ------------------------------------------------------------------ 内部
    def _played_ms_locked(self) -> int:
        if self.sample_rate <= 0:
            return 0
        ms = int(round(self._pos * 1000 / self.sample_rate)) - self._latency_ms
        return max(0, min(ms, int(round(self._pcm.size * 1000 / self.sample_rate))))

    def _total_ms_locked(self) -> int:
        return int(round(self._pcm.size * 1000 / self.sample_rate))

    def _finish_locked(self, *, cancelled: bool) -> tuple[str, int, int, str]:
        """结束当前片段（锁内），返回 (piece_id, played_ms, total_ms, text)。"""
        pid, text = self._piece_id, self._text
        played = self._played_ms_locked()
        total = self._total_ms_locked()
        self._playing = False
        self._finished.set()
        if cancelled:
            self._pcm = np.zeros(0, dtype=np.float32)
            self._pos = 0
        else:
            self._stats["pieces_finished"] += 1
        return pid, played, total, text

    def _post(self, event) -> None:
        if self.bus is None:
            return
        try:
            self.bus.post(event)
        except Exception:
            pass

    def _callback(self, outdata, frames, time_info, status) -> None:  # PortAudio 线程
        n = int(frames)
        try:
            out = outdata if outdata.ndim == 1 else outdata[:, 0]
        except AttributeError:
            out = np.asarray(outdata).reshape(-1)
        with self._lock:
            if status is not None:
                self._stats["xruns"] += 1
            out.fill(0.0)
            if not self._playing:
                return
            remaining = int(self._pcm.size) - self._pos
            take = min(n, remaining)
            if take > 0:
                chunk = self._pcm[self._pos : self._pos + take]
                out[:take] = chunk
                self._last_ref = chunk.copy()
                self._pos += take
                self._stats["frames_out"] += take
            if take < n:
                self._stats["underrun_frames"] += n - take
            if self._pos >= int(self._pcm.size):
                pid, played, total, text = self._finish_locked(cancelled=False)
                payload = PlaybackFinished(
                    piece_id=pid,
                    t_ms=mono_ms(),
                    played_ms=played,
                    total_ms=total,
                    cancelled=False,
                    text=text,
                )
            else:
                payload = None
        if payload is not None:
            self._post(payload)


__all__ = ["AudioSink", "StreamFactory"]
