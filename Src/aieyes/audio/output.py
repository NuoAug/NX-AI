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
* **卡死看门狗**：设备中途失联时 PortAudio 回调不再推进 ``_pos``，
  ``play()`` 返回的句柄会永远等不到结束。``_watchdog_loop`` 在连续
  ``stall_timeout_ms`` 内看不到 ``frames_out`` 增长时结束该片段
  （``stats['stalled']++``、``wait()`` 返回 False），避免上层播报协程永久挂起。
* **结束原因分层可辨**：设备卡死与用户主动停止都必须带上共同字段
  ``stop_reason``，取值 ``completed`` / ``user_cancel`` / ``preempted`` /
  ``playback_stalled``（见 ``STOP_*``）。卡死额外补一条
  ``DeviceFault(device="audio_out")``，让 T14 故障恢复与证据记录走既有通道，
  不得落成“用户主动打断”。事件、句柄（``handle.stop_reason``）与
  ``stats`` 三层都可判定。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from aieyes.clock import mono_ms
from aieyes.errors import AudioDeviceError
from aieyes.events import DeviceFault, PlaybackFinished, PlaybackStarted
from aieyes.interfaces import SAMPLE_RATE, PlaybackHandle

from .wav import resample_linear

StreamFactory = Callable[[Callable, int, int], object]

_WATCHDOG_TICK_S = 0.05

# 片段结束原因（``PlaybackStopped.stop_reason`` ／ ``handle.stop_reason`` 取值）
STOP_COMPLETED = "completed"
STOP_USER_CANCEL = "user_cancel"
STOP_PREEMPTED = "preempted"
STOP_STALLED = "playback_stalled"

_STOP_REASON_LIMIT = 32


@dataclass(frozen=True, slots=True)
class PlaybackStopped(PlaybackFinished):
    """带 ``stop_reason`` 的 ``PlaybackFinished``；父类字段与语义保持不变。

    读方用 ``getattr(ev, "stop_reason", "")`` 兼容旧事件（空 = 未标注）。
    """

    stop_reason: str = ""


@dataclass(slots=True)
class SinkPlaybackHandle(PlaybackHandle):
    """播放句柄：额外暴露 ``stop_reason``（``""`` 表示仍在播）。"""

    _reason: Callable[[], str] | None = field(default=None, repr=False)

    @property
    def stop_reason(self) -> str:
        return self._reason() if self._reason is not None else ""


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
        stall_timeout_ms: int = 4000,
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
        self._stall_timeout_ms = max(0, int(stall_timeout_ms))
        self._watchdog: threading.Thread | None = None
        self._stalled_piece: str = ""
        self._stop_reasons: dict[str, str] = {}
        self._stats = {
            "frames_out": 0,
            "pieces_started": 0,
            "pieces_finished": 0,
            "cancelled": 0,
            "preempted": 0,
            "xruns": 0,
            "underrun_frames": 0,
            "stalled": 0,
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
            preempt_evt = (
                self._finish_locked(cancelled=True, reason=STOP_PREEMPTED)
                if self._playing
                else None
            )
            if preempt_evt is not None:
                self._stats["preempted"] += 1

            self._piece_counter += 1
            pid = piece_id or f"piece-{self._piece_counter}"
            finished = threading.Event()
            self._pcm = samples
            self._pos = 0
            self._piece_id = pid
            self._text = text
            self._stalled_piece = ""
            self._last_ref = None
            self._finished = finished
            self._playing = samples.size > 0
            self._stats["pieces_started"] += 1
            total_ms = int(round(samples.size * 1000 / self.sample_rate))
            if not self._playing:
                finished.set()
            post_started = PlaybackStarted(piece_id=pid, t_ms=mono_ms(), text=text)
            preempt_payload = (
                PlaybackStopped(
                    piece_id=preempt_evt[0],
                    t_ms=mono_ms(),
                    played_ms=preempt_evt[1],
                    total_ms=preempt_evt[2],
                    cancelled=True,
                    text=preempt_evt[3],
                    stop_reason=STOP_PREEMPTED,
                )
                if preempt_evt is not None
                else None
            )
        if preempt_payload is not None:
            self._post(preempt_payload)
        self._post(post_started)

        self._ensure_watchdog()

        return SinkPlaybackHandle(
            piece_id=pid,
            total_ms=total_ms,
            text=text,
            _cancel=lambda: self.cancel(pid),
            _wait=lambda timeout=None: self._wait_piece(pid, finished, timeout),
            _reason=lambda: self.stop_reason(pid),
        )

    def cancel(self, piece_id: str | None = None) -> bool:
        """取消当前（或指定 id 且仍为当前）片段；返回是否真的取消。"""
        with self._lock:
            if piece_id is not None and piece_id != self._piece_id:
                return False
            if not self._playing:
                return False
            finished = self._finish_locked(cancelled=True, reason=STOP_USER_CANCEL)
            self._stats["cancelled"] += 1
            pid, played, total, text = finished
            payload = PlaybackStopped(
                piece_id=pid,
                t_ms=mono_ms(),
                played_ms=played,
                total_ms=total,
                cancelled=True,
                text=text,
                stop_reason=STOP_USER_CANCEL,
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

    def stop_reason(self, piece_id: str) -> str:
        """该片段已记录的结束原因（``STOP_*``）；未知/仍在播返回空串。"""
        with self._lock:
            return self._stop_reasons.get(piece_id, "")

    @property
    def latency_ms(self) -> int:
        return self._latency_ms

    # ------------------------------------------------------------------ 内部
    def _remember_stop_locked(self, piece_id: str, reason: str) -> None:
        self._stop_reasons[piece_id] = reason
        while len(self._stop_reasons) > _STOP_REASON_LIMIT:
            self._stop_reasons.pop(next(iter(self._stop_reasons)))

    def _played_ms_locked(self) -> int:
        if self.sample_rate <= 0:
            return 0
        ms = int(round(self._pos * 1000 / self.sample_rate)) - self._latency_ms
        return max(0, min(ms, int(round(self._pcm.size * 1000 / self.sample_rate))))

    def _total_ms_locked(self) -> int:
        return int(round(self._pcm.size * 1000 / self.sample_rate))

    def _finish_locked(
        self, *, cancelled: bool, reason: str = ""
    ) -> tuple[str, int, int, str]:
        """结束当前片段（锁内），返回 (piece_id, played_ms, total_ms, text)。

        ``reason`` 记入 ``stop_reason`` 表，供 ``handle.stop_reason`` 与证据读取。
        """
        pid, text = self._piece_id, self._text
        played = self._played_ms_locked()
        total = self._total_ms_locked()
        if reason:
            self._remember_stop_locked(pid, reason)
        self._playing = False
        self._finished.set()
        if cancelled:
            self._pcm = np.zeros(0, dtype=np.float32)
            self._pos = 0
        else:
            self._stats["pieces_finished"] += 1
        return pid, played, total, text

    def _wait_piece(self, piece_id: str, event: threading.Event, timeout: float | None = None) -> bool:
        """等本片段结束；被看门狗判定为卡死时返回 False（按中断处理）。"""
        if not event.wait(timeout):
            return False
        with self._lock:
            return self._stalled_piece != piece_id

    def _ensure_watchdog(self) -> None:
        if self._stall_timeout_ms <= 0:
            return
        with self._lock:
            if self._watchdog is not None and self._watchdog.is_alive():
                return
            self._watchdog = threading.Thread(
                target=self._watchdog_loop, name="aieyes-output-watchdog", daemon=True
            )
            self._watchdog.start()

    def _watchdog_loop(self) -> None:
        """设备失联看门狗：播放中长时间没有样本被写出即结束该片段。"""
        timeout_s = self._stall_timeout_ms / 1000.0
        last_frames = -1
        last_change = time.monotonic()
        while True:
            time.sleep(_WATCHDOG_TICK_S)
            payload = None
            with self._lock:
                if not self._playing:
                    self._watchdog = None
                    return
                frames = self._stats["frames_out"]
                now = time.monotonic()
                if frames != last_frames:
                    last_frames = frames
                    last_change = now
                    continue
                if now - last_change < timeout_s:
                    continue
                self._stats["stalled"] += 1
                pid, played, total, text = self._finish_locked(
                    cancelled=True, reason=STOP_STALLED
                )
                self._stalled_piece = pid
                now_ms = mono_ms()
                payload = PlaybackStopped(
                    piece_id=pid,
                    t_ms=now_ms,
                    played_ms=played,
                    total_ms=total,
                    cancelled=True,
                    text=text,
                    stop_reason=STOP_STALLED,
                )
                fault = DeviceFault(
                    device="audio_out",
                    detail=(
                        f"playback_stalled: {self._stall_timeout_ms}ms 内 frames_out 未增长"
                        f"（piece={pid or '-'}, played={played}/{total}ms）"
                    ),
                    t_ms=now_ms,
                )
                self._watchdog = None
            # 事件层先说清原因，再交给 T14 故障恢复通道
            self._post(payload)
            self._post(fault)
            return

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
                pid, played, total, text = self._finish_locked(
                    cancelled=False, reason=STOP_COMPLETED
                )
                payload = PlaybackStopped(
                    piece_id=pid,
                    t_ms=mono_ms(),
                    played_ms=played,
                    total_ms=total,
                    cancelled=False,
                    text=text,
                    stop_reason=STOP_COMPLETED,
                )
            else:
                payload = None
        if payload is not None:
            self._post(payload)


__all__ = [
    "AudioSink",
    "PlaybackStopped",
    "SinkPlaybackHandle",
    "StreamFactory",
    "STOP_COMPLETED",
    "STOP_PREEMPTED",
    "STOP_STALLED",
    "STOP_USER_CANCEL",
]
