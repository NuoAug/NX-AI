"""实时采集：sounddevice + 独立分发线程。

PortAudio 的输入回调运行在实时线程上，不能在回调里做 KWS/ASR 这类重活，
否则会丢帧。这里回调只做“拷贝 + 入队”，由一个普通线程取出后调用
``on_frame``；队列满时丢最旧帧并计数（欠载/溢出可观测）。

``AudioFrame.t_start_ms`` 以首样本时间对齐：流启动时取 ``clock.mono_ms()``
作为锚点，第 n 帧起点 = 锚点 + 已采集样本数 × 1000 / sample_rate，与
PortAudio 的样本计数严格对应，不受回调抖动影响。

断流如实上报：给了 ``bus`` 时，连续 ``fault_timeout_ms`` 收不到新帧就 post 一次
``DeviceFault(device="audio_in")``，恢复交给 engine 侧 T14 统一决定；本模块
**不**自动重连（流程说明要求"权限撤销或用户主动关闭设备时不自动重启"）。
用户 ``stop()``、运行代次变化（旧会话）都不发；同一次断流只报一次，收到
新帧后才允许再报。
"""

from __future__ import annotations

import queue
import threading
from typing import Callable

import numpy as np

from aieyes.clock import mono_ms
from aieyes.config import AudioConfig
from aieyes.errors import AudioDeviceError
from aieyes.events import DeviceFault
from aieyes.interfaces import SAMPLE_RATE, AudioFrame

#: 采集断流上报阈值（毫秒）。取 100 个 20ms 帧周期的量级：正常抖动/短暂
#: 欠载（队列满、GC）远达不到；同时远小于流程说明 T14 的"30 秒未恢复"
#: 预算，engine 收到 fault 后仍有时间走 ``_recover_audio``。
DEFAULT_FAULT_TIMEOUT_MS = 2000


class SoundDeviceSource:
    """sounddevice 采集源（16 kHz / mono / float32 / 20 ms）。"""

    def __init__(
        self,
        cfg: AudioConfig | None = None,
        *,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 20,
        device: str | int | None = None,
        gain_db: float | None = None,
        queue_size: int = 100,
        bus=None,
        fault_timeout_ms: int = DEFAULT_FAULT_TIMEOUT_MS,
    ) -> None:
        cfg = cfg or AudioConfig()
        self.cfg = cfg
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_samples = self.sample_rate * self.frame_ms // 1000
        self.device = cfg.input_device if device is None else device
        self.gain_db = float(cfg.input_gain_db if gain_db is None else gain_db)
        self._gain = float(10.0 ** (self.gain_db / 20.0))
        self.queue_size = max(1, int(queue_size))

        self._on_frame: Callable[[AudioFrame], None] | None = None
        self._queue: queue.Queue | None = None
        self._thread: threading.Thread | None = None
        self._stream = None
        self._running = False
        self._t0_ms = 0
        self._samples_seen = 0
        self._seq = 0
        self._dropped = 0
        self.xruns = 0

        # 断流上报（只报故障，不自动重连）
        self.bus = bus
        self.fault_timeout_ms = max(0, int(fault_timeout_ms))
        self._gen = 0
        self._last_frame_ms = 0
        self._fault_reported = False
        self._watchdog: threading.Thread | None = None
        self._watchdog_stop = threading.Event()

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def dropped_frames(self) -> int:
        return self._dropped

    # ------------------------------------------------------------------
    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        if self._running:
            return
        try:
            import sounddevice as sd  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - 依赖系统库
            raise AudioDeviceError(
                f"sounddevice 不可用（缺少 libportaudio2？）: {exc}"
            ) from exc

        self._on_frame = on_frame
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._dropped = 0
        self._samples_seen = 0
        self._seq = 0
        self._t0_ms = mono_ms()

        def _callback(indata, frames, time_info, status) -> None:  # PortAudio 线程
            if status:
                self.xruns += 1
            samples = np.asarray(indata[:, 0], dtype=np.float32).copy()
            if self._gain != 1.0:
                samples *= self._gain
            t_start = self._t0_ms + int(self._samples_seen * 1000 / self.sample_rate)
            self._samples_seen += len(samples)
            frame = AudioFrame(samples=samples, t_start_ms=t_start, seq=self._seq)
            self._seq += 1
            q = self._queue
            if q is None:
                return
            try:
                q.put_nowait(frame)
            except queue.Full:
                self._dropped += 1
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(frame)
                except queue.Full:
                    pass

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self.frame_samples,
                device=self.device,
                callback=_callback,
            )
            self._stream.start()
        except Exception as exc:
            self._stream = None
            raise AudioDeviceError(f"打开采集设备失败 device={self.device!r}: {exc}") from exc

        self._running = True
        self._gen += 1
        self._last_frame_ms = self._t0_ms
        self._fault_reported = False
        self._thread = threading.Thread(
            target=self._dispatch_loop, name="aieyes-capture-dispatch", daemon=True
        )
        self._thread.start()

        if self.fault_timeout_ms > 0 and self.bus is not None:
            self._watchdog_stop = threading.Event()
            self._watchdog = threading.Thread(
                target=self._watchdog_loop,
                args=(self._gen,),
                name="aieyes-capture-watchdog",
                daemon=True,
            )
            self._watchdog.start()

    # ------------------------------------------------------------------ 断流上报
    def _note_frame(self) -> None:
        """收到一帧：更新“最后见帧时间”，并允许下一次断流再报。"""
        self._last_frame_ms = mono_ms()
        self._fault_reported = False

    def _watchdog_tick(self, gen: int) -> bool:
        """检查一次是否断流；返回是否应继续监控。

        只有“代次未变 + 仍在运行 + 未上报过本次断流”三者同时成立才 post，
        保证用户 ``stop()`` 与旧会话（run）不会发故障。
        """
        if self.fault_timeout_ms <= 0:
            return False
        if not self._running or gen != self._gen:
            return False
        if self._fault_reported:
            return True
        if mono_ms() - self._last_frame_ms < self.fault_timeout_ms:
            return True
        self._fault_reported = True
        now_ms = mono_ms()
        fault = DeviceFault(
            device="audio_in",
            detail=(
                f"未收到音频帧 {self.fault_timeout_ms}ms（采集流仍在运行，"
                f"回调未推进；最后一次见帧 t={self._last_frame_ms}ms）"
            ),
            t_ms=now_ms,
        )
        bus = self.bus
        if bus is not None:
            try:
                bus.post(fault)
            except Exception:  # 事件总线异常不能打断采集
                pass
        return True

    def _watchdog_loop(self, gen: int) -> None:
        tick_s = max(0.02, min(0.2, self.fault_timeout_ms / 1000.0 / 10.0))
        while not self._watchdog_stop.wait(tick_s):
            if not self._watchdog_tick(gen):
                break

    def _dispatch_loop(self) -> None:
        q = self._queue
        while True:
            try:
                frame = q.get(timeout=0.2) if q is not None else None
            except queue.Empty:
                if not self._running:
                    break
                continue
            if frame is None:
                break
            self._note_frame()
            cb = self._on_frame
            if cb is not None:
                try:
                    cb(frame)
                except Exception:  # 单帧回调异常不能终止采集线程
                    pass
            if not self._running and q.empty():
                break

    def stop(self) -> None:
        self._running = False
        self._gen += 1  # 让仍在跑的看门狗失效（旧会话不得再上报）
        self._watchdog_stop.set()
        watchdog, self._watchdog = self._watchdog, None
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=1.0)
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
            finally:
                try:
                    stream.close()
                except Exception:
                    pass
        q = self._queue
        if q is not None:
            try:
                q.put_nowait(None)
            except queue.Full:
                pass
        thread, self._thread = self._thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._queue = None

    def __enter__(self) -> "SoundDeviceSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
