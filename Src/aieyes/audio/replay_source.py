"""ReplaySource：不依赖声卡，把 WAV/数组按实时节奏喂成 AudioFrame。

阶段 0 的离线回放入口：用真实录音验证“唤醒 → VAD → 流式 ASR”整条软件
链路。``realtime=True`` 时按采集节拍（20 ms/帧）投递，接近真实设备；
``realtime=False`` 时尽快跑完，供单元测试/批处理使用。

片段之间可插入静音（``gap_ms``），用于拼接“唤醒词 wav + 静音 + 问题 wav”。
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from aieyes.clock import mono_ms
from aieyes.interfaces import SAMPLE_RATE, AudioFrame

from .wav import concat_segments, read_wav

SegmentLike = str | Path | np.ndarray


class ReplaySource:
    """把一段或多段音频按帧回放为 ``AudioFrame`` 流。"""

    def __init__(
        self,
        segments: Sequence[SegmentLike] | SegmentLike,
        *,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 20,
        realtime: bool = True,
        gap_ms: int = 0,
        start_delay_ms: int = 0,
    ) -> None:
        if isinstance(segments, (str, Path, np.ndarray)):
            segments = [segments]
        self._segments_in: list[SegmentLike] = list(segments)
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)
        self.frame_samples = self.sample_rate * self.frame_ms // 1000
        self.realtime = bool(realtime)
        self.gap_ms = int(gap_ms)
        self.start_delay_ms = int(start_delay_ms)

        self._on_frame: Callable[[AudioFrame], None] | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._running = False
        self._frames_sent = 0
        self._t0_ms = 0

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frames_sent(self) -> int:
        return self._frames_sent

    #: 便于测试检查总帧数
    @property
    def total_frames(self) -> int:
        total = 0
        for i in range(len(self._segments_in)):
            if i:
                total += self._gap_frames()
            total += self._frames_for(self._load(self._segments_in[i]))
        return total

    # ------------------------------------------------------------------
    def start(self, on_frame: Callable[[AudioFrame], None]) -> None:
        if self._running:
            return
        self._on_frame = on_frame
        self._stop.clear()
        self._frames_sent = 0
        self._t0_ms = mono_ms()
        self._running = True
        self._thread = threading.Thread(
            target=self._run, name="aieyes-replay", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._running = False
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

    def wait(self, timeout: float | None = None) -> bool:
        """等待回放线程结束；返回是否已结束。"""
        thread = self._thread
        if thread is None:
            return True
        thread.join(timeout=timeout)
        return not thread.is_alive()

    def __enter__(self) -> "ReplaySource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ------------------------------------------------------------------
    def _load(self, seg: SegmentLike) -> np.ndarray:
        if isinstance(seg, np.ndarray):
            return np.asarray(seg, dtype=np.float32).reshape(-1)
        samples, _ = read_wav(seg, target_rate=self.sample_rate)
        return samples

    def _frames_for(self, samples: np.ndarray) -> int:
        return int(np.ceil(samples.size / self.frame_samples)) if samples.size else 0

    def _gap_frames(self) -> int:
        return int(round(self.gap_ms / self.frame_ms)) if self.gap_ms > 0 else 0

    # ------------------------------------------------------------------
    def _run(self) -> None:
        n = self.frame_samples
        try:
            self._sleep_ms(self.start_delay_ms)
            first = True
            for i, seg in enumerate(self._segments_in):
                if self._stop.is_set():
                    break
                samples = self._load(seg)
                if not first and self.gap_ms > 0:
                    self._emit_zeros(self._gap_frames(), n)
                first = False
                total = self._frames_for(samples)
                for k in range(total):
                    if self._stop.is_set():
                        break
                    chunk = samples[k * n : (k + 1) * n]
                    if chunk.size < n:
                        chunk = np.pad(chunk, (0, n - chunk.size))
                    self._emit(chunk.astype(np.float32, copy=False))
        finally:
            self._running = False

    def _emit_zeros(self, count: int, n: int) -> None:
        zeros = np.zeros(n, dtype=np.float32)
        for _ in range(count):
            if self._stop.is_set():
                return
            self._emit(zeros)

    def _emit(self, samples: np.ndarray) -> None:
        seq = self._frames_sent
        if self.realtime:
            target = self._t0_ms + int(round(seq * self.frame_ms))
            delay = target - mono_ms()
            if delay > 0:
                self._sleep_ms(delay)
        frame = AudioFrame(
            samples=samples,
            t_start_ms=self._t0_ms + int(round(seq * self.frame_ms)),
            seq=seq,
        )
        self._frames_sent += 1
        cb = self._on_frame
        if cb is not None:
            cb(frame)

    def _sleep_ms(self, ms: int) -> None:
        if ms <= 0:
            return
        factor = 1.0 if self.realtime else 0.0
        self._stop.wait(ms / 1000.0 * factor if factor else 0.0)
        # 非实时模式不做 sleep；实时模式上面已等待
        if not self.realtime:
            return


def build_replay_playlist(
    prompt_files: Sequence[SegmentLike],
    *,
    gap_ms: int = 600,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """把多段 WAV 合成一条音频（调试/测试辅助）。"""
    segs = [read_wav(p, target_rate=sample_rate)[0] if not isinstance(p, np.ndarray) else p for p in prompt_files]
    return concat_segments(segs, gap_ms=gap_ms, sample_rate=sample_rate)
