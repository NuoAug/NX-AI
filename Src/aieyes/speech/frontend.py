"""语音前端：AudioSource →（常开 KWS）→ VAD → 流式 ASR 的粘合层。

约定（与 Lead 的状态机对齐）：

* Frontend **只调用回调**：``on_wake`` / ``on_speech_start`` / ``on_partial`` /
  ``on_utterance`` / ``on_level``；**不直接往 EventBus post 事件**，避免重复；
* 默认 ``listening=False``：KWS 常开；命中唤醒词只回调 ``on_wake``，由上层决定
  何时 ``set_listening(True)``（``auto_listen_on_wake=True`` 可自动进入听写）；
* 听写中由 VAD 判定段落：开口时把 ``preroll_ms`` 的环形缓冲补喂给 ASR（首字保护），
  段结束时 ``finalize`` 得到带时间戳的 ``Utterance``；
* 一次 utterance 结束后保持 listening，可继续下一段（上层用
  ``set_listening(False)``/``flush()`` 收口）；
* 回调在采集线程内执行，只做轻量操作；异常被吞掉，不会打断采集。
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Callable

import numpy as np

from aieyes.config import VadConfig
from aieyes.interfaces import (
    SAMPLE_RATE,
    Asr,
    AudioFrame,
    AudioSource,
    SpeechSegment,
    Utterance,
    Vad,
    WakeWordDetector,
)


class SpeechFrontend:
    """实现 ``VoiceFrontend`` 协议。"""

    def __init__(
        self,
        source: AudioSource,
        *,
        kws: WakeWordDetector | None = None,
        vad: Vad | None = None,
        asr: Asr | None = None,
        sample_rate: int = SAMPLE_RATE,
        preroll_ms: int | None = None,
        auto_listen_on_wake: bool = False,
    ) -> None:
        self.source = source
        self.kws = kws
        self.vad = vad
        self.asr = asr
        self.sample_rate = int(sample_rate)
        self.preroll_ms = int(VadConfig().preroll_ms if preroll_ms is None else preroll_ms)
        self.auto_listen_on_wake = bool(auto_listen_on_wake)

        #: 上层状态机的回调（由上层赋值；Frontend 不 post 事件）
        self.on_wake: Callable[[str, int], None] | None = None
        self.on_speech_start: Callable[[int], None] | None = None
        self.on_partial: Callable[[str, int], None] | None = None
        self.on_utterance: Callable[[Utterance], None] | None = None
        self.on_level: Callable[[float, int], None] | None = None

        self._lock = threading.RLock()
        self._running = False
        self._wake_enabled = True
        self._listening = False
        self._asr_active = False
        self._in_speech = False
        self._preroll: deque[AudioFrame] = deque(maxlen=max(1, int(round(self.preroll_ms / 20))))
        self._frames = 0

    # ------------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._running

    @property
    def listening(self) -> bool:
        with self._lock:
            return self._listening

    @property
    def wake_enabled(self) -> bool:
        with self._lock:
            return self._wake_enabled

    # ------------------------------------------------------------------
    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._running = True
            self._asr_active = False
            self._in_speech = False
            self._preroll.clear()
            self._frames = 0
            if self.vad is not None:
                self.vad.reset()
            if self.asr is not None:
                self.asr.reset()
        # 冷启动首帧可能耗时 2s+（KWS/ASR 首次推理），在开始回放/采集前消化掉，
        # 否则首段音频会被阻塞，唤醒时间戳与真实时间脱节。
        self.warmup()
        self.source.start(self._on_frame)

    def warmup(self, frames: int = 3) -> None:
        """用静音帧预热 KWS/VAD/ASR，避免首次推理落在真实音频上。"""
        silence = np.zeros(int(self.sample_rate * 0.02), dtype=np.float32)
        warm_frames = []
        for i in range(max(1, int(frames))):
            warm_frames.append(
                AudioFrame(samples=silence.copy(), t_start_ms=i * 20, seq=i)
            )
        if self.kws is not None:
            for frame in warm_frames:
                try:
                    self.kws.accept(frame)
                except Exception:
                    break
            self.kws.reset()
        if self.vad is not None:
            for frame in warm_frames:
                try:
                    self.vad.accept(frame)
                except Exception:
                    break
            self.vad.reset()
        if self.asr is not None:
            try:
                self.asr.start_utterance()
                for frame in warm_frames:
                    self.asr.accept(frame)
                self.asr.finalize()
            except Exception:
                pass
            finally:
                self.asr.reset()

    def stop(self) -> None:
        with self._lock:
            self._running = False
        try:
            self.source.stop()
        finally:
            self.flush()
        with self._lock:
            self._listening = False

    # ------------------------------------------------------------------
    def set_listening(self, enabled: bool) -> None:
        with self._lock:
            self._set_listening_locked(bool(enabled))

    def _set_listening_locked(self, enabled: bool) -> None:
        if enabled == self._listening:
            return
        self._listening = enabled
        if enabled:
            if self.vad is not None:
                self.vad.reset()
            if self.asr is not None:
                self.asr.reset()
            self._asr_active = False
            self._in_speech = False
            # 保留 preroll，用于打开听写瞬间的首字保护
        else:
            self._flush_locked()

    def set_wake_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        with self._lock:
            if enabled == self._wake_enabled:
                return
            self._wake_enabled = enabled
            if self.kws is not None:
                self.kws.reset()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self.vad is not None:
            self.vad.reset()
        if self.asr is not None:
            self.asr.reset()
        self._asr_active = False
        self._in_speech = False
        self._preroll.clear()

    # ------------------------------------------------------------------
    def _on_frame(self, frame: AudioFrame) -> None:
        if not self._running:
            return
        try:
            actions = self._process(frame)
        except Exception:
            return
        self._dispatch(actions)

    def _process(self, frame: AudioFrame) -> list[tuple]:
        actions: list[tuple] = []
        with self._lock:
            samples = np.asarray(frame.samples, dtype=np.float32)
            level = float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0
            actions.append(("level", level, int(frame.t_start_ms)))
            self._preroll.append(frame)
            self._frames += 1

            listening = self._listening
            if self._wake_enabled and not listening and self.kws is not None:
                keyword = self.kws.accept(frame)
                if keyword:
                    actions.append(("wake", keyword, int(frame.t_start_ms)))
                    if self.auto_listen_on_wake:
                        self._set_listening_locked(True)
                return actions

            if not listening or self.vad is None or self.asr is None:
                return actions

            segment: SpeechSegment | None = self.vad.accept(frame)
            if bool(self.vad.in_speech) and not self._in_speech:
                self._in_speech = True
                actions.append(("speech_start", int(frame.t_start_ms)))
                self._begin_asr_locked(actions)
            elif self._asr_active:
                partial = self.asr.accept(frame)
                if partial:
                    actions.append(("partial", partial, int(frame.t_start_ms)))

            if segment is not None:
                if self._asr_active:
                    utt = self.asr.finalize()
                    self._asr_active = False
                    utt = Utterance(
                        text=utt.text,
                        t_audio_start_ms=int(segment.t_start_ms),
                        t_audio_end_ms=int(segment.t_end_ms),
                        t_final_ms=int(utt.t_final_ms),
                        samples=np.asarray(segment.samples, dtype=np.float32),
                    )
                    actions.append(("utterance", utt))
                self._in_speech = False
            return actions

    def _begin_asr_locked(self, actions: list[tuple]) -> None:
        """VAD 确认开口：新建 ASR 片段并把 preroll 补喂进去（首字保护）。"""
        if self.asr is None:
            return
        self.asr.start_utterance()
        self._asr_active = True
        for buffered in list(self._preroll):
            partial = self.asr.accept(buffered)
            if partial:
                actions.append(("partial", partial, int(buffered.t_start_ms)))

    def _dispatch(self, actions: list[tuple]) -> None:
        for action in actions:
            kind = action[0]
            try:
                if kind == "level":
                    if self.on_level is not None:
                        self.on_level(float(action[1]), int(action[2]))
                elif kind == "wake":
                    if self.on_wake is not None:
                        self.on_wake(str(action[1]), int(action[2]))
                elif kind == "speech_start":
                    if self.on_speech_start is not None:
                        self.on_speech_start(int(action[1]))
                elif kind == "partial":
                    if self.on_partial is not None:
                        self.on_partial(str(action[1]), int(action[2]))
                elif kind == "utterance":
                    if self.on_utterance is not None:
                        self.on_utterance(action[1])
            except Exception:
                continue


__all__ = ["SpeechFrontend"]
