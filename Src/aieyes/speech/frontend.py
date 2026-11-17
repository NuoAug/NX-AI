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

import re
import threading
from collections import deque
from typing import Callable

import numpy as np

from aieyes.clock import mono_ms
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
        # 尾音误触发保护：真人录音里句尾低电平噪声会被能量 VAD 当成第二段，
        # 再被 ASR  hallucinate 成“不东西”。这里只丢弃“明显低于本次说话电平、
        # 很短、且没有可执行短词”的尾段，不按固定绝对阈值误杀轻声说话。
        self._level_history: deque[float] = deque(maxlen=5)
        self._speech_rms_ref = 0.0
        #: 被当作尾音噪声丢弃的段数（可观测，不参与播报）。
        self.dropped_low_energy = 0
        # 有界续接：句末是明确连接词时，等一个短停顿再提交；避免自然停顿被
        # 拆成两轮（“重拍第一件，然后……只比较价格”）。只等一次，超时必提交。
        self._continuation_grace_ms = 700
        self._pending_segment: SpeechSegment | None = None
        self._pending_text = ""
        self._pending_until_ms = 0

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
            self._level_history.clear()
            self._speech_rms_ref = 0.0
            self.dropped_low_energy = 0
            self._clear_pending()
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
        self._level_history.clear()
        self._speech_rms_ref = 0.0
        self._clear_pending()

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
                self._clear_pending()
                return actions

            # 有界续接的等待超时：用户没有再说，按已识别文本提交，不能无限等。
            if (self._pending_segment is not None
                    and not self.vad.in_speech
                    and int(frame.t_start_ms) >= self._pending_until_ms):
                utt = self._flush_pending_locked()
                if utt is not None:
                    actions.append(("utterance", utt))
                return actions

            segment: SpeechSegment | None = self.vad.accept(frame)
            if bool(self.vad.in_speech) and not self._in_speech:
                self._in_speech = True
                actions.append(("speech_start", int(frame.t_start_ms)))
                if self._pending_segment is None:
                    self._begin_asr_locked(actions)
                # 续接：同一个 ASR 流继续喂，不重建、不丢前半句。
            elif self._asr_active:
                partial = self.asr.accept(frame)
                if partial:
                    actions.append(("partial", partial, int(frame.t_start_ms)))

            if segment is not None:
                self._in_speech = False
                if self._asr_active:
                    if self._pending_segment is not None:
                        # 停顿期间又说了：把两段并成一次话轮提交。
                        utt = self.asr.finalize()
                        self._asr_active = False
                        merged = self._merge_pending_locked(segment)
                        text = utt.text or self._pending_text
                        self._clear_pending()
                        level = _segment_rms(merged.samples)
                        self._note_speech_level(level)
                        actions.append(("utterance", Utterance(
                            text=text,
                            t_audio_start_ms=int(merged.t_start_ms),
                            t_audio_end_ms=int(merged.t_end_ms),
                            t_final_ms=int(utt.t_final_ms),
                            samples=merged.samples,
                        )))
                        return actions
                    preview = self._asr_preview()
                    if self._should_wait_for_more(segment, preview):
                        self._pending_segment = segment
                        self._pending_text = preview or ""
                        self._pending_until_ms = (
                            int(segment.t_end_ms) + int(self._continuation_grace_ms)
                        )
                        return actions
                    utt = self.asr.finalize()
                    self._asr_active = False
                    samples = np.asarray(segment.samples, dtype=np.float32)
                    level = _segment_rms(samples)
                    if self._looks_like_tail_noise(segment, level, utt.text):
                        self.dropped_low_energy += 1
                        return actions
                    self._note_speech_level(level)
                    utt = Utterance(
                        text=utt.text,
                        t_audio_start_ms=int(segment.t_start_ms),
                        t_audio_end_ms=int(segment.t_end_ms),
                        t_final_ms=int(utt.t_final_ms),
                        samples=samples,
                    )
                    actions.append(("utterance", utt))
            return actions

    # 短词白名单：即便电平偏低、时长很短，也可能是有效指令，不能当尾音丢掉。
    _SHORT_MEANINGFUL = frozenset(
        {
            "嗯", "嗯嗯", "好", "好的", "行", "对", "可以", "继续", "接着",
            "停", "停止", "停下", "重拍", "重来", "换一个", "开麦", "关麦",
            "静音", "清空", "取消", "算了", "不用", "不要", "第一个", "第二个",
            "好了", "不是", "是的",
        }
    )

    def _looks_like_tail_noise(self, segment: SpeechSegment, level: float, text: str) -> bool:
        """句尾低电平假段：明显低于本次说话电平、很短、且不是有效短词。"""
        reference = self._speech_rms_ref
        if reference <= 0 or segment.cause != "vad_end":
            return False
        duration = int(segment.t_end_ms) - int(segment.t_start_ms)
        if duration >= 1500 or level >= 0.45 * reference:
            return False
        core = re.sub(r"[\s，。！？!?、,.；;：:]+", "", text or "")
        return 0 < len(core) <= 4 and core not in self._SHORT_MEANINGFUL

    def _note_speech_level(self, level: float) -> None:
        if level <= 0:
            return
        self._level_history.append(level)
        ordered = sorted(self._level_history)
        self._speech_rms_ref = ordered[len(ordered) // 2]

    #: 只有句末是明确连接词时才等短停顿；普通陈述不能因为一个“了/的”多等。
    _CONTINUATION_ENDINGS = (
        "然后", "还有", "但是", "因为", "所以", "如果", "或者", "以及",
        "并且", "而且", "接着", "不过", "那么", "比如说", "比如", "就是",
    )

    def _asr_preview(self) -> str | None:
        """当前流结果（不结束、不重置）；ASR 实现不支持时返回 None。"""
        preview = getattr(self.asr, "result", None)
        if not callable(preview):
            return None
        try:
            return str(preview() or "")
        except Exception:  # noqa: BLE001 - 预览失败不影响正常收尾
            return None

    def _should_wait_for_more(self, segment: SpeechSegment, preview: str | None) -> bool:
        if preview is None or segment.cause != "vad_end":
            return False
        if int(segment.t_end_ms) - int(segment.t_start_ms) >= 8000:
            return False
        core = re.sub(r"[\s，。！？!?、,.；;：:]+", "", preview)
        return len(core) >= 2 and any(core.endswith(word) for word in self._CONTINUATION_ENDINGS)

    def _merge_pending_locked(self, second: SpeechSegment) -> SpeechSegment:
        first = self._pending_segment
        assert first is not None
        overlap_ms = max(0, int(first.t_end_ms) - int(second.t_start_ms))
        drop = min(len(second.samples), int(overlap_ms * self.sample_rate / 1000))
        samples = np.concatenate(
            [
                np.asarray(first.samples, dtype=np.float32),
                np.asarray(second.samples, dtype=np.float32)[drop:],
            ]
        )
        return SpeechSegment(
            samples=samples,
            t_start_ms=int(first.t_start_ms),
            t_end_ms=int(second.t_end_ms),
            cause="continuation",
        )

    def _flush_pending_locked(self) -> Utterance | None:
        pending = self._pending_segment
        if pending is None:
            return None
        text = self._pending_text
        t_final = mono_ms()
        if self._asr_active:
            utt = self.asr.finalize()
            self._asr_active = False
            if utt.text:
                text = utt.text
            t_final = int(utt.t_final_ms)
        samples = np.asarray(pending.samples, dtype=np.float32)
        self._clear_pending()
        self._note_speech_level(_segment_rms(samples))
        return Utterance(
            text=text,
            t_audio_start_ms=int(pending.t_start_ms),
            t_audio_end_ms=int(pending.t_end_ms),
            t_final_ms=t_final,
            samples=samples,
        )

    def _clear_pending(self) -> None:
        self._pending_segment = None
        self._pending_text = ""
        self._pending_until_ms = 0

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


def _segment_rms(samples: np.ndarray) -> float:
    if not samples.size:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples))))
