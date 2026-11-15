"""能量 VAD：绝对能量门限 + 迟滞 + 前置保护（preroll）。

判定量是每帧归一化 RMS（float32 [-1,1] 满量程）。``VadConfig.threshold``
映射为满量程 RMS 门限：``threshold`` 0.5 → 0.01 RMS（约 -40 dBFS），典型
近讲语音 RMS 在 0.03~0.2 之间，环境底噪通常 < 0.01。升级/回落采用两个
门限（迟滞），避免清辅音、尾音把一段话切碎。

状态机
------
``idle`` → 连续有声且累计时长 ≥ ``min_speech_ms`` → ``speech``
``speech`` → 连续静音累计 ≥ ``min_silence_ms`` → 输出 SpeechSegment(vad_end)
``speech`` → 段长 ≥ ``max_utterance_ms`` → 输出 SpeechSegment(max_len)

段音频包含 ``preroll_ms`` 的前置帧，保证首字不丢；尾部静音保留到
``min_silence_ms`` 为止（不超过），便于 ASR 看到停顿。
"""

from __future__ import annotations

from collections import deque

import numpy as np

from aieyes.config import VadConfig
from aieyes.interfaces import SAMPLE_RATE, AudioFrame, SpeechSegment

#: 有声候选期允许的最长中断（连续低于回落门限的时长）
_CANDIDATE_HANG_MS = 120


class EnergyVad:
    """基于帧 RMS 的门限 + 迟滞 VAD。"""

    def __init__(
        self,
        cfg: VadConfig | None = None,
        *,
        sample_rate: int = SAMPLE_RATE,
        frame_ms: int = 20,
    ) -> None:
        self.cfg = cfg or VadConfig()
        self.sample_rate = int(sample_rate)
        self.frame_ms = int(frame_ms)

        threshold = float(np.clip(self.cfg.threshold, 0.01, 5.0))
        #: 开门限：归一化 RMS
        self.open_level = 0.02 * threshold
        #: 关门限：迟滞（约 4.4 dB）
        self.close_level = self.open_level * 0.6

        self._preroll: deque[AudioFrame] = deque()
        self._preroll_ms = 0
        self._cand: list[AudioFrame] = []
        self._cand_pre: list[AudioFrame] = []
        self._seg: list[AudioFrame] = []
        self._trailing_sil_ms = 0
        self._in_speech = False

        self._reset_state()

    # ------------------------------------------------------------------
    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        self._preroll.clear()
        self._preroll_ms = 0
        self._cand = []
        self._cand_pre = []
        self._seg = []
        self._trailing_sil_ms = 0
        self._in_speech = False

    # ------------------------------------------------------------------
    @staticmethod
    def frame_level(frame: AudioFrame) -> float:
        x = np.asarray(frame.samples, dtype=np.float32)
        if x.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(x))))

    def accept(self, frame: AudioFrame) -> SpeechSegment | None:
        frame = _as_frame(frame, self.sample_rate)
        level = self.frame_level(frame)
        dur = _frame_duration_ms(frame, self.sample_rate)

        if self._in_speech:
            return self._accept_in_speech(frame, level, dur)

        if self._cand:
            return self._accept_candidate(frame, level, dur)

        # idle
        if level >= self.open_level:
            self._cand = [frame]
            self._cand_pre = list(self._preroll)
            if _frames_ms(self._cand, self.sample_rate) >= self.cfg.min_speech_ms:
                return self._enter_speech()
        else:
            self._push_preroll(frame)
        return None

    # ------------------------------------------------------------------
    def _accept_candidate(
        self, frame: AudioFrame, level: float, dur: int
    ) -> SpeechSegment | None:
        if level >= self.close_level:
            self._cand.append(frame)
            if _frames_ms(self._cand, self.sample_rate) >= self.cfg.min_speech_ms:
                return self._enter_speech()
            return None

        self._cand.append(frame)
        # 候选期内短暂低于回落门限不立即丢弃，超过 hang 时间才丢掉；
        # 只有“非尾部静音”的累计时长才计入 min_speech。
        tail = self._trailing_below_ms(self._cand)
        voiced_ms = _frames_ms(self._cand, self.sample_rate) - tail
        if voiced_ms >= self.cfg.min_speech_ms:
            return self._enter_speech()
        if tail >= _CANDIDATE_HANG_MS:
            for f in self._cand:
                self._push_preroll(f)
            self._cand = []
            self._cand_pre = []
        return None

    def _enter_speech(self) -> SpeechSegment | None:
        self._seg = self._cand_pre + self._cand
        self._cand = []
        self._cand_pre = []
        self._trailing_sil_ms = 0
        self._in_speech = True
        # 进入即可能已超 max_len（极短配置），交给上层下一次 accept 处理
        if _frames_ms(self._seg, self.sample_rate) >= self.cfg.max_utterance_ms:
            return self._finish("max_len")
        return None

    def _accept_in_speech(
        self, frame: AudioFrame, level: float, dur: int
    ) -> SpeechSegment | None:
        self._seg.append(frame)
        if level < self.close_level:
            self._trailing_sil_ms += dur
        else:
            self._trailing_sil_ms = 0

        if self._trailing_sil_ms >= self.cfg.min_silence_ms:
            return self._finish("vad_end")
        if _frames_ms(self._seg, self.sample_rate) >= self.cfg.max_utterance_ms:
            return self._finish("max_len")
        return None

    def _finish(self, cause: str) -> SpeechSegment:
        frames = self._seg
        self._seg = []
        self._in_speech = False
        if not frames:
            return SpeechSegment(
                samples=np.zeros(0, dtype=np.float32), t_start_ms=0, t_end_ms=0, cause=cause
            )
        samples = np.concatenate([np.asarray(f.samples, dtype=np.float32) for f in frames])
        t_start = int(frames[0].t_start_ms)
        t_end = int(frames[-1].t_start_ms + _frame_duration_ms(frames[-1], self.sample_rate))
        # 保留最近的帧作为下一段可能的 preroll
        for f in frames[-_preroll_frames(self.cfg.preroll_ms, self.sample_rate) :]:
            self._push_preroll(f)
        return SpeechSegment(samples=samples, t_start_ms=t_start, t_end_ms=t_end, cause=cause)

    # ------------------------------------------------------------------
    def _push_preroll(self, frame: AudioFrame) -> None:
        self._preroll.append(frame)
        self._preroll_ms += _frame_duration_ms(frame, self.sample_rate)
        limit = max(0, self.cfg.preroll_ms)
        while self._preroll and self._preroll_ms - _frame_duration_ms(
            self._preroll[0], self.sample_rate
        ) >= limit:
            old = self._preroll.popleft()
            self._preroll_ms -= _frame_duration_ms(old, self.sample_rate)

    def _trailing_below_ms(self, frames: list[AudioFrame]) -> int:
        total = 0
        for f in reversed(frames):
            if self.frame_level(f) >= self.close_level:
                break
            total += _frame_duration_ms(f, self.sample_rate)
        return total


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _as_frame(frame: AudioFrame, sample_rate: int) -> AudioFrame:
    samples = np.asarray(frame.samples, dtype=np.float32).reshape(-1)
    if samples is frame.samples:
        return frame
    return AudioFrame(samples=samples, t_start_ms=frame.t_start_ms, seq=frame.seq)


def _frame_duration_ms(frame: AudioFrame, sample_rate: int) -> int:
    return int(round(len(frame.samples) * 1000 / sample_rate))


def _frames_ms(frames: list[AudioFrame], sample_rate: int) -> int:
    return sum(_frame_duration_ms(f, sample_rate) for f in frames)


def _preroll_frames(preroll_ms: int, sample_rate: int, frame_ms: int = 20) -> int:
    return max(1, int(round(preroll_ms / max(1, frame_ms))))
