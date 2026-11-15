"""身份与近期资料权限（阶段 5）。

实现范围（最小可用，软件层可验证）：

- ``SpeakerGate``：本地声纹注册（campplus，L2 归一化 192 维）与核验
  （余弦相似度；阈值与最短时长取 ``SpeakerConfig``）。样本过短判为“不确定”，
  取不到向量判为“不可用”——这两类都不放行任何操作。
- ``TopicWindow``：正常问答结束并进入 T17 空闲后才建立的 15 分钟话题窗口（T21）；
  仅唤醒、失败、取消、停止都不建立/刷新窗口。
- ``WakeFreeCandidate``：免唤醒候选准入——窗口有效 + 声纹确认 + 对设备发话意图 +
  T19 预算（首段 40s + 识别 21s + 身份/意图 10s）＋一次 15 秒重入。
- 资料访问：读取/清除/恢复旧资料需要“近期已确认身份”（默认 5 分钟）；不确定、
  不匹配、不可用一律拒绝并保留原资料。

阈值版本写入 ``profile_version``，阶段报告引用该字符串。
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from aieyes.audio.wav import read_wav
from aieyes.clock import mono_ms
from aieyes.config import SpeakerConfig, TimersConfig
from aieyes.interfaces import SAMPLE_RATE


class IdentityStatus(str, Enum):
    """一次身份核验的结果。"""

    VERIFIED = "verified"
    UNCERTAIN = "uncertain"
    MISMATCH = "mismatch"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True, slots=True)
class IdentityResult:
    status: IdentityStatus
    score: float = 0.0
    reason: str = ""
    duration_ms: int = 0
    profile_version: str = ""


@dataclass(frozen=True, slots=True)
class CandidateDecision:
    """免唤醒候选的处置结果。"""

    action: str  # accept | ignore | reentry | need_explicit_wake | release
    reason: str = ""
    identity: IdentityResult | None = None
    retry_text: str | None = None
    prompt: str | None = None


class SpeakerGate:
    """说话人注册与核验（薄封装，测试可注入替身）。"""

    def __init__(
        self,
        embedder: Any | None = None,
        cfg: SpeakerConfig | None = None,
        *,
        models_root: str | Path | None = None,
    ) -> None:
        self.cfg = cfg or SpeakerConfig()
        self._embedder = embedder
        self._models_root = Path(models_root) if models_root else None
        self._embeddings: list[np.ndarray] = []
        self._samples_enrolled = 0

    # ------------------------------------------------------------------
    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from aieyes.speech.speaker import SherpaSpeaker  # noqa: PLC0415

            self._embedder = SherpaSpeaker(self.cfg)
        return self._embedder

    @property
    def enrolled(self) -> bool:
        return bool(self._embeddings)

    @property
    def profile_version(self) -> str:
        return (
            f"campplus-zh-cn-16k@v1:threshold={float(self.cfg.threshold):.2f}"
            f":min_speech={int(self.cfg.min_speech_ms)}ms"
        )

    def close(self) -> None:
        close = getattr(self._embedder, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # pragma: no cover - 释放失败不影响主流程
                pass
        self._embedder = None

    # ------------------------------------------------------------------
    def enroll(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bool:
        """注册一段本人语音；短于 ``min_speech_ms`` 直接拒绝。"""
        duration_ms = int(round(len(samples) * 1000 / max(1, sample_rate)))
        if duration_ms < int(self.cfg.min_speech_ms):
            return False
        if len(self._embeddings) >= int(self.cfg.max_samples):
            return False
        emb = self._get_embedder().embed(samples, sample_rate)
        self._embeddings.append(np.asarray(emb, dtype=np.float32))
        self._samples_enrolled += 1
        return True

    def enroll_file(self, path: str | Path) -> bool:
        samples, sr = read_wav(path, target_rate=SAMPLE_RATE)
        return self.enroll(samples, sr)

    def enroll_files(self, paths: Iterable[str | Path]) -> int:
        count = 0
        for path in paths:
            try:
                if self.enroll_file(path):
                    count += 1
            except Exception:  # 单个坏样本不影响其它样本
                continue
        return count

    def enroll_dir(self, directory: str | Path) -> int:
        directory = Path(directory)
        if not directory.is_dir():
            return 0
        files = sorted(p for p in directory.iterdir() if p.suffix.lower() == ".wav")
        return self.enroll_files(files[: int(self.cfg.max_samples)])

    def enroll_models_dir(self) -> int:
        """按 ``SpeakerConfig.enroll_dir`` 从模型目录注册。"""
        if self._models_root is None:
            return 0
        return self.enroll_dir(self._models_root / self.cfg.enroll_dir)

    # ------------------------------------------------------------------
    def verify(self, samples: np.ndarray, duration_ms: int | None = None) -> IdentityResult:
        duration = int(
            duration_ms
            if duration_ms is not None
            else round(len(np.asarray(samples).reshape(-1)) * 1000 / SAMPLE_RATE)
        )
        if not self._embeddings:
            return IdentityResult(IdentityStatus.UNAVAILABLE, 0.0, "no_profile", duration)
        if duration < int(self.cfg.min_speech_ms):
            return IdentityResult(IdentityStatus.UNCERTAIN, 0.0, "short_sample", duration)
        try:
            emb = np.asarray(self._get_embedder().embed(samples, SAMPLE_RATE), dtype=np.float32)
        except Exception as exc:  # 模型不可用或输入异常都按“不可用”
            return IdentityResult(IdentityStatus.UNAVAILABLE, 0.0, f"embed_error:{exc}", duration)
        scores = [float(np.dot(emb, ref)) for ref in self._embeddings]
        score = max(scores) if scores else 0.0
        if score >= float(self.cfg.threshold):
            return IdentityResult(IdentityStatus.VERIFIED, score, "match", duration, self.profile_version)
        return IdentityResult(IdentityStatus.MISMATCH, score, "below_threshold", duration)


@dataclass(slots=True)
class TopicWindow:
    """T21 话题窗口：只在正常问答结束后的空闲退出建立。"""

    window_ms: int = 900000
    opened_ms: int = 0
    label: str = ""

    @property
    def open(self) -> bool:
        return self.opened_ms > 0

    def open_at(self, now_ms: int, label: str = "") -> None:
        self.opened_ms = int(now_ms)
        self.label = label

    def close(self) -> None:
        self.opened_ms = 0
        self.label = ""

    def is_open(self, now_ms: int) -> bool:
        return self.open and now_ms < self.deadline_ms

    @property
    def deadline_ms(self) -> int:
        return self.opened_ms + int(self.window_ms) if self.open else 0

    def remaining_ms(self, now_ms: int) -> int:
        return max(0, self.deadline_ms - int(now_ms))


class IdentityManager:
    """把声纹门、话题窗口、免唤醒候选与资料访问组合成引擎可用的一个小对象。"""

    def __init__(
        self,
        gate: SpeakerGate | Any,
        timers: TimersConfig | None = None,
        *,
        topic_window_ms: int | None = None,
        access_max_age_ms: int = 300000,
        now: Any = mono_ms,
    ) -> None:
        self.gate = gate
        timers = timers or TimersConfig()
        self.timers = timers
        self.topic = TopicWindow(
            window_ms=int(topic_window_ms if topic_window_ms is not None else timers.t21_topic_window_ms)
        )
        self.access_max_age_ms = int(access_max_age_ms)
        self._now = now
        self._last_verified_ms = 0
        self._last_status = IdentityStatus.UNAVAILABLE
        # T19 候选状态
        self._candidate_onset_ms = 0
        self._retained_text: str | None = None
        self._retained_until_ms = 0
        self._reentry_until_ms = 0
        self._prompt_last_ms = 0
        self.candidates_seen = 0
        self.candidates_accepted = 0
        self.candidates_rejected = 0

    # ------------------------------------------------------------------ 基础
    @property
    def enrolled(self) -> bool:
        return bool(getattr(self.gate, "enrolled", False))

    @property
    def profile_version(self) -> str:
        return str(getattr(self.gate, "profile_version", "unknown"))

    def verify(self, samples: np.ndarray, duration_ms: int | None = None) -> IdentityResult:
        return self.gate.verify(samples, duration_ms)

    def note(self, result: IdentityResult, now_ms: int | None = None) -> None:
        now = self._now() if now_ms is None else int(now_ms)
        self._last_status = result.status
        if result.status is IdentityStatus.VERIFIED:
            self._last_verified_ms = now

    @property
    def last_status(self) -> IdentityStatus:
        return self._last_status

    def access_ok(self, now_ms: int | None = None) -> bool:
        """旧资料（清除/恢复）访问是否可用：需要近期确认过身份。"""
        now = self._now() if now_ms is None else int(now_ms)
        return (
            self._last_status is IdentityStatus.VERIFIED
            and self._last_verified_ms > 0
            and now - self._last_verified_ms <= self.access_max_age_ms
        )

    # ------------------------------------------------------------------ 话题窗口
    def open_topic(self, now_ms: int | None = None, label: str = "") -> int:
        now = self._now() if now_ms is None else int(now_ms)
        self.topic.open_at(now, label)
        return self.topic.deadline_ms

    def topic_open(self, now_ms: int | None = None) -> bool:
        now = self._now() if now_ms is None else int(now_ms)
        return self.topic.is_open(now)

    def close_topic(self) -> None:
        self.topic.close()
        self.release_candidate()

    # ------------------------------------------------------------------ 候选（T19）
    def begin_candidate(self, onset_ms: int) -> None:
        self._candidate_onset_ms = int(onset_ms)
        self.candidates_seen += 1

    def candidate_budget_ms(self) -> int:
        t = self.timers
        return int(t.t19_candidate_ms) + int(t.t19_extend_ms) + int(t.t19_final_ms)

    def decide_candidate(
        self,
        result: IdentityResult,
        text: str,
        *,
        directed: bool,
        onset_ms: int = 0,
        audio_end_ms: int = 0,
        now_ms: int | None = None,
    ) -> CandidateDecision:
        """免唤醒候选的准入判定；内部维护一次 15 秒重入状态。"""
        now = self._now() if now_ms is None else int(now_ms)
        t = self.timers
        if not self.topic.is_open(now):
            self.candidates_rejected += 1
            return CandidateDecision("ignore", "no_topic_window", result)
        onset = int(onset_ms or self._candidate_onset_ms)
        if onset > 0 and now - onset > self.candidate_budget_ms():
            self.release_candidate()
            self.candidates_rejected += 1
            return CandidateDecision("release", "t19_budget", result)
        # 重入窗口内的候选：确认身份后处理之前暂存的文字
        if self._reentry_until_ms and now <= self._reentry_until_ms:
            self._reentry_until_ms = 0
            if result.status is IdentityStatus.VERIFIED:
                retry, self._retained_text = self._retained_text, None
                self._retained_until_ms = 0
                if retry and result.status is IdentityStatus.VERIFIED:
                    self.candidates_accepted += 1
                    return CandidateDecision("accept", "reentry_confirmed", result, retry_text=retry)
            self._retained_text = None
            self._retained_until_ms = 0
            self.candidates_rejected += 1
            return CandidateDecision("release", "reentry_denied", result)
        if result.status is IdentityStatus.VERIFIED:
            if not directed:
                self.candidates_rejected += 1
                return CandidateDecision("ignore", "not_directed", result)
            self.candidates_accepted += 1
            return CandidateDecision("accept", "verified", result)
        if result.status is IdentityStatus.MISMATCH:
            self.candidates_rejected += 1
            return CandidateDecision("ignore", "identity_mismatch", result)
        if result.status is IdentityStatus.UNAVAILABLE:
            self.candidates_rejected += 1
            return CandidateDecision("need_explicit_wake", "voiceprint_unavailable", result)
        # UNCERTAIN：最多一次提示，并把文字暂存 15 秒等待重入
        self.candidates_rejected += 1
        cooldown = int(t.t19_prompt_cooldown_ms)
        if self._prompt_last_ms and now - self._prompt_last_ms < cooldown:
            return CandidateDecision("ignore", "uncertain_cooldown", result)
        self._prompt_last_ms = now
        self._retained_text = text
        self._retained_until_ms = now + int(t.t19_reentry_ms)
        self._reentry_until_ms = self._retained_until_ms
        return CandidateDecision(
            "reentry",
            "uncertain",
            result,
            retry_text=text,
            prompt="请再叫一次“小冰”，我确认一下是不是你。",
        )

    def release_candidate(self) -> None:
        self._retained_text = None
        self._retained_until_ms = 0
        self._reentry_until_ms = 0

    @property
    def reentry_active(self) -> bool:
        return self._reentry_until_ms > 0


__all__ = [
    "CandidateDecision",
    "IdentityManager",
    "IdentityResult",
    "IdentityStatus",
    "SpeakerGate",
    "TopicWindow",
]
