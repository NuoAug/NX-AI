"""阶段 5：本地声纹门、话题窗口、免唤醒候选与资料访问的单测（全部用假声纹）。"""

from __future__ import annotations

import numpy as np
import pytest

from aieyes.config import AppConfig, SpeakerConfig, TimersConfig
from aieyes.identity import (
    IdentityManager,
    IdentityResult,
    IdentityStatus,
    SpeakerGate,
    TopicWindow,
)


class FakeEmbedder:
    """把音频统计量当特征：均值/绝对均值二维向量，便于构造同一/不同人。"""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0

    @property
    def dim(self) -> int:
        return 2

    def embed(self, samples: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        self.calls += 1
        if self.fail:
            raise RuntimeError("embedding backend crashed")
        v = np.asarray(samples, dtype=np.float32).reshape(-1)
        feats = np.array([float(v.mean()), float(np.abs(v).mean())], dtype=np.float32)
        norm = float(np.linalg.norm(feats)) or 1.0
        return feats / norm


def _voice(mean: float, seconds: float = 1.0) -> np.ndarray:
    return np.full(int(16000 * seconds), mean, dtype=np.float32)


def _gate(**cfg_kwargs) -> SpeakerGate:
    cfg = SpeakerConfig(min_speech_ms=500, **cfg_kwargs)
    return SpeakerGate(embedder=FakeEmbedder(), cfg=cfg)


# ---------------------------------------------------------------------------
# SpeakerGate
# ---------------------------------------------------------------------------
def test_gate_enroll_and_verify_same_and_other_voice() -> None:
    gate = _gate()
    assert gate.enrolled is False
    assert gate.enroll(_voice(0.2)) is True
    assert gate.enroll(_voice(0.2)) is True
    assert gate.enrolled is True

    same = gate.verify(_voice(0.2))
    assert same.status is IdentityStatus.VERIFIED
    assert same.score > 0.9, same
    assert same.reason == "match"
    assert "campplus-zh-cn-16k@v1" in same.profile_version

    other = gate.verify(_voice(-0.2))
    assert other.status is IdentityStatus.MISMATCH
    assert other.score < 0.65, other
    assert other.reason == "below_threshold"


def test_gate_rejects_short_and_extra_samples() -> None:
    gate = _gate(max_samples=2)
    assert gate.enroll(np.zeros(1000, dtype=np.float32)) is False, "太短不能注册"
    assert gate.enroll(_voice(0.1)) is True
    assert gate.enroll(_voice(0.1)) is True
    assert gate.enroll(_voice(0.1)) is False, "超过 max_samples 不再接收"

    short = gate.verify(np.zeros(1000, dtype=np.float32))
    assert short.status is IdentityStatus.UNCERTAIN
    assert short.reason == "short_sample"


def test_gate_without_profile_is_unavailable() -> None:
    gate = _gate()
    result = gate.verify(_voice(0.1))
    assert result.status is IdentityStatus.UNAVAILABLE
    assert result.reason == "no_profile"


def test_gate_embed_failure_is_unavailable() -> None:
    gate = SpeakerGate(embedder=FakeEmbedder(fail=True), cfg=SpeakerConfig(min_speech_ms=500))
    # 注册也会失败，用直接注入档案的方式构造“有档案但算不出”的场景
    gate._embeddings.append(np.array([1.0, 0.0], dtype=np.float32))  # noqa: SLF001
    result = gate.verify(_voice(0.1))
    assert result.status is IdentityStatus.UNAVAILABLE
    assert result.reason.startswith("embed_error:")
    assert gate.profile_version.endswith("min_speech=500ms")


def test_gate_close_is_idempotent() -> None:
    gate = _gate()
    gate.enroll(_voice(0.1))
    gate.close()
    gate.close()


# ---------------------------------------------------------------------------
# TopicWindow
# ---------------------------------------------------------------------------
def test_topic_window_lifecycle() -> None:
    window = TopicWindow(window_ms=5000)
    assert window.open is False and window.remaining_ms(0) == 0
    window.open_at(1000, "answer")
    assert window.open is True
    assert window.deadline_ms == 6000
    assert window.is_open(1000) and window.is_open(5999)
    assert not window.is_open(6000)
    assert window.remaining_ms(2000) == 4000
    window.close()
    assert window.open is False


# ---------------------------------------------------------------------------
# IdentityManager：访问权限与免唤醒候选
# ---------------------------------------------------------------------------
def _manager(now: list[int] | None = None) -> IdentityManager:
    clock = now if now is not None else [0]
    gate = _gate()
    gate.enroll(_voice(0.2))
    timers = TimersConfig()
    return IdentityManager(gate, timers, topic_window_ms=timers.t21_topic_window_ms, now=lambda: clock[0])


def _result(status: IdentityStatus, score: float = 0.8) -> IdentityResult:
    return IdentityResult(status, score, status.value, 1200, "campplus-zh-cn-16k@v1")


def test_access_requires_recent_verified_identity() -> None:
    clock = [10_000]
    manager = _manager(clock)
    assert manager.access_ok() is False
    manager.note(_result(IdentityStatus.VERIFIED), now_ms=10_000)
    assert manager.access_ok() is True
    clock[0] = 10_000 + manager.access_max_age_ms
    assert manager.access_ok() is True
    clock[0] = 10_000 + manager.access_max_age_ms + 1
    assert manager.access_ok() is False, "超过访问有效期必须重新确认"
    manager.note(_result(IdentityStatus.MISMATCH), now_ms=clock[0])
    assert manager.access_ok() is False


def test_candidate_without_topic_is_ignored() -> None:
    manager = _manager()
    decision = manager.decide_candidate(_result(IdentityStatus.VERIFIED), "今天天气", directed=True)
    assert decision.action == "ignore"
    assert decision.reason == "no_topic_window"


def test_candidate_verified_directed_accepted() -> None:
    manager = _manager()
    manager.open_topic(1000)
    decision = manager.decide_candidate(
        _result(IdentityStatus.VERIFIED), "今天天气怎么样？", directed=True, now_ms=2000
    )
    assert decision.action == "accept"
    assert decision.reason == "verified"
    assert manager.candidates_accepted == 1


def test_candidate_verified_undirected_ignored() -> None:
    manager = _manager()
    manager.open_topic(1000)
    decision = manager.decide_candidate(
        _result(IdentityStatus.VERIFIED), "这个真好吃", directed=False, now_ms=2000
    )
    assert decision.action == "ignore"
    assert decision.reason == "not_directed"


def test_candidate_mismatch_ignored_without_prompt() -> None:
    manager = _manager()
    manager.open_topic(1000)
    decision = manager.decide_candidate(
        _result(IdentityStatus.MISMATCH, 0.2), "今天天气怎么样？", directed=True, now_ms=2000
    )
    assert decision.action == "ignore"
    assert decision.reason == "identity_mismatch"
    assert manager.candidates_rejected == 1


def test_candidate_unavailable_requires_explicit_wake() -> None:
    manager = _manager()
    manager.open_topic(1000)
    decision = manager.decide_candidate(
        _result(IdentityStatus.UNAVAILABLE, 0.0), "今天天气怎么样？", directed=True, now_ms=2000
    )
    assert decision.action == "need_explicit_wake"
    assert decision.reason == "voiceprint_unavailable"


def test_candidate_uncertain_prompts_once_then_accepts_reentry() -> None:
    manager = _manager()
    manager.open_topic(1000)
    uncertain = _result(IdentityStatus.UNCERTAIN, 0.55)
    decision = manager.decide_candidate(uncertain, "今天天气怎么样？", directed=True, now_ms=2000)
    assert decision.action == "reentry"
    assert decision.prompt and "小冰" in decision.prompt
    assert manager.reentry_active is True

    # 15 秒内重入并核验通过 → 处理暂存的原话
    decision2 = manager.decide_candidate(
        _result(IdentityStatus.VERIFIED), "小冰", directed=True, now_ms=3000
    )
    assert decision2.action == "accept"
    assert decision2.reason == "reentry_confirmed"
    assert decision2.retry_text == "今天天气怎么样？"
    assert manager.reentry_active is False


def test_candidate_reentry_denied_is_released() -> None:
    manager = _manager()
    manager.open_topic(1000)
    manager.decide_candidate(
        _result(IdentityStatus.UNCERTAIN, 0.55), "今天天气怎么样？", directed=True, now_ms=2000
    )
    decision = manager.decide_candidate(
        _result(IdentityStatus.MISMATCH, 0.2), "小冰", directed=True, now_ms=3000
    )
    assert decision.action == "release"
    assert decision.reason == "reentry_denied"
    assert manager.reentry_active is False


def test_candidate_uncertain_cooldown_suppresses_second_prompt() -> None:
    manager = _manager()
    manager.open_topic(1000)
    uncertain = _result(IdentityStatus.UNCERTAIN, 0.55)
    first = manager.decide_candidate(uncertain, "今天天气怎么样？", directed=True, now_ms=2000)
    assert first.action == "reentry"
    manager.release_candidate()
    second = manager.decide_candidate(uncertain, "今天天气怎么样？", directed=True, now_ms=5000)
    assert second.action == "ignore"
    assert second.reason == "uncertain_cooldown"


def test_candidate_budget_exceeded_releases() -> None:
    manager = _manager()
    manager.open_topic(1000)
    manager.begin_candidate(1000)
    decision = manager.decide_candidate(
        _result(IdentityStatus.VERIFIED),
        "今天天气怎么样？",
        directed=True,
        onset_ms=1000,
        now_ms=1000 + manager.candidate_budget_ms() + 1,
    )
    assert decision.action == "release"
    assert decision.reason == "t19_budget"


def test_topic_and_candidate_reset() -> None:
    manager = _manager()
    manager.open_topic(1000)
    manager.begin_candidate(1000)
    manager.open_topic(2000, label="again")
    assert manager.topic_open(2000) is True
    manager.close_topic()
    assert manager.topic_open(2000) is False
    assert manager.reentry_active is False


def test_budget_is_sum_of_t19_phases() -> None:
    timers = TimersConfig(t19_candidate_ms=1000, t19_extend_ms=2000, t19_final_ms=3000)
    manager = IdentityManager(_gate(), timers)
    assert manager.candidate_budget_ms() == 6000


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
