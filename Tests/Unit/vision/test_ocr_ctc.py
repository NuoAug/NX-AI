"""CTC 解码/概率归一化回归测试（不需要模型文件）。

背景：PP-OCRv6 rec 模型输出已是每行和为 1 的概率，早期实现再次 softmax
导致识别置信度被压到 ~1e-4，全部被 min_confidence 过滤（零文本行）。
"""

from __future__ import annotations

import numpy as np
import pytest

from aieyes.vision.ocr import OcrEngine, _probabilities

CHARS = ["A", "B", "C"]


def _engine() -> OcrEngine:
    engine = OcrEngine()
    engine._chars = list(CHARS)
    engine._rec_classes = len(CHARS) + 2  # blank=0, offset=1, 末尾空格类
    return engine


def test_probabilities_keeps_normalized_distribution():
    probs = np.array([[0.7, 0.2, 0.05, 0.03, 0.02]], dtype=np.float32)
    out = _probabilities(probs)
    assert out.shape == probs.shape
    assert float(out[0, 0]) == pytest.approx(0.7)


def test_probabilities_softmaxes_logits():
    logits = np.array([[5.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    out = _probabilities(logits)
    assert float(out.sum()) == pytest.approx(1.0)
    assert float(out[0, 0]) > 0.9


def test_probabilities_keeps_confident_cls_output():
    # cls 模型输出也是概率：[0, 1]，不能再次 softmax（会压到 0.73）
    out = _probabilities(np.array([1.6e-11, 1.0], dtype=np.float32))
    assert float(out[1]) > 0.99


def _onehot_probs(indices: list[int], n_classes: int = 5) -> np.ndarray:
    probs = np.full((len(indices), n_classes), 0.01, dtype=np.float32)
    probs[np.arange(len(indices)), indices] = 0.96
    return probs


def test_ctc_decode_probabilities_gives_high_score():
    engine = _engine()
    # A(1) A(1) blank(0) B(2) blank(0) 空格(4) C(3)
    probs = _onehot_probs([1, 1, 0, 2, 0, 4, 3])
    text, score = engine._ctc_decode(probs[None])
    assert text == "AB C"
    assert score > 0.9


def test_ctc_decode_blank_and_repeat_collapse():
    engine = _engine()
    probs = _onehot_probs([1, 1, 1, 0, 0, 2])
    text, score = engine._ctc_decode(probs[None])
    assert text == "AB"
    assert score > 0.9
