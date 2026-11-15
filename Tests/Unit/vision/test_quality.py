"""quality 单元测试：合成清晰/模糊/过暗/过曝图验证阈值行为。"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from aieyes.vision import quality


def _text_like_image(width=640, height=480) -> np.ndarray:
    """白底黑字的合成“文档照”，边缘多、拉普拉斯方差大。"""
    img = Image.new("RGB", (width, height), (245, 245, 245))
    draw = ImageDraw.Draw(img)
    for i in range(12):
        y = 20 + i * 36
        draw.rectangle((20, y, 20 + 40 + (i * 7) % 300, y + 18), fill=(20, 20, 20))
    return np.asarray(img)


def test_laplacian_variance_orders_sharp_above_blurred():
    sharp = _text_like_image()
    blurred = np.asarray(
        Image.fromarray(sharp).filter(ImageFilter.GaussianBlur(radius=4))
    )
    sharp_score = quality.laplacian_variance(sharp)
    blurred_score = quality.laplacian_variance(blurred)
    assert sharp_score > blurred_score
    assert sharp_score > quality.DEFAULT_BLUR_THRESHOLD
    assert blurred_score < quality.DEFAULT_BLUR_THRESHOLD


def test_check_quality_clear_image():
    report = quality.check_quality(_text_like_image())
    assert report.clear, report.labels
    assert report.size_ok and report.width == 640 and report.height == 480
    assert not report.is_blurry
    assert 0 <= report.brightness <= 255
    assert report.detail == "ok"
    assert report.to_dict()["clear"] is True


def test_check_quality_blurry():
    blurred = np.asarray(
        Image.fromarray(_text_like_image()).filter(ImageFilter.GaussianBlur(radius=5))
    )
    report = quality.check_quality(blurred)
    assert report.is_blurry and not report.clear
    assert "blurry" in report.labels


def test_check_quality_dark_and_bright():
    dark = np.full((120, 160, 3), 5, dtype=np.uint8)
    report = quality.check_quality(dark)
    assert report.is_dark and "dark" in report.labels
    assert report.brightness < quality.DEFAULT_DARK_THRESHOLD

    bright = np.full((120, 160, 3), 252, dtype=np.uint8)
    report = quality.check_quality(bright)
    assert report.is_bright and "bright" in report.labels


def test_check_quality_size_limits():
    tiny = np.full((16, 16, 3), 128, dtype=np.uint8)
    report = quality.check_quality(tiny)
    assert not report.size_ok and "too_small" in report.labels

    huge = np.full((10, 5000, 3), 128, dtype=np.uint8)
    report = quality.check_quality(huge, max_side=4096)
    assert "too_large" in report.labels


def test_check_quality_can_skip_blur():
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    report = quality.check_quality(flat, compute_blur=False)
    assert not report.is_blurry
    assert report.blur_score == 0.0


def test_check_quality_accepts_frame_like_uint8():
    report = quality.check_quality(_text_like_image().astype(np.uint8))
    assert isinstance(report.blur_score, float)


def test_contract_dict_maps_checked_items():
    report = quality.check_quality(_text_like_image())
    quality_dict = report.to_contract_dict()
    assert quality_dict == {
        "sharpness": "clear",
        "glare": "unknown",
        "exposure": "normal",
    }


def test_contract_dict_reports_blur_and_darkness():
    blurred = np.asarray(
        Image.fromarray(_text_like_image()).filter(ImageFilter.GaussianBlur(radius=5))
    )
    assert quality.check_quality(blurred).to_contract_dict()["sharpness"] == "blurry"

    dark = np.full((120, 160, 3), 5, dtype=np.uint8)
    assert quality.check_quality(dark).to_contract_dict()["exposure"] == "underexposed"

    bright = np.full((120, 160, 3), 252, dtype=np.uint8)
    assert quality.check_quality(bright).to_contract_dict()["exposure"] == "overexposed"


def test_contract_dict_does_not_claim_unchecked_sharpness():
    flat = np.full((100, 100, 3), 128, dtype=np.uint8)
    report = quality.check_quality(flat, compute_blur=False)
    assert report.blur_checked is False
    assert report.to_contract_dict()["sharpness"] == "unknown"
    assert report.to_dict()["blur_checked"] is False
