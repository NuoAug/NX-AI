"""静态 fixture 图冒烟测试：FileCamera 可直接读取、质量检查可用。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aieyes.vision import FileCamera, check_quality, load_image

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "Fixtures" / "images"
EXPECTED = (
    "text_cn_price.png",
    "text_cn_price_blur.png",
    "text_ascii.png",
    "scene_red_square.png",
    "scene_blue_square.png",
)


def test_fixture_images_present():
    assert FIXTURE_DIR.is_dir(), f"missing fixture dir: {FIXTURE_DIR}"
    for name in EXPECTED:
        assert (FIXTURE_DIR / name).is_file(), name


def test_fixture_images_load_and_have_expected_sizes():
    price = load_image(FIXTURE_DIR / "text_cn_price.png")
    assert price.shape == (540, 960, 3)
    red = load_image(FIXTURE_DIR / "scene_red_square.png")
    assert red.shape == (480, 640, 3)
    assert red[240, 320, 0] > red[240, 320, 2]  # 中心偏红


def test_file_camera_reads_fixture_dir():
    camera = FileCamera(FIXTURE_DIR)
    camera.open()
    try:
        assert camera.count >= len(EXPECTED)
        frame = camera.grab()
        assert frame.rgb.dtype == np.uint8 and frame.frame_id.startswith("file-")
    finally:
        camera.close()


def test_fixture_quality_flags_blur():
    sharp = check_quality(load_image(FIXTURE_DIR / "text_cn_price.png"))
    blurred = check_quality(load_image(FIXTURE_DIR / "text_cn_price_blur.png"))
    assert sharp.blur_score > blurred.blur_score
    assert blurred.is_blurry and not blurred.clear
    assert pytest.approx(blurred.width) == 960
