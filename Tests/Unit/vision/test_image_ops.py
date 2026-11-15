"""image_ops 单元测试（纯 numpy/PIL，不依赖相机/模型）。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aieyes.vision import image_ops


def test_ensure_rgb_from_gray_and_rgba():
    gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
    rgb = image_ops.ensure_rgb(gray)
    assert rgb.shape == (3, 4, 3) and rgb.dtype == np.uint8
    assert np.array_equal(rgb[:, :, 0], gray)

    rgba = np.zeros((2, 3, 4), dtype=np.uint8)
    rgba[..., 3] = 128
    assert image_ops.ensure_rgb(rgba).shape == (2, 3, 3)


def test_ensure_rgb_float_0_1_scaled():
    arr = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    rgb = image_ops.ensure_rgb(arr)
    assert rgb[0, 0].tolist() == [0, 128, 255]


def test_ensure_rgb_rejects_bad_shape():
    with pytest.raises(ValueError):
        image_ops.ensure_rgb(np.zeros((2, 2, 5), dtype=np.uint8))


def test_rgb_bgr_roundtrip():
    rgb = np.random.default_rng(0).integers(0, 256, (8, 9, 3), dtype=np.uint8)
    assert np.array_equal(image_ops.bgr_to_rgb(image_ops.rgb_to_bgr(rgb)), rgb)


def test_gray_conversion_constant_color():
    rgb = np.full((5, 7, 3), 200, dtype=np.uint8)
    assert np.allclose(image_ops.to_gray(rgb), 200, atol=1)


def test_yuyv_to_rgb_neutral_gray():
    # Y=128, U=V=128 的中性灰：BT.601 有限范围公式约还原到 130
    width, height = 4, 2
    pixel = np.array([128, 128, 128, 128], dtype=np.uint8)
    data = np.tile(pixel, width // 2 * height)
    rgb = image_ops.yuyv_to_rgb(data, width, height)
    assert rgb.shape == (height, width, 3) and rgb.dtype == np.uint8
    assert np.all(np.abs(rgb.astype(int) - 130) <= 6), rgb


def test_yuyv_to_rgb_known_colors():
    # V4L2 YUYV 打包：Y0 U Y1 V，每 4 字节 2 像素；纯红/纯白 1x2 图
    red = image_ops.yuyv_to_rgb(np.array([76, 84, 76, 255], dtype=np.uint8), 2, 1)
    assert red.shape == (1, 2, 3)
    assert np.array_equal(red[0, 0], np.array([255, 0, 0], dtype=np.uint8)), red
    white = image_ops.yuyv_to_rgb(np.array([235, 128, 235, 128], dtype=np.uint8), 2, 1)
    assert np.all(white >= 250), white


def test_yuyv_to_rgb_odd_width_rejected():
    with pytest.raises(ValueError):
        image_ops.yuyv_to_rgb(np.zeros(5, dtype=np.uint8), 1, 2)


def test_yuyv_to_rgb_2d_input():
    width, height = 4, 2
    rows = np.tile(np.array([100, 120, 110, 130], dtype=np.uint8), (height, width // 2))
    rgb = image_ops.yuyv_to_rgb(rows, width, height)
    assert rgb.shape == (height, width, 3)


def test_yuyv_to_rgb_too_small():
    with pytest.raises(ValueError):
        image_ops.yuyv_to_rgb(np.zeros(4, dtype=np.uint8), 4, 2)


def test_resize_max_side_keeps_aspect():
    rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    out = image_ops.resize_max_side(rgb, 320)
    assert out.shape == (240, 320, 3)
    # 已经够小则原样返回副本
    small = np.ones((10, 12, 3), dtype=np.uint8)
    same = image_ops.resize_max_side(small, 320)
    assert np.array_equal(same, small) and same is not small


def test_resize_keep_aspect_pad_multiple():
    rgb = np.zeros((50, 100, 3), dtype=np.uint8)
    out, sx, sy = image_ops.resize_keep_aspect(rgb, max_side_px=64, pad_multiple=32)
    assert out.shape == (32, 64, 3)
    assert pytest.approx(sx, rel=1e-6) == 0.64
    assert pytest.approx(sy, rel=1e-6) == 0.64


def test_crop_roi_and_clip():
    img = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
    crop = image_ops.crop_roi(img, (2, 3, 6, 8))
    assert crop.shape == (5, 4, 3)
    assert np.array_equal(crop, img[3:8, 2:6])
    clipped = image_ops.crop_roi(img, (-5, -5, 4, 4))
    assert clipped.shape == (4, 4, 3)
    with pytest.raises(ValueError):
        image_ops.crop_roi(img, (20, 20, 30, 30))  # 完全在图外


def test_expand_box_unclip():
    x1, y1, x2, y2 = image_ops.expand_box((10.0, 20.0, 110.0, 30.0), 1.5)
    # w=100,h=10, area=1000, perimeter=220 → d=6.8
    assert x1 < 10 and x2 > 110 and y1 < 20 and y2 > 30
    assert (x2 - x1) > 100 and (y2 - y1) > 10


def test_jpeg_roundtrip_and_decode():
    rgb = np.zeros((32, 48, 3), dtype=np.uint8)
    rgb[:, :] = (200, 30, 40)
    data = image_ops.encode_jpeg(rgb, quality=95)
    assert isinstance(data, bytes) and data[:2] == b"\xff\xd8"
    decoded = image_ops.decode_image(data)
    assert decoded.shape == (32, 48, 3)
    assert np.all(np.abs(decoded.astype(int) - np.array([200, 30, 40])).max(axis=-1) < 20)


def test_save_and_load_image(tmp_path):
    rgb = np.zeros((20, 30, 3), dtype=np.uint8)
    rgb[..., 1] = 255
    path = tmp_path / "x.png"
    image_ops.save_image(path, rgb)
    loaded = image_ops.load_image(path)
    assert np.array_equal(loaded, rgb)
    assert image_ops.image_size(loaded) == (30, 20)


def test_load_image_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        image_ops.load_image(tmp_path / "nope.png")


def test_pad_to():
    rgb = np.full((4, 5, 3), 7, dtype=np.uint8)
    out = image_ops.pad_to(rgb, 8, 6, fill_value=0)
    assert out.shape == (6, 8, 3)
    assert np.array_equal(out[:4, :5], rgb)
    assert out[4, 0, 0] == 0 and out[0, 5, 0] == 0


def test_pil_image_input():
    img = Image.new("RGB", (7, 5), (10, 20, 30))
    assert image_ops.ensure_rgb(img).shape == (5, 7, 3)
