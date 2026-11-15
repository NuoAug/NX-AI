"""FileCamera / load_frame / list_images 单元测试。"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from aieyes.clock import mono_ms
from aieyes.errors import CameraUnavailable
from aieyes.vision import FileCamera, list_images, load_frame


def _write(path, size, color):
    Image.new("RGB", size, color).save(path)
    return path


@pytest.fixture()
def image_dir(tmp_path):
    _write(tmp_path / "a.png", (64, 48), (255, 0, 0))
    _write(tmp_path / "b.jpg", (40, 30), (0, 255, 0))
    (tmp_path / "note.txt").write_text("not an image", encoding="utf-8")
    return tmp_path


def test_list_images_filters_and_sorts(image_dir):
    names = [p.name for p in list_images(image_dir)]
    assert names == ["a.png", "b.jpg"]


def test_list_images_missing_dir(tmp_path):
    with pytest.raises(CameraUnavailable):
        list_images(tmp_path / "missing")


def test_load_frame_metadata(image_dir):
    before = mono_ms()
    frame = load_frame(image_dir / "a.png")
    assert frame.rgb.shape == (48, 64, 3)
    assert frame.rgb.dtype == np.uint8
    assert frame.source == "file"
    assert frame.frame_id.startswith("file-")
    assert before - 1000 <= frame.t_capture_ms <= mono_ms() + 1000
    assert frame.width == 64 and frame.height == 48


def test_load_frame_content_id_is_stable(tmp_path):
    a = _write(tmp_path / "same1.png", (8, 8), (1, 2, 3))
    b = _write(tmp_path / "same2.png", (8, 8), (1, 2, 3))
    assert load_frame(a).frame_id == load_frame(b).frame_id


def test_load_frame_missing_file(tmp_path):
    with pytest.raises(CameraUnavailable):
        load_frame(tmp_path / "nope.png")


def test_load_frame_bad_image(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"definitely not a png")
    with pytest.raises(CameraUnavailable):
        load_frame(bad)


def test_file_camera_directory_loops(image_dir):
    cam = FileCamera(image_dir)
    assert not cam.opened
    cam.open()
    try:
        assert cam.opened and cam.count == 2
        frames = [cam.grab() for _ in range(5)]
        assert [f.rgb.shape[:2] for f in frames[:2]] == [(48, 64), (30, 40)]
        # 第 3 帧应回到第一张（循环）
        assert np.array_equal(frames[0].rgb, frames[2].rgb)
        assert frames[0].frame_id == frames[2].frame_id
        assert cam.grabs == 5
    finally:
        cam.close()
    assert not cam.opened


def test_file_camera_single_file(image_dir):
    cam = FileCamera(image_dir / "b.jpg")
    cam.open()
    frame = cam.grab()
    assert frame.rgb.shape == (30, 40, 3)
    assert cam.count == 1


def test_file_camera_list_and_exhausted(image_dir):
    paths = [image_dir / "a.png", image_dir / "b.jpg"]
    cam = FileCamera(paths, loop=False)
    cam.open()
    assert cam.grab().rgb.shape[:2] == (48, 64)
    assert cam.grab().rgb.shape[:2] == (30, 40)
    with pytest.raises(CameraUnavailable):
        cam.grab()
    # reset 后可重新取
    cam.reset()
    assert cam.grab().rgb.shape[:2] == (48, 64)


def test_file_camera_grab_before_open(image_dir):
    cam = FileCamera(image_dir)
    with pytest.raises(CameraUnavailable):
        cam.grab()


def test_file_camera_missing_source(tmp_path):
    cam = FileCamera(tmp_path / "missing")
    with pytest.raises(CameraUnavailable):
        cam.open()


def test_file_camera_empty_directory(tmp_path):
    (tmp_path / "readme.txt").write_text("x", encoding="utf-8")
    cam = FileCamera(tmp_path)
    with pytest.raises(CameraUnavailable):
        cam.open()
