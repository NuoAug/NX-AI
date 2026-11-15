"""目标板真实相机（Intel RealSense D435i 的 UVC 彩色节点）测试。

本机无法访问设备时 pytest 会 skip；失败信息里带 v4l2 探测诊断。
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from aieyes.clock import mono_ms
from aieyes.config import VisionConfig
from aieyes.errors import CameraUnavailable
from aieyes.vision import V4L2Camera, probe_video_devices

pytestmark = pytest.mark.target


def test_probe_video_devices_finds_color_node():
    infos = probe_video_devices()
    assert infos, "没有 /dev/video* 设备"
    report = "\n".join(info.describe() for info in infos)
    infos_with_color = [info for info in infos if info.color_formats]
    assert infos_with_color, f"没有 YUYV/MJPG 彩色节点：\n{report}"


def test_v4l2_camera_grab_realtime():
    config = VisionConfig()
    camera = V4L2Camera(config)
    try:
        camera.open()
    except CameraUnavailable as exc:
        pytest.skip(f"camera unavailable on target board: {exc}")
    try:
        assert camera.width > 0 and camera.height > 0
        assert camera.pixel_format in ("YUYV", "MJPG")
        assert camera.is_open
        times: list[float] = []
        frames = []
        for _ in range(3):
            t0 = time.monotonic()
            frame = camera.grab(timeout_ms=5000)
            times.append((time.monotonic() - t0) * 1000.0)
            frames.append(frame)
        first = frames[0]
        assert first.rgb.dtype == np.uint8
        assert first.rgb.ndim == 3 and first.rgb.shape[2] == 3
        assert first.rgb.shape == (camera.height, camera.width, 3)
        assert first.source == "camera" and first.frame_id
        assert abs(first.t_capture_ms - mono_ms()) < 10_000
        # 帧唯一（连续取流不应重复）
        assert len({f.frame_id for f in frames}) == 3
        # 相机对着真实场景时应有亮度变化（不是全黑/全白恒定帧）
        assert first.rgb.std() > 1.0
        print(
            f"\ncamera={camera.describe()} open_ms={camera.open_ms} "
            f"grab_ms={[round(t, 1) for t in times]} "
            f"size={camera.width}x{camera.height} fmt={camera.pixel_format} "
            f"mean_brightness={first.rgb.mean():.1f} std={first.rgb.std():.1f}"
        )
    finally:
        camera.close()
    assert not camera.is_open


def test_camera_describe_after_open():
    camera = V4L2Camera()
    try:
        camera.open()
    except CameraUnavailable as exc:
        pytest.skip(f"camera unavailable on target board: {exc}")
    try:
        description = camera.describe()
        assert camera.device_path in description
        assert camera.pixel_format in description
    finally:
        camera.close()
