"""取图 provider 单元测试（重试、预算、失败类型、文件图源）。"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from aieyes.errors import CameraUnavailable
from aieyes.scene import (
    CameraFrameProvider,
    FileFrameProvider,
    FrameAcquisitionError,
    FramePolicy,
    FrameQualityError,
)
from scene_fakes import FakeCamera, make_frame

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "Tests" / "Fixtures" / "images"


class SlowCamera(FakeCamera):
    """每次 grab 固定耗时，用来验证 T09 总预算而不是 3×grab_timeout。"""

    def __init__(self, frames, delay_s=0.05):
        super().__init__(frames)
        self.delay_s = delay_s

    def grab(self, timeout_ms: int = 10000):
        time.sleep(self.delay_s)
        return super().grab(timeout_ms)


def _camera_provider(frames):
    policy = FramePolicy(grab_timeout_ms=5000, max_attempts=3, retry_delay_ms=0)
    return CameraFrameProvider(camera=FakeCamera(frames), policy=policy, sleep=lambda _s: None)


def test_camera_retries_then_succeeds():
    provider = _camera_provider([make_frame("flat", frame_id="f1"), make_frame("flat", frame_id="f2"), make_frame("sharp", frame_id="f3")])
    frame = provider.acquire("ocr")

    assert frame.frame_id == "f3"
    assert provider.last_acquisition is not None
    assert len(provider.last_acquisition.attempts) == 3
    assert provider.last_acquisition.purpose == "ocr"
    assert provider.last_acquisition.quality is not None and provider.last_acquisition.quality.clear


def test_camera_quality_failure_after_three_attempts():
    provider = _camera_provider([make_frame("flat", frame_id=f"f{i}") for i in range(3)])
    with pytest.raises(FrameQualityError) as excinfo:
        provider.acquire("vqa")

    exc = excinfo.value
    assert len(exc.attempts) == 3
    assert exc.frame is not None and exc.frame.frame_id == "f2"
    assert exc.report is not None and exc.report.is_blurry
    assert exc.cause == "quality"


def test_camera_acquisition_failure_is_not_quality_error():
    provider = _camera_provider([CameraUnavailable("EBUSY: /dev/video4") for _ in range(3)])
    with pytest.raises(FrameAcquisitionError) as excinfo:
        provider.acquire("vqa")

    exc = excinfo.value
    assert not isinstance(exc, FrameQualityError)
    assert len(exc.attempts) == 3
    assert "EBUSY" in exc.cause
    assert provider.last_acquisition is None


def test_camera_respects_total_budget():
    policy = FramePolicy(grab_timeout_ms=80, max_attempts=3, retry_delay_ms=0)
    provider = CameraFrameProvider(
        camera=SlowCamera([make_frame("flat", frame_id=f"f{i}") for i in range(3)]),
        policy=policy,
        sleep=lambda _s: None,
    )
    with pytest.raises(FrameQualityError) as excinfo:
        provider.acquire("vqa")

    assert len(excinfo.value.attempts) <= 2  # 80ms 预算内最多两次 50ms 取图
    assert sum(attempt.elapsed_ms for attempt in excinfo.value.attempts) < 1000


def test_camera_bright_frame_is_reported_but_not_retried():
    provider = _camera_provider([make_frame("white_noise", frame_id="fb"), make_frame("sharp", frame_id="fs")])
    frame = provider.acquire("ocr")

    assert frame.frame_id == "fb"
    assert provider.last_acquisition is not None
    assert len(provider.last_acquisition.attempts) == 1
    assert "bright" in provider.last_acquisition.quality.labels


def test_file_provider_reads_real_fixture():
    path = FIXTURE_DIR / "text_cn_price.png"
    if not path.is_file():
        pytest.skip(f"缺少夹具 {path}")
    provider = FileFrameProvider(path)
    frame = provider.acquire("ocr")

    assert frame.source == "file"
    assert frame.frame_id.startswith("file-")
    assert (frame.width, frame.height) == (960, 540)
    assert provider.last_acquisition is not None and provider.last_acquisition.purpose == "ocr"


def test_file_provider_blurry_file_is_reported():
    path = FIXTURE_DIR.parent / "scene" / "blurry_label.png"
    if not path.is_file():
        pytest.skip(f"缺少夹具 {path}")
    provider = FileFrameProvider(path)
    frame = provider.acquire("ocr")

    assert frame.frame_id
    assert provider.last_acquisition is not None
    assert "blurry" in provider.last_acquisition.quality.labels


def test_file_provider_missing_file():
    with pytest.raises(FrameAcquisitionError):
        FileFrameProvider(FIXTURE_DIR / "nope.png").acquire("ocr")


def test_file_provider_strict_quality_raises():
    path = FIXTURE_DIR.parent / "scene" / "blurry_label.png"
    if not path.is_file():
        pytest.skip(f"缺少夹具 {path}")
    provider = FileFrameProvider(path, policy=FramePolicy(require_clear=True, max_attempts=1))
    with pytest.raises(FrameQualityError):
        provider.acquire("ocr")
