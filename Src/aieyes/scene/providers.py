"""取图提供者：相机与图片文件两条来源，统一 ``acquire(purpose) -> ImageFrame``。

两个实现共享同一套质量门控：

* 每帧都做尺寸/亮度/模糊启发式检查（``aieyes.vision.quality``）；
* 相机在 :class:`FramePolicy` 规定的总预算（默认 T09=10s）内最多尝试
  ``max_attempts`` 次（T10=3 次）；
* 尝试用尽仍不达标时抛 :class:`FrameQualityError`，并携带最后一帧与质量报告，
  上层可以选择“降级成部分可读”；设备/超时类失败抛 :class:`FrameAcquisitionError`。

文件来源默认不拒绝低质量帧（文件重拍结果不变），只在 ``FrameAcquisition``
里报告质量，由 ``SceneService`` 决定是否标注降级。
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence, runtime_checkable

from aieyes.clock import mono_ms
from aieyes.interfaces import Camera, ImageFrame
from aieyes.vision.file_camera import FileCamera
from aieyes.vision.quality import QualityReport, check_quality

from .errors import FrameAcquisitionError, FrameQualityError
from .models import FrameAcquisition, FrameAttempt

#: 画面不达标时允许重试的标签（换一帧可能变好；过曝的白纸重拍也一样，不重试）
DEFAULT_RETRY_LABELS: tuple[str, ...] = ("blurry", "dark", "too_small")


@dataclass(frozen=True, slots=True)
class FramePolicy:
    """取图策略。

    ``grab_timeout_ms`` 是**整次 acquire** 的总预算（T09=10 秒）；
    ``max_attempts`` 是同一位置最多取几次（T10=3 次）。
    """

    grab_timeout_ms: int = 10_000
    max_attempts: int = 3
    require_clear: bool = True
    retry_labels: tuple[str, ...] = DEFAULT_RETRY_LABELS
    retry_delay_ms: int = 120
    compute_blur: bool = True


@runtime_checkable
class FrameProvider(Protocol):
    """场景层取图接口。"""

    @property
    def name(self) -> str:
        """来源名称（camera / file …），写入 ``ImageFrame.source``。"""

    def acquire(self, purpose: str = "vqa") -> ImageFrame:
        """取一张可用于 OCR/VQA 的画面；失败抛 scene 层异常。"""

    def close(self) -> None:
        """释放底层资源（可重复调用）。"""


def _should_retry(report: QualityReport, policy: FramePolicy) -> bool:
    return any(label in policy.retry_labels for label in report.labels)


def _attempt_loop(
    camera: Camera,
    policy: FramePolicy,
    purpose: str,
    *,
    sleep,
) -> FrameAcquisition:
    """在总预算内反复取图，直到质量达标或次数用尽。"""
    start = mono_ms()
    attempts: list[FrameAttempt] = []
    last_frame: ImageFrame | None = None
    last_report: QualityReport | None = None
    last_error = ""
    max_attempts = max(1, int(policy.max_attempts))

    for index in range(max_attempts):
        remaining = policy.grab_timeout_ms - (mono_ms() - start)
        if remaining <= 0:
            last_error = f"取图超时（{policy.grab_timeout_ms} ms 预算用尽）"
            break
        t0 = mono_ms()
        try:
            frame = camera.grab(timeout_ms=max(1, min(remaining, policy.grab_timeout_ms)))
        except Exception as exc:  # noqa: BLE001 - 相机后端异常统一转成取图失败
            last_error = str(exc) or type(exc).__name__
            attempts.append(FrameAttempt(index=index, elapsed_ms=mono_ms() - t0, error=last_error))
        else:
            report = check_quality(frame.rgb, compute_blur=policy.compute_blur)
            last_frame, last_report = frame, report
            attempts.append(
                FrameAttempt(
                    index=index,
                    elapsed_ms=mono_ms() - t0,
                    frame_id=frame.frame_id,
                    quality=report,
                )
            )
            if (
                not policy.require_clear
                or report.clear
                or not _should_retry(report, policy)
            ):
                return _finish(frame, report, purpose, attempts, start)
            last_error = "画面质量不达标：" + "、".join(report.labels)

        if index < max_attempts - 1 and policy.retry_delay_ms > 0:
            sleep(policy.retry_delay_ms / 1000.0)

    elapsed = mono_ms() - start
    if last_frame is not None and last_report is not None:
        labels = "、".join(last_report.labels) or "未知"
        raise FrameQualityError(
            f"取图质量不达标（尝试 {len(attempts)} 次）：{labels}",
            frame=last_frame,
            report=last_report,
            attempts=tuple(attempts),
        )
    raise FrameAcquisitionError(
        f"取图失败（尝试 {len(attempts)} 次）：{last_error or '未知错误'}",
        attempts=tuple(attempts),
        cause=last_error,
    )


def _finish(
    frame: ImageFrame,
    report: QualityReport,
    purpose: str,
    attempts: Sequence[FrameAttempt],
    start: int,
) -> FrameAcquisition:
    return FrameAcquisition(
        frame=frame,
        purpose=purpose,
        attempts=tuple(attempts),
        quality=report,
        elapsed_ms=mono_ms() - start,
    )


class CameraFrameProvider:
    """从相机取图（默认 ``/dev/video4``，由 :class:`V4L2Camera` 自动选彩色节点）。"""

    def __init__(
        self,
        camera: Camera | None = None,
        *,
        device: str | None = None,
        policy: FramePolicy | None = None,
        sleep=time.sleep,
    ) -> None:
        self._camera = camera
        self._device = device
        self._owns_camera = camera is None
        self.policy = policy or FramePolicy()
        self._sleep = sleep
        self.last_acquisition: FrameAcquisition | None = None
        self.last_error = ""

    @property
    def name(self) -> str:
        return "camera"

    @property
    def camera(self) -> Camera:
        return self._ensure_camera()

    def _ensure_camera(self) -> Camera:
        if self._camera is None:
            # 延迟导入：V4L2Camera 依赖 fcntl，只在真的用相机时才需要
            from aieyes.vision.camera import V4L2Camera

            self._camera = V4L2Camera(device=self._device) if self._device else V4L2Camera()
        return self._camera

    def open(self) -> None:
        self._ensure_camera().open()

    def acquire(self, purpose: str = "vqa") -> ImageFrame:
        camera = self._ensure_camera()
        try:
            acquisition = _attempt_loop(camera, self.policy, purpose, sleep=self._sleep)
        except FrameAcquisitionError as exc:
            self.last_error = str(exc)
            frame = getattr(exc, "frame", None)
            if frame is not None:
                # 质量不达标但取到了帧：留下最后一次取图记录，供上层降级复用
                attempts = tuple(exc.attempts)
                self.last_acquisition = FrameAcquisition(
                    frame=frame,
                    purpose=purpose,
                    attempts=attempts,
                    quality=getattr(exc, "report", None),
                    elapsed_ms=sum(attempt.elapsed_ms for attempt in attempts),
                )
            raise
        self.last_acquisition = acquisition
        self.last_error = ""
        return acquisition.frame

    def close(self) -> None:
        camera, self._camera = self._camera, None
        if camera is not None:
            try:
                camera.close()
            except Exception:  # noqa: BLE001 - 释放设备失败不该掩盖原答案
                pass

    def __enter__(self) -> "CameraFrameProvider":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


class FileFrameProvider:
    """从图片文件/目录取图，供离线复跑与真机夹具测试。"""

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        *,
        policy: FramePolicy | None = None,
        sleep=time.sleep,
    ) -> None:
        self.source = source
        self.policy = policy or FramePolicy(
            grab_timeout_ms=5_000,
            max_attempts=1,
            require_clear=False,
        )
        self._sleep = sleep
        self._camera: FileCamera | None = None
        self.last_acquisition: FrameAcquisition | None = None
        self.last_error = ""

    @property
    def name(self) -> str:
        return "file"

    @property
    def camera(self) -> FileCamera:
        if self._camera is None:
            self._camera = FileCamera(self.source)
        return self._camera

    def open(self) -> None:
        self.camera.open()

    def acquire(self, purpose: str = "vqa") -> ImageFrame:
        camera = self.camera
        try:
            if not camera.opened:
                camera.open()
        except Exception as exc:  # noqa: BLE001 - CameraUnavailable 等统一转取图失败
            self.last_error = str(exc) or type(exc).__name__
            raise FrameAcquisitionError(
                f"打不开图片源：{self.last_error}", cause=self.last_error
            ) from exc
        try:
            acquisition = _attempt_loop(camera, self.policy, purpose, sleep=self._sleep)
        except FrameAcquisitionError as exc:
            self.last_error = str(exc)
            raise
        self.last_acquisition = acquisition
        self.last_error = ""
        return acquisition.frame

    def reset(self) -> None:
        if self._camera is not None:
            self._camera.reset()

    def close(self) -> None:
        camera, self._camera = self._camera, None
        if camera is not None:
            camera.close()

    def __enter__(self) -> "FileFrameProvider":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()


__all__ = [
    "DEFAULT_RETRY_LABELS",
    "CameraFrameProvider",
    "FileFrameProvider",
    "FramePolicy",
    "FrameProvider",
]
