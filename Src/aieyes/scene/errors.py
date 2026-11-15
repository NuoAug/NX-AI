"""scene 层异常。

都继承 ``aieyes.errors.VisionError``，不改动 ``errors.py``（由 Lead 维护）：

* :class:`FrameAcquisitionError` —— 取图本身失败（设备不可用、超时、文件读不出）；
* :class:`FrameQualityError` —— 图取到了但质量不达标，三次尝试后仍不能用于 OCR/VQA；
  携带最后一帧与质量报告，调用方可以决定“降级成部分可读”还是直接答看不清；
* :class:`SceneAnalysisError` —— OCR / VLM 推理失败，调用方应降级而不是崩溃。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from aieyes.errors import VisionError

if TYPE_CHECKING:  # pragma: no cover - 仅类型
    from aieyes.interfaces import ImageFrame
    from aieyes.vision.quality import QualityReport

    from .models import FrameAttempt


class SceneError(VisionError):
    """scene 模块错误基类。"""


class FrameAcquisitionError(SceneError):
    """取图失败：设备不可用 / 超时 / 文件读取失败。"""

    def __init__(
        self,
        message: str,
        *,
        attempts: tuple[FrameAttempt, ...] = (),
        cause: str = "",
    ) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.cause = cause


class FrameQualityError(FrameAcquisitionError):
    """多次尝试后画面质量仍不达标（模糊/欠曝/过曝/尺寸非法）。

    ``frame`` 是最后一次取到的帧，``report`` 是它的质量报告；调用方可以
    选择把这一帧当“部分可读”继续分析，而不是直接拒绝回答。
    """

    def __init__(
        self,
        message: str,
        *,
        frame: ImageFrame | None = None,
        report: QualityReport | None = None,
        attempts: tuple[FrameAttempt, ...] = (),
        cause: str = "quality",
    ) -> None:
        super().__init__(message, attempts=attempts, cause=cause)
        self.frame = frame
        self.report = report


class SceneAnalysisError(SceneError):
    """OCR / VLM 推理失败（降级处理，不向上抛成崩溃）。"""

    def __init__(self, message: str, *, cause: str = "", provider: str = "") -> None:
        super().__init__(message)
        self.cause = cause
        self.provider = provider
