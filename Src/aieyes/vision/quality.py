"""图像质量检查：清晰度（灰度拉普拉斯方差）、亮度、尺寸合法性。

阈值都是针对“640x480 左右、正常曝光的取景照片”的经验值，只作粗筛：
用户/流程需要的是“这张图是否值得送 OCR/VLM”，不是摄影测量。
数值口径：

* ``blur_score`` = 灰度图 3x3 拉普拉斯响应的方差。清晰文字/边缘通常 >100，
  失焦图通常 <30。可按场景用参数覆盖。
* ``brightness`` = 灰度均值（0..255）。``<40`` 视为过暗、``>235`` 视为过曝。

清晰度阈值与分辨率相关：同样的模糊在放大图上方差会变小，因此报告里保留
原始 ``round`` 值，调用方按实际相机档位校准一次即可。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .image_ops import ensure_rgb, to_gray_float

#: 默认清晰度阈值（640x480 档位的经验值）
DEFAULT_BLUR_THRESHOLD = 60.0
#: 默认亮度下限 / 上限（灰度均值）
DEFAULT_DARK_THRESHOLD = 40.0
DEFAULT_BRIGHT_THRESHOLD = 235.0
#: 尺寸合法性：短边下限与长边上限（像素）
DEFAULT_MIN_SIDE = 32
DEFAULT_MAX_SIDE = 4096


def laplacian_variance(rgb: np.ndarray) -> float:
    """灰度拉普拉斯方差，越大越清晰。接受 RGB（HxWx3/4）或灰度（HxW）。"""
    arr = np.asarray(rgb)
    gray = to_gray_float(arr) if arr.ndim == 3 else np.asarray(arr, dtype=np.float32)
    if gray.shape[0] < 3 or gray.shape[1] < 3:
        return 0.0
    p = np.pad(gray, 1, mode="edge")
    lap = (
        p[:-2, 1:-1]
        + p[2:, 1:-1]
        + p[1:-1, :-2]
        + p[1:-1, 2:]
        - 4.0 * p[1:-1, 1:-1]
    )
    return float(np.var(lap))


def brightness_mean(rgb: np.ndarray) -> float:
    """灰度均值（0..255）。"""
    arr = np.asarray(rgb)
    gray = to_gray_float(arr) if arr.ndim == 3 else np.asarray(arr, dtype=np.float32)
    return float(np.mean(gray)) if gray.size else 0.0


@dataclass(frozen=True)
class QualityReport:
    """一次质量检查的结果；``labels`` 非空即“不干净”。"""

    width: int
    height: int
    blur_score: float
    brightness: float
    size_ok: bool
    labels: tuple[str, ...] = ()
    blur_checked: bool = True

    @property
    def clear(self) -> bool:
        """无任何告警（不模糊、不欠曝/过曝、尺寸合法）。"""
        return not self.labels

    @property
    def is_blurry(self) -> bool:
        return "blurry" in self.labels

    @property
    def is_dark(self) -> bool:
        return "dark" in self.labels

    @property
    def is_bright(self) -> bool:
        return "bright" in self.labels

    @property
    def detail(self) -> str:
        return "ok" if self.clear else ",".join(self.labels)

    def to_dict(self) -> dict[str, object]:
        return {
            "width": self.width,
            "height": self.height,
            "blur_score": round(self.blur_score, 2),
            "brightness": round(self.brightness, 2),
            "size_ok": self.size_ok,
            "labels": list(self.labels),
            "clear": self.clear,
            "blur_checked": self.blur_checked,
        }

    def to_contract_dict(self) -> dict[str, str]:
        """映射到 MCPT frame.acquire 响应里的 ``quality`` 三字段。

        ``glare`` 未实现（不做高光检测，填 unknown）；``exposure`` 由亮度均值判定；
        未做清晰度检查时 ``sharpness=unknown``，不能默认宣称 clear。
        """
        if not self.blur_checked:
            sharpness = "unknown"
        else:
            sharpness = "blurry" if self.is_blurry else "clear"
        if self.is_dark:
            exposure = "underexposed"
        elif self.is_bright:
            exposure = "overexposed"
        else:
            exposure = "normal"
        return {"sharpness": sharpness, "glare": "unknown", "exposure": exposure}


def check_quality(
    rgb: np.ndarray,
    *,
    blur_threshold: float = DEFAULT_BLUR_THRESHOLD,
    dark_threshold: float = DEFAULT_DARK_THRESHOLD,
    bright_threshold: float = DEFAULT_BRIGHT_THRESHOLD,
    min_side: int = DEFAULT_MIN_SIDE,
    max_side: int = DEFAULT_MAX_SIDE,
    compute_blur: bool = True,
) -> QualityReport:
    """检查一张图：清晰度 / 亮度 / 尺寸。

    ``labels`` 取值：``too_small``、``too_large``、``blurry``、``dark``、
    ``bright``。``clear=True`` 表示四项都通过，不代表内容里有需要的目标。
    """
    arr = ensure_rgb(rgb)
    h, w = arr.shape[:2]
    labels: list[str] = []
    short_side, long_side = min(h, w), max(h, w)
    if short_side < int(min_side):
        labels.append("too_small")
    if long_side > int(max_side):
        labels.append("too_large")
    blur = laplacian_variance(arr) if compute_blur else 0.0
    if compute_blur and blur < float(blur_threshold):
        labels.append("blurry")
    brightness = brightness_mean(arr)
    if brightness < float(dark_threshold):
        labels.append("dark")
    elif brightness > float(bright_threshold):
        labels.append("bright")
    return QualityReport(
        width=w,
        height=h,
        blur_score=blur,
        brightness=brightness,
        size_ok=not ("too_small" in labels or "too_large" in labels),
        labels=tuple(labels),
        blur_checked=bool(compute_blur),
    )
