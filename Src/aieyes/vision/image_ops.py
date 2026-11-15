"""图像几何与颜色空间工具（仅 numpy + Pillow，不依赖 OpenCV）。

约定
----
* 图像数组一律 ``RGB``、``uint8``、``HxWx3``；其他形状用 :func:`ensure_rgb`
  归一（灰度会复制成三通道）。
* 坐标一律 ``(x, y)`` 像素、左上角为原点；ROI 为 ``(x1, y1, x2, y2)``，
  ``x2/y2`` 开区间（含义与 ``PIL.Image.crop`` 一致）。
* 缩放使用 Pillow 重采样；JPEG 编码/解码也走 Pillow，避免引入 OpenCV。
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

# Pillow 9.1+ 用 Image.Resampling 枚举；老版本退回模块常量。
_RS = getattr(Image, "Resampling", Image)
BILINEAR = _RS.BILINEAR
BICUBIC = _RS.BICUBIC
LANCZOS = _RS.LANCZOS

#: ITU-R BT.601 灰度权重（与 OpenCV/PIL ``convert("L")`` 同源）
_GRAY_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


# ---------------------------------------------------------------------------
# 归一与颜色空间
# ---------------------------------------------------------------------------
def ensure_rgb(image: np.ndarray | Image.Image) -> np.ndarray:
    """把 ``HxW`` / ``HxWx3`` / ``HxWx4`` / PIL.Image 统一成 ``uint8 HxWx3``。"""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    if not isinstance(image, np.ndarray):
        raise TypeError(f"expected numpy array or PIL.Image, got {type(image)!r}")
    arr = image
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        elif arr.shape[2] != 3:
            raise ValueError(f"unsupported channel count: {arr.shape}")
    else:
        raise ValueError(f"unsupported image shape: {arr.shape}")
    if arr.dtype != np.uint8:
        if arr.dtype.kind == "f" and arr.size and float(np.nanmax(arr)) <= 1.0:
            arr = arr * 255.0
        arr = np.clip(np.rint(np.asarray(arr, dtype=np.float32)), 0, 255).astype(
            np.uint8
        )
    return np.ascontiguousarray(arr)


def rgb_to_bgr(rgb: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(ensure_rgb(rgb)[:, :, ::-1])


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return rgb_to_bgr(bgr)


def to_gray_float(rgb: np.ndarray) -> np.ndarray:
    """转 float32 灰度（0..255），供拉普拉斯等数值计算用。"""
    arr = ensure_rgb(rgb).astype(np.float32)
    return arr @ _GRAY_WEIGHTS


def to_gray(rgb: np.ndarray) -> np.ndarray:
    """转 ``uint8`` 灰度图（HxW），供显示/调试用。"""
    return np.clip(np.rint(to_gray_float(rgb)), 0, 255).astype(np.uint8)


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    return ensure_rgb(gray)


def yuyv_to_rgb(data: bytes | np.ndarray, width: int, height: int) -> np.ndarray:
    """V4L2 ``YUYV``（Y0 U Y1 V，每 4 字节 2 像素）转 RGB。

    使用 BT.601 有限范围整数公式（与 libyuv/v4l2 常用实现一致）。
    """
    raw = (
        np.frombuffer(data, dtype=np.uint8)
        if not isinstance(data, np.ndarray)
        else np.asarray(data, dtype=np.uint8).reshape(-1)
    )
    need = int(width) * int(height) * 2
    if raw.size < need:
        raise ValueError(f"YUYV buffer too small: {raw.size} < {need}")
    if width % 2:
        raise ValueError(f"YUYV width must be even, got {width}")
    pack = raw[:need].reshape(height, width // 2, 4)
    y = np.empty((height, width), dtype=np.int32)
    y[:, 0::2] = pack[:, :, 0]
    y[:, 1::2] = pack[:, :, 2]
    u = np.repeat(pack[:, :, 1].astype(np.int32), 2, axis=1)
    v = np.repeat(pack[:, :, 3].astype(np.int32), 2, axis=1)
    c = y - 16
    d = u - 128
    e = v - 128
    r = (298 * c + 409 * e + 128) >> 8
    g = (298 * c - 100 * d - 208 * e + 128) >> 8
    b = (298 * c + 516 * d + 128) >> 8
    rgb = np.stack([r, g, b], axis=-1)
    return np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8))


# ---------------------------------------------------------------------------
# 缩放 / 裁剪 / 补齐
# ---------------------------------------------------------------------------
def _pil_resize(
    rgb: np.ndarray, width: int, height: int, resample: int = BILINEAR
) -> np.ndarray:
    img = Image.fromarray(ensure_rgb(rgb))
    return np.asarray(img.resize((int(width), int(height)), resample))


def resize_to(
    rgb: np.ndarray, width: int, height: int, *, resample: int = BILINEAR
) -> np.ndarray:
    arr = ensure_rgb(rgb)
    width, height = int(width), int(height)
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid resize target: {width}x{height}")
    if arr.shape[1] == width and arr.shape[0] == height:
        return arr.copy()
    return _pil_resize(arr, width, height, resample)


def resize_max_side(
    rgb: np.ndarray, max_side_px: int, *, resample: int = BILINEAR
) -> np.ndarray:
    """等比缩放到最长边不超过 ``max_side_px``；已足够小则原样返回副本。"""
    arr = ensure_rgb(rgb)
    if max_side_px is None or int(max_side_px) <= 0:
        return arr.copy()
    h, w = arr.shape[:2]
    scale = float(max_side_px) / float(max(h, w))
    if scale >= 1.0:
        return arr.copy()
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return _pil_resize(arr, nw, nh, resample)


def resize_keep_aspect(
    rgb: np.ndarray,
    *,
    max_side_px: int | None = None,
    target_width: int | None = None,
    target_height: int | None = None,
    pad_multiple: int | None = None,
    pad_value: int = 0,
    resample: int = BILINEAR,
) -> tuple[np.ndarray, float, float]:
    """等比缩放到目标框内，可选右下补齐到整数倍。

    返回 ``(image, scale_x, scale_y)``；``scale_x`` 为“缩放后 / 原图”的
    宽高倍数，坐标映射回原图时用 ``x_orig = x_scaled / scale_x``。
    只支持单一缩放倍数（宽高倍率相同）。
    """
    arr = ensure_rgb(rgb)
    h, w = arr.shape[:2]
    frame_w = int(target_width) if target_width else None
    frame_h = int(target_height) if target_height else None
    if max_side_px:
        frame_w = frame_w or int(max_side_px)
        frame_h = frame_h or int(max_side_px)
    if not frame_w or not frame_h:
        raise ValueError("resize_keep_aspect requires max_side_px or target size")
    scale = min(frame_w / float(w), frame_h / float(h))
    nw = max(1, min(frame_w, int(round(w * scale))))
    nh = max(1, min(frame_h, int(round(h * scale))))
    out = arr if (nw, nh) == (w, h) else _pil_resize(arr, nw, nh, resample)
    scale_x = nw / float(w)
    scale_y = nh / float(h)
    if pad_multiple:
        m = int(pad_multiple)
        target_w = int(np.ceil(nw / m)) * m
        target_h = int(np.ceil(nh / m)) * m
        out = pad_to(out, target_w, target_h, fill_value=pad_value)
    return out, scale_x, scale_y


def pad_to(
    rgb: np.ndarray, width: int, height: int, *, fill_value: int = 0
) -> np.ndarray:
    """左上对齐、右下补零到不小于目标尺寸（尺寸已足够则返回副本）。"""
    arr = ensure_rgb(rgb)
    h, w = arr.shape[:2]
    width, height = max(int(width), w), max(int(height), h)
    if (width, height) == (w, h):
        return arr.copy()
    out = np.full((height, width, 3), int(fill_value), dtype=np.uint8)
    out[:h, :w] = arr
    return out


def crop_roi(
    rgb: np.ndarray, roi_xyxy: tuple[int, int, int, int], *, clip: bool = True
) -> np.ndarray:
    """按 ``(x1, y1, x2, y2)``（开区间）裁剪；``clip=False`` 时越界报错。"""
    arr = ensure_rgb(rgb)
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = (int(round(float(v))) for v in roi_xyxy)
    cx1, cy1 = max(0, x1), max(0, y1)
    cx2, cy2 = min(w, x2), min(h, y2)
    if not clip and (cx1, cy1, cx2, cy2) != (x1, y1, x2, y2):
        raise ValueError(f"roi {roi_xyxy} outside image {w}x{h}")
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError(f"empty roi {roi_xyxy} for image {w}x{h}")
    return np.ascontiguousarray(arr[cy1:cy2, cx1:cx2])


def expand_box(
    box_xyxy: tuple[float, float, float, float], ratio: float
) -> tuple[float, float, float, float]:
    """按面积/周长比例膨胀轴对齐框（DB unclip 的矩形近似）。

    对矩形 ``w*h``：偏移量 ``d = area * ratio / perimeter``，宽高各加 ``2d``。
    """
    x1, y1, x2, y2 = (float(v) for v in box_xyxy)
    w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
    perimeter = 2.0 * (w + h)
    if perimeter <= 0:
        return (x1, y1, x2, y2)
    d = (w * h) * float(ratio) / perimeter
    return (x1 - d, y1 - d, x2 + d, y2 + d)


# ---------------------------------------------------------------------------
# 编解码 / 文件
# ---------------------------------------------------------------------------
def image_size(rgb: np.ndarray) -> tuple[int, int]:
    arr = np.asarray(rgb)
    return (int(arr.shape[1]), int(arr.shape[0]))


def encode_jpeg(rgb: np.ndarray, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(ensure_rgb(rgb)).save(
        buf, format="JPEG", quality=int(quality), optimize=False
    )
    return buf.getvalue()


def decode_image(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def load_image(path: str | Path) -> np.ndarray:
    """读图片文件为 RGB；按 EXIF 方向自动摆正。"""
    with Image.open(Path(path)) as img:
        return np.asarray(ImageOps.exif_transpose(img).convert("RGB"))


def save_image(path: str | Path, rgb: np.ndarray, *, quality: int | None = None) -> None:
    img = Image.fromarray(ensure_rgb(rgb))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if quality is not None or path.suffix.lower() in (".jpg", ".jpeg"):
        img.save(path, quality=int(quality if quality is not None else 90))
    else:
        img.save(path)
