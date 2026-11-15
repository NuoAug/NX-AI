"""文件取图相机：阶段 0 离线评测、双图对比/翻面测试的图源。

实现 ``interfaces.Camera`` 协议（``open`` / ``grab`` / ``close``），与
:class:`aieyes.vision.camera.V4L2Camera` 输出同样形状的 ``ImageFrame``，
上层只按帧处理，不必区分真机与文件。

``t_capture_ms`` 对文件图没有真实拍摄时刻，取“读入时刻”的单调时钟
（``evidence_mode`` 为 replay/synthetic 时应如此标注）；``frame_id`` 由
像素内容摘要生成，同一文件内容稳定，重拍/换图才会变化。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

from aieyes.clock import mono_ms
from aieyes.errors import CameraUnavailable
from aieyes.interfaces import ImageFrame

from . import image_ops

#: 视为图片的扩展名（小写，含点）
IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
)


def list_images(directory: str | Path) -> list[Path]:
    """按文件名排序列出目录下的图片文件（不递归）。"""
    root = Path(directory)
    if not root.is_dir():
        raise CameraUnavailable(f"image directory not found: {root}")
    return sorted(
        p
        for p in root.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def _content_frame_id(rgb) -> str:
    digest = hashlib.sha1(image_ops.ensure_rgb(rgb).tobytes()).hexdigest()[:12]
    return f"file-{digest}"


def load_frame(path: str | Path, *, source: str = "file") -> ImageFrame:
    """读取单张图片为 ``ImageFrame``；失败统一抛 ``CameraUnavailable``。"""
    p = Path(path)
    try:
        rgb = image_ops.load_image(p)
    except FileNotFoundError as exc:
        raise CameraUnavailable(f"image not found: {p}") from exc
    except Exception as exc:  # PIL 的解码错误种类很多，统一归为相机不可用
        raise CameraUnavailable(f"failed to decode image {p}: {exc}") from exc
    return ImageFrame(
        rgb=rgb,
        t_capture_ms=mono_ms(),
        frame_id=_content_frame_id(rgb),
        source=source,
    )


class FileCamera:
    """从目录/文件列表循环取图的 ``Camera`` 实现。

    参数
    ----
    source : 目录、单个文件路径，或文件路径序列。
    loop   : True=按序列循环取图；False=取完后再取报 ``CameraUnavailable``。
    """

    def __init__(
        self,
        source: str | Path | Sequence[str | Path],
        *,
        loop: bool = True,
    ) -> None:
        self._source = source
        self.loop = bool(loop)
        self._paths: list[Path] = []
        self._index = 0
        self._opened = False
        self._grabs = 0
        self.last_error = ""

    # --- Camera 协议 ---------------------------------------------------
    def open(self) -> None:
        self._paths = self._resolve_paths()
        if not self._paths:
            raise CameraUnavailable(
                f"no image files under {self._source!r}; "
                f"supported extensions: {', '.join(IMAGE_EXTENSIONS)}"
            )
        self._index = 0
        self._grabs = 0
        self._opened = True

    def grab(self, timeout_ms: int = 10_000) -> ImageFrame:
        if not self._opened:
            raise CameraUnavailable("FileCamera.grab() called before open()")
        if self._index >= len(self._paths):
            if not self.loop:
                raise CameraUnavailable(
                    f"file camera exhausted after {self._grabs} grab(s): "
                    f"{len(self._paths)} image(s) consumed"
                )
            self._index = 0
        path = self._paths[self._index]
        self._index += 1
        self._grabs += 1
        try:
            return load_frame(path)
        except CameraUnavailable as exc:
            self.last_error = str(exc)
            raise

    def close(self) -> None:
        self._opened = False

    # --- 辅助 ----------------------------------------------------------
    @property
    def opened(self) -> bool:
        return self._opened

    @property
    def paths(self) -> list[Path]:
        return list(self._paths)

    @property
    def count(self) -> int:
        return len(self._paths)

    @property
    def grabs(self) -> int:
        return self._grabs

    def reset(self) -> None:
        self._index = 0
        self._grabs = 0

    def _resolve_paths(self) -> list[Path]:
        source = self._source
        if isinstance(source, (str, Path)):
            path = Path(source)
            if path.is_dir():
                return list_images(path)
            if path.is_file():
                return [path]
            raise CameraUnavailable(f"file camera source not found: {path}")
        paths: list[Path] = []
        for item in source:
            p = Path(item)
            if not p.is_file():
                raise CameraUnavailable(f"file camera source not found: {p}")
            paths.append(p)
        return paths
