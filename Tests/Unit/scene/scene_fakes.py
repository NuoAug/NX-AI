"""scene 单元测试用的假 OCR / VLM / 相机与造帧工具。"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from aieyes.interfaces import ImageFrame, OcrLine, OcrResult


class FakeOcr:
    """按行返回预置文本；可按 frame_id 区分（compare 双图用）。``error`` 抛异常。"""

    def __init__(
        self,
        lines: Iterable[str | OcrLine] = (),
        *,
        by_frame: dict[str, Iterable[str | OcrLine]] | None = None,
        error: Exception | None = None,
    ):
        self._lines = list(lines)
        self._by_frame = {key: list(value) for key, value in (by_frame or {}).items()}
        self._error = error
        self.calls: list[tuple[int, str]] = []

    def recognize(self, rgb: np.ndarray, *, source_frame_id: str = "") -> OcrResult:
        self.calls.append((int(rgb.shape[0]), source_frame_id))
        if self._error is not None:
            raise self._error
        items = self._by_frame.get(source_frame_id, self._lines)
        lines = tuple(_as_line(item, index) for index, item in enumerate(items))
        return OcrResult(
            lines=lines,
            t_ms=1,
            source_frame_id=source_frame_id,
            image_size=(int(rgb.shape[1]), int(rgb.shape[0])),
        )


class FakeVlm:
    """返回固定文本（或 ``reply_factory(prompt)``）；``error`` 非空时抛异常。"""

    def __init__(self, reply: str = "", *, reply_factory=None, error: Exception | None = None):
        self.reply = reply
        self.reply_factory = reply_factory
        self.error = error
        self.calls: list[dict] = []

    def complete(
        self,
        prompt: str,
        *,
        images: Sequence[np.ndarray] = (),
        max_tokens: int = 512,
        timeout_s: float = 120.0,
        temperature: float | None = None,
    ) -> str:
        self.calls.append(
            {
                "prompt": prompt,
                "images": len(images),
                "max_tokens": max_tokens,
                "timeout_s": timeout_s,
                "temperature": temperature,
            }
        )
        if self.error is not None:
            raise self.error
        if self.reply_factory is not None:
            return self.reply_factory(prompt)
        return self.reply


class FakeCamera:
    """依次返回预置帧；某一项是异常实例时抛出它（模拟 EBUSY/超时）。"""

    name = "fake-camera"

    def __init__(self, frames: Iterable[ImageFrame | Exception]):
        self._frames = list(frames)
        self.grabs = 0
        self.opened = False

    def open(self) -> None:
        self.opened = True

    def grab(self, timeout_ms: int = 10000) -> ImageFrame:
        self.grabs += 1
        if not self._frames:
            raise RuntimeError("fake camera 没有更多帧")
        item = self._frames.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        self.opened = False


def make_frame(kind: str = "sharp", *, width: int = 640, height: int = 480, frame_id: str = "") -> ImageFrame:
    """造可控制质量标签的帧：sharp(噪声)/flat(模糊)/dark/bright。"""
    rng = np.random.default_rng(20261111)
    if kind == "sharp":
        rgb = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    elif kind == "flat":
        rgb = np.full((height, width, 3), 128, dtype=np.uint8)
    elif kind == "dark":
        rgb = np.full((height, width, 3), 8, dtype=np.uint8)
    elif kind == "bright":
        rgb = np.full((height, width, 3), 255, dtype=np.uint8)
    elif kind == "white_noise":
        # 明亮但有纹理：只触发 bright 标签，不触发 blurry（模拟白纸/白墙）
        rgb = rng.integers(230, 256, size=(height, width, 3), dtype=np.uint8)
    else:
        raise ValueError(f"未知 kind: {kind}")
    return ImageFrame(
        rgb=rgb,
        t_capture_ms=1000,
        frame_id=frame_id or f"fake-{kind}",
        source="fake",
    )


def solid_frame(color: tuple[int, int, int], *, width: int = 320, height: int = 240, frame_id: str = "") -> ImageFrame:
    rgb = np.full((height, width, 3), color, dtype=np.uint8)
    return ImageFrame(rgb=rgb, t_capture_ms=1000, frame_id=frame_id or "fake-solid", source="fake")


def make_lines(specs: Iterable[tuple[str, tuple[int, int, int, int], float]]) -> tuple[OcrLine, ...]:
    return tuple(
        OcrLine(text=text, box=box, score=score, polygon=_polygon(box))
        for text, box, score in specs
    )


def _as_line(item: str | OcrLine, index: int) -> OcrLine:
    if isinstance(item, OcrLine):
        return item
    box = (20, 40 + index * 40, 300, 30)
    return OcrLine(text=item, box=box, score=0.95, polygon=_polygon(box))


def _polygon(box: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    x, y, w, h = box
    return ((x, y), (x + w, y), (x + w, y + h), (x, y + h))
