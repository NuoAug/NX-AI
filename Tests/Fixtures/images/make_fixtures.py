"""生成离线评测/演示用静态 fixture 图（在目标板 venv 里运行）。

用法::

    cd /home/mylin/Desktop/Nuo && . .venv/bin/activate
    python Tests/Fixtures/images/make_fixtures.py

生成的文件随仓库提交（Tests/Fixtures/images/），供 FileCamera 离线评测、
质量检查与 VLM 联调复用；不依赖相机。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in CJK_FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _text_image(lines: list[str], size: tuple[int, int], font_size: int) -> Image.Image:
    image = Image.new("RGB", size, (250, 250, 248))
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    y = 40
    for line in lines:
        draw.text((40, y), line, font=font, fill=(15, 15, 15))
        y += font_size + 36
    return image


def _scene_with_square(color: tuple[int, int, int], size: tuple[int, int]) -> Image.Image:
    width, height = size
    # 浅灰渐变背景 + 居中彩色方块（用于 VLM 颜色/位置问答）
    base = np.linspace(200, 235, height, dtype=np.uint8)
    rgb = np.repeat(base[:, None], width, axis=1)
    image = Image.fromarray(np.stack([rgb, rgb, rgb], axis=-1), mode="RGB")
    draw = ImageDraw.Draw(image)
    margin_x, margin_y = width // 5, height // 5
    draw.rectangle(
        (margin_x, margin_y, width - margin_x, height - margin_y), fill=color
    )
    return image


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)

    price = _text_image(
        ["价格 12.50 元", "日期 2026-11-11", "AI_EYES 测试 12345"],
        size=(960, 540),
        font_size=44,
    )
    price.save(HERE / "text_cn_price.png")

    price.filter(ImageFilter.GaussianBlur(radius=4)).save(HERE / "text_cn_price_blur.png")

    ascii_image = _text_image(
        ["HELLO AI-EYES", "No.12345", "2026/11/11"],
        size=(640, 480),
        font_size=40,
    )
    ascii_image.save(HERE / "text_ascii.png")

    _scene_with_square((220, 30, 30), (640, 480)).save(HERE / "scene_red_square.png")
    _scene_with_square((30, 60, 220), (320, 240)).save(HERE / "scene_blue_square.png")

    print(f"fixtures written to {HERE}")
    for path in sorted(HERE.glob("*.png")):
        with Image.open(path) as image:
            print(f"  {path.name}: {image.size[0]}x{image.size[1]} {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
