"""生成阶段 4 双图比较用的两个单价格标签（A: 12.50 元，B: 9.90 元）。

在目标板 venv 里运行（需要 PIL + 中文字体）::

    cd /home/mylin/Desktop/Nuo && . .venv/bin/activate
    python Tools/make_compare_fixtures.py

输出到 ``Tests/Fixtures/scene/compare_price_a.png`` / ``compare_price_b.png``，
随仓库提交，供 ``Tests/Integration/test_compare_engine_real.py`` 使用。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "Tests" / "Fixtures" / "scene"
CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "C:/Windows/Fonts/msyh.ttc",
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in CJK_FONT_CANDIDATES:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _label(title: str, price: str) -> Image.Image:
    image = Image.new("RGB", (640, 480), (252, 252, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 40, 600, 440), outline=(20, 20, 20), width=6)
    draw.text((90, 120), title, font=_font(64), fill=(15, 15, 15))
    draw.text((90, 240), price, font=_font(96), fill=(10, 10, 10))
    return image


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    _label("价签 A", "12.50元").save(OUT / "compare_price_a.png")
    _label("价签 B", "9.90元").save(OUT / "compare_price_b.png")
    print("生成：compare_price_a.png / compare_price_b.png")


if __name__ == "__main__":
    main()
