"""生成 scene 场景测试图（价格/日期标签）。在目标板 venv 或本机 PIL 环境运行均可。

用法::

    cd /home/mylin/Desktop/Nuo && . .venv/bin/activate
    python Tests/Fixtures/scene/make_scene_fixtures.py

生成的文件随仓库提交（Tests/Fixtures/scene/），用于真实 OCR 的价格/日期正反例。
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
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


def _text_image(lines: list[str], size: tuple[int, int], font_size: int) -> Image.Image:
    image = Image.new("RGB", size, (250, 250, 248))
    draw = ImageDraw.Draw(image)
    font = _font(font_size)
    y = 30
    for line in lines:
        draw.text((30, y), line, font=font, fill=(15, 15, 15))
        y += font_size + 24
    return image


def _two_tags() -> Image.Image:
    """相邻两个价签：OCR 会读到 12.50 和 9.90，服务不应硬说对应哪个商品。"""
    image = Image.new("RGB", (800, 360), (245, 245, 242))
    draw = ImageDraw.Draw(image)
    font = _font(40)
    small = _font(28)
    for index, (title, price) in enumerate((("价签 A", "12.50元"), ("价签 B", "9.90元"))):
        x = 40 + index * 380
        draw.rectangle((x, 70, x + 340, 300), fill=(255, 255, 255), outline=(60, 60, 60), width=3)
        draw.text((x + 40, 100), title, font=small, fill=(40, 40, 40))
        draw.text((x + 40, 180), price, font=font, fill=(10, 10, 10))
    return image


def _date_label() -> Image.Image:
    """生产日期/到期日/批号各一行（批号行不应被当成日期）。"""
    return _text_image(
        ["生产日期 2026-01-31", "到期日 2026-12-01", "批号 20261111"],
        (760, 300),
        38,
    )


def _blurry(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.GaussianBlur(radius=6))


def main() -> None:
    tags = _two_tags()
    tags.save(HERE / "price_two_tags.png")

    dates = _date_label()
    dates.save(HERE / "date_label.png")

    _blurry(dates).save(HERE / "blurry_label.png")
    print("生成：price_two_tags.png / date_label.png / blurry_label.png")


if __name__ == "__main__":
    main()
