"""目标板真实 OCR 测试：PP-OCRv6 small + PIL 合成中文图（Noto CJK）。

测定单图耗时（阶段 3 的 P95 ≤3s 目标）、识别文本、原图像素坐标、置信度。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from aieyes.config import ModelsConfig, OcrConfig
from aieyes.errors import ModelUnavailable
from aieyes.vision import OcrEngine

pytestmark = pytest.mark.target

CJK_FONT_CANDIDATES = (
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Medium.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
)


def _find_font() -> str | None:
    for path in CJK_FONT_CANDIDATES:
        if Path(path).is_file():
            return path
    return None


def _render(lines: list[str], size=(640, 480), font_size=40) -> np.ndarray:
    font_path = _find_font()
    if font_path is not None:
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default(size=font_size)
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    y = 40
    for line in lines:
        draw.text((40, y), line, font=font, fill=(10, 10, 10))
        y += font_size + 40
    return np.asarray(image)


def _ocr_dir() -> Path:
    models = ModelsConfig()
    return Path(models.resolve(models.ocr_dir))


def _require_models() -> Path:
    ocr_dir = _ocr_dir()
    if not (ocr_dir / OcrConfig().det_model).is_file():
        pytest.skip(f"OCR models not uploaded yet: {ocr_dir}")
    return ocr_dir


def test_load_models_and_char_dict():
    _require_models()
    engine = OcrEngine()
    engine.load()
    assert engine.last_init_ms >= 0
    assert len(engine._chars) > 18_000  # 模型 onnx metadata 里的中文字典
    assert engine.dict_source.startswith("onnx-metadata:"), engine.dict_source
    assert engine._rec_classes == 18_710
    engine.close()


def test_recognize_chinese_price_and_date():
    _require_models()
    engine = OcrEngine()
    engine.load()
    image = _render(
        ["价格 12.50 元", "日期 2026-11-11", "AI_EYES 测试 12345"],
        size=(640, 480),
        font_size=40,
    )
    engine.warmup()  # 首次推理含 ORT 预热，不计入耗时统计
    durations: list[int] = []
    result = None
    for _ in range(3):
        t0 = time.monotonic()
        result = engine.recognize(image, source_frame_id="synthetic-ocr-1")
        durations.append(int((time.monotonic() - t0) * 1000))
    assert result is not None
    assert result.source_frame_id == "synthetic-ocr-1"
    assert result.image_size == (640, 480)
    assert result.lines, "合成图上没有检测到任何文本行"
    text = result.text
    print("\nOCR text lines:")
    for line in result.lines:
        print(f"  {line.text!r} box={line.box} score={line.score:.3f}")
    print(f"t_ms_reported={result.t_ms} durations={durations}")
    compact = text.replace(" ", "")
    assert "12.50" in compact, f"未识别出价格: {text!r}"
    assert "2026" in compact, f"未识别出日期: {text!r}"
    # 坐标是原图像素：第一行在上半部、日期行在中间
    for line in result.lines:
        x, y, w, h = line.box
        assert 0 <= x < 640 and 0 <= y < 480
        assert 0 < w <= 640 and 0 < h <= 480
        assert 0.0 <= line.score <= 1.0
        assert line.score >= OcrConfig().min_confidence
        assert len(line.polygon) == 4
    price_lines = [ln for ln in result.lines if "12.50" in ln.text.replace(" ", "")]
    assert price_lines and price_lines[0].box[1] < 160, price_lines
    date_lines = [ln for ln in result.lines if "2026" in ln.text.replace(" ", "")]
    assert date_lines and 80 <= date_lines[0].box[1] < 300, date_lines
    # 单图耗时（P95 ≤3s 目标）：合成 640x480 上取最优值应远小于 3s
    assert min(durations) < 3000, durations
    engine.close()


def test_recognize_blank_image_returns_no_lines():
    _require_models()
    engine = OcrEngine()
    engine.load()
    blank = np.full((240, 320, 3), 255, dtype=np.uint8)
    result = engine.recognize(blank)
    print(f"\nblank image: lines={[ln.text for ln in result.lines]} t_ms={result.t_ms}")
    assert result.lines == ()


def test_missing_models_raise_model_unavailable(tmp_path):
    engine = OcrEngine(ocr_dir=tmp_path)
    with pytest.raises(ModelUnavailable):
        engine.load()


def test_recognize_upside_down_text_uses_cls():
    """180° 倒置文本应经方向分类翻转后正确识别（cls 输出也是概率，不能二次 softmax）。"""
    _require_models()
    engine = OcrEngine()
    engine.load()
    upright = _render(["价格 12.50 元"], size=(640, 480), font_size=48)
    upside = np.ascontiguousarray(np.rot90(upright, 2))
    engine.warmup()
    result = engine.recognize(upside, source_frame_id="synthetic-rotated")
    print(f"\nupside-down OCR: lines={[(ln.text, ln.box, ln.score) for ln in result.lines]}")
    assert result.lines, "倒置文本没有检出"
    compact = result.text.replace(" ", "")
    assert "12.50" in compact, f"倒置文本识别错误: {result.text!r}"
    assert all(ln.score >= OcrConfig().min_confidence for ln in result.lines)
    engine.close()

