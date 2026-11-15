"""目标板真实 VLM 联调：llama.cpp server（OpenAI 兼容）+ 合成图问答。

server 未启动时 skip；Lead 启动 llama-server 后运行本文件验证图文问答与耗时。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aieyes.config import VlmConfig
from aieyes.vision import VlmClient

pytestmark = pytest.mark.target

FIXTURES = Path(__file__).resolve().parents[2] / "Fixtures" / "images"


def _skip_if_not_ready(client: VlmClient) -> None:
    if not client.health():
        pytest.skip(
            f"llama.cpp server not ready at {client.llm.base_url}: {client.last_error}"
        )


def _color_image(rgb=(220, 30, 30), size=(320, 240)) -> np.ndarray:
    image = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    image[:, :] = (240, 240, 240)
    image[40:200, 60:260] = rgb
    return image


def test_vlm_real_color_question():
    client = VlmClient()
    _skip_if_not_ready(client)
    image = _color_image()
    completion = client.complete_detailed(
        "这张图中间方块是什么颜色？只回答颜色名称。",
        images=[image],
        max_tokens=32,
        timeout_s=120.0,
    )
    print(
        f"\nVLM text={completion.text!r} t_ms={completion.t_ms} "
        f"usage={completion.usage} attempts={completion.attempts}"
    )
    assert completion.text
    assert completion.images == 1
    assert completion.attempts == 1
    assert completion.usage, "server 未返回 usage"
    lowered = completion.text.lower()
    assert "红" in completion.text or "red" in lowered, completion.text


def test_vlm_real_text_ocr_question():
    client = VlmClient()
    _skip_if_not_ready(client)
    image = np.full((240, 480, 3), 255, dtype=np.uint8)
    # 用 PIL 写出高对比度数字，问 VLM 图里的数字
    from PIL import Image, ImageDraw, ImageFont

    pil = Image.fromarray(image)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 80
        )
    except OSError:
        font = ImageFont.load_default(size=80)
    draw.text((60, 70), "12.50", font=font, fill=(0, 0, 0))
    image = np.asarray(pil)
    completion = client.complete_detailed(
        "图中写的数字是多少？只回答数字。",
        images=[image],
        max_tokens=32,
        timeout_s=120.0,
    )
    print(f"\nVLM OCR text={completion.text!r} t_ms={completion.t_ms} usage={completion.usage}")
    assert completion.text
    assert "12" in completion.text or "12.50" in completion.text


def test_vlm_real_two_image_compare():
    client = VlmClient()
    _skip_if_not_ready(client)
    left = _color_image((220, 30, 30))  # 红
    right = _color_image((30, 60, 220))  # 蓝
    completion = client.complete_detailed(
        "第一张图和第二张图分别是什么颜色？用一句话回答。",
        images=[left, right],
        max_tokens=48,
        timeout_s=120.0,
    )
    print(f"\nVLM compare text={completion.text!r} t_ms={completion.t_ms} usage={completion.usage}")
    assert completion.text
    assert completion.images == 2
    assert "红" in completion.text or "red" in completion.text.lower()
    assert "蓝" in completion.text or "blue" in completion.text.lower()


def test_vlm_real_price_label_fixture_reads_text():
    """阶段 3 核心链路：价格标签/日期图（fixtures 静态图）→ VLM 直接读文字。"""
    client = VlmClient()
    _skip_if_not_ready(client)
    from aieyes.vision.image_ops import load_image

    price_path = FIXTURES / "text_cn_price.png"
    assert price_path.is_file(), f"缺少 fixture: {price_path}"
    image = load_image(price_path)
    completion = client.complete_detailed(
        "请读出图中所有文字，原样返回，不要解释。",
        images=[image],
        max_tokens=200,
        temperature=0.1,
    )
    print(
        f"price-label text={completion.text!r} t_ms={completion.t_ms} "
        f"usage={completion.usage} images={completion.images}"
    )
    assert completion.images == 1
    assert "12.50" in completion.text, completion.text
    assert "2026" in completion.text, completion.text
