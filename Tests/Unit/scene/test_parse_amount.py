"""金额解析与来源标注单元测试（§5 价格正反例）。"""

from __future__ import annotations

from decimal import Decimal

import pytest

from aieyes.scene import extract_amounts, format_amount_cn, parse_amount
from scene_fakes import make_lines


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("12.50元", Decimal("12.50")),
        ("￥12.50", Decimal("12.50")),
        ("12元5角", Decimal("12.50")),
        ("12元5角3分", Decimal("12.53")),
        ("12块5", Decimal("12.50")),
        ("价格 12.5", Decimal("12.50")),
        ("1,234.50元", Decimal("1234.50")),
        ("3件 12.5元", Decimal("12.50")),
        ("12元 5件", Decimal("12")),  # 5 件是数量，不能拼成 12.5
        ("12.50", Decimal("12.50")),
    ],
)
def test_parse_amount_positive(text, expected):
    assert parse_amount(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "-12.50元",  # 负数不是价签
        "12.567元",  # 三位小数通常是规格/批号，不是金额
        "AI_EYES 测试 12345",  # 编号不是价格
        "2026-11-11",  # 日期行不产金额
        "批号 20261111",
        "已售罄",
    ],
)
def test_parse_amount_negative(text):
    assert parse_amount(text) is None


def test_extract_amounts_labels_fields_and_raw_text():
    facts = extract_amounts(make_lines([("原价 15元", (0, 0, 100, 20), 0.9), ("现价 12.5元", (0, 30, 100, 20), 0.9)]))
    assert [(fact.field, fact.value) for fact in facts] == [
        ("original_price", Decimal("15")),
        ("promo_price", Decimal("12.50")),
    ]
    assert facts[0].raw_text == "原价 15元"
    assert facts[0].line_index == 0


def test_extract_amounts_adjacent_price_tags_stays_two_candidates():
    lines = make_lines([("价签A 12.50元", (0, 0, 100, 20), 0.9), ("价签B 9.90元", (0, 30, 100, 20), 0.9)])
    values = [fact.value for fact in extract_amounts(lines)]
    assert values == [Decimal("12.50"), Decimal("9.90")]


def test_format_amount_cn():
    assert format_amount_cn(Decimal("12.50")) == "12元5角"
    assert format_amount_cn(Decimal("12.05")) == "12元零5分"
    assert format_amount_cn(Decimal("12")) == "12元"
