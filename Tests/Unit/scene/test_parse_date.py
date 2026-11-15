"""日期解析单元测试（生产/到期/批号/保质期/无标签）。"""

from __future__ import annotations

from datetime import date

import pytest

from aieyes.scene import extract_dates, format_date_cn, parse_date, pick_date
from scene_fakes import make_lines


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("2026-11-11", date(2026, 11, 11)),
        ("2026年11月11日", date(2026, 11, 11)),
        ("2026.11.11", date(2026, 11, 11)),
        ("2026/11/11", date(2026, 11, 11)),
    ],
)
def test_parse_date_positive(text, expected):
    assert parse_date(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "2026-13-45",  # 非法月日
        "2026-02-30",  # 非法日
        "20261201",  # 裸 8 位没有标签，不能直接认定为日期
        "批号 20261111",
        "保质期 12个月",
        "AI_EYES 测试",
    ],
)
def test_parse_date_negative(text):
    assert parse_date(text) is None


def test_extract_dates_fields_and_batch_exclusion():
    lines = make_lines(
        [
            ("生产日期 2026-01-31", (0, 0, 200, 20), 0.95),
            ("到期日 2026-12-01", (0, 30, 200, 20), 0.95),
            ("批号 20261111", (0, 60, 200, 20), 0.9),
            ("保质期 12个月", (0, 90, 200, 20), 0.9),
        ]
    )
    facts = extract_dates(lines)
    by_field = {fact.field: fact for fact in facts}
    assert by_field["production_date"].value == date(2026, 1, 31)
    assert by_field["expiry_date"].value == date(2026, 12, 1)
    assert by_field["batch_id"].value is None  # 批号只保留原文
    assert by_field["batch_id"].raw_text == "批号 20261111"
    assert by_field["shelf_life"].value is None

    picked = pick_date(facts)
    assert picked is not None and picked.field == "expiry_date"
    assert picked.value == date(2026, 12, 1)


def test_extract_dates_exp_keyword_and_bare_date():
    facts = extract_dates(make_lines([("EXP 2026-12-01", (0, 0, 160, 20), 0.9)]))
    assert [(fact.field, fact.value) for fact in facts] == [("expiry_date", date(2026, 12, 1))]

    bare = extract_dates(make_lines([("2026-12-01", (0, 0, 120, 20), 0.9)]))
    assert [(fact.field, fact.value) for fact in bare] == [("unlabeled_date", date(2026, 12, 1))]


def test_format_date_cn():
    assert format_date_cn(date(2026, 1, 31)) == "2026年1月31日"
    assert format_date_cn(date(2026, 12, 1)) == "2026年12月1日"
