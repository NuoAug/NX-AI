"""compare 双图同口径比较单元测试（价格/日期/颜色、缺侧与坏输入）。"""

from __future__ import annotations

from aieyes.scene import SceneService
from scene_fakes import FakeOcr, FakeVlm, make_frame, solid_frame


def _service(by_frame=None):
    return SceneService(ocr=FakeOcr(by_frame=by_frame or {}), vlm=FakeVlm(""))


def test_compare_price_deterministic():
    service = _service({"a": ["价格 12.5元"], "b": ["价格 15元"]})
    result = service.compare(make_frame(frame_id="a"), make_frame(frame_id="b"), "哪个更贵")

    assert result.common_items == ("价格",)
    assert result.missing == ()
    item = result.items[0]
    assert (item.name, item.unit, item.a, item.b, item.comparable) == ("价格", "元", "12元5角", "15元", True)
    assert "B更贵" in result.text and "15元" in result.text
    assert result.a_frame_id == "a" and result.b_frame_id == "b"
    assert set(result.timings) >= {"a_ocr", "b_ocr", "total"}


def test_compare_same_price():
    result = _service({"a": ["价格 12.5元"], "b": ["价格 12.5元"]}).compare(
        make_frame(frame_id="a"), make_frame(frame_id="b"), "价格一样吗"
    )

    assert result.items[0].comparable
    assert "价格一样" in result.text


def test_compare_expiry_date():
    service = _service({"a": ["到期日 2026-12-01"], "b": ["到期日 2026-06-01"]})
    result = service.compare(make_frame(frame_id="a"), make_frame(frame_id="b"), "哪个保质期长")

    item = result.items[0]
    assert item.name == "日期" and item.comparable
    assert item.a == "2026年12月1日" and item.b == "2026年6月1日"
    assert "A更晚" in result.text


def test_compare_date_missing_expiry():
    service = _service({"a": ["生产日期 2026-01-31"], "b": ["到期日 2026-06-01"]})
    result = service.compare(make_frame(frame_id="a"), make_frame(frame_id="b"), "日期")

    item = result.items[0]
    assert not item.comparable
    assert "没读到到期日" in item.note
    assert result.missing == ("日期",)


def test_compare_color_without_ocr():
    result = _service().compare(
        solid_frame((220, 30, 30), frame_id="red"),
        solid_frame((30, 60, 220), frame_id="blue"),
        "颜色一样吗",
    )

    item = result.items[0]
    assert (item.name, item.a, item.b, item.comparable) == ("颜色", "红色", "蓝色", True)
    assert "颜色不同" in result.text
    # 纯色图会被质量启发式判模糊，只做降级标注、不阻止颜色比较
    assert "low_quality" in result.degraded


def test_compare_bad_path():
    result = _service().compare("Tests/Fixtures/images/不存在.png", make_frame(frame_id="b"))

    assert "bad_input" in result.degraded
    assert "打不开" in result.text


def test_compare_no_common_items():
    result = _service().compare(make_frame(frame_id="a"), make_frame(frame_id="b"), "价格和日期")

    assert result.common_items == ()
    assert set(result.missing) == {"价格", "日期"}
    assert "没能按同一口径比较" in result.text


def test_compare_default_question_checks_three_items():
    result = _service({"a": ["价格 12.5元"], "b": ["价格 15元"]}).compare(
        make_frame(frame_id="a"), make_frame(frame_id="b")
    )

    assert [item.name for item in result.items] == ["价格", "日期", "颜色"]
    assert "价格" in result.common_items
    assert "日期" in result.missing
