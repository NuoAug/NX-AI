"""SceneService.answer 单元测试：OCR-first、价格/日期正反例、降级与来源。"""

from __future__ import annotations

from aieyes.errors import CameraUnavailable
from aieyes.scene import (
    CameraFrameProvider,
    FramePolicy,
    SceneService,
)
from scene_fakes import FakeCamera, FakeOcr, FakeVlm, make_frame


def _service(ocr_lines=(), *, vlm_reply="画面里有一个东西。", ocr_error=None, vlm_error=None, provider=None):
    return SceneService(
        ocr=FakeOcr(ocr_lines, error=ocr_error),
        vlm=FakeVlm(vlm_reply, error=vlm_error),
        provider=provider,
    )


def test_price_single_value_with_source_label():
    frame = make_frame(frame_id="f-price")
    answer = _service(["价格 12.50 元"]).answer("这个多少钱", frame=frame)

    assert answer.intent == "price"
    assert "12元5角" in answer.text
    assert answer.frame_id == "f-price"
    assert answer.confidence >= 0.8
    assert {tag.kind for tag in answer.source_tags} == {"ocr"}
    assert answer.source_tags[0].label == "包装"
    assert answer.source_tags[0].raw_text == "价格 12.50 元"
    assert "ocr" in answer.timings
    assert "total" in answer.timings
    assert answer.clear


def test_price_original_and_promo_are_both_read():
    answer = _service(["原价 15元", "现价 12.5元"]).answer("这个多少钱", frame=make_frame())

    assert "原价" in answer.text and "现价" in answer.text
    assert "12元5角" in answer.text
    assert answer.confidence >= 0.7


def test_price_adjacent_tags_is_ambiguous():
    frame = make_frame(frame_id="f-two")
    answer = _service(["价签A 12.50元", "价签B 9.90元"]).answer("这个多少钱", frame=frame)

    assert answer.intent == "price"
    assert "两个价格" in answer.text
    assert "ambiguous_price" in answer.degraded
    assert answer.confidence < 0.5
    assert not answer.clear


def test_price_not_found_is_spoken_clearly():
    answer = _service(["商品名称 洗衣液"]).answer("这个多少钱", frame=make_frame())

    assert "没读到明确的价格" in answer.text
    assert "no_price" in answer.degraded
    assert answer.confidence < 0.2


def test_date_production_and_expiry():
    frame = make_frame(frame_id="f-date")
    answer = _service(["生产日期 2026-01-31", "到期日 2026-12-01"]).answer("什么时候过期", frame=frame)

    assert answer.intent == "date"
    assert "到期日" in answer.text
    assert "2026年12月1日" in answer.text
    assert {tag.field for tag in answer.source_tags} >= {"expiry_date", "production_date"}
    assert answer.confidence >= 0.8


def test_date_batch_only_does_not_invent_expiry():
    answer = _service(["批号 20261111"]).answer("什么时候过期", frame=make_frame())

    assert "只读到批号" in answer.text
    assert "no_date" in answer.degraded
    assert "到期" not in answer.text


def test_read_text_joins_lines():
    frame = make_frame(frame_id="f-read")
    answer = _service(["价格 12.50 元", "日期 2026-11-11", "AI_EYES 测试 12345"]).answer("帮我读一下", frame=frame)

    assert answer.intent == "read"
    assert answer.text.startswith("画面上写着：")
    assert len(answer.ocr_lines) == 3
    assert "AI_EYES" in answer.text


def test_read_text_truncates_long_ocr():
    lines = [f"第{i}行文字内容" for i in range(40)]
    answer = _service(lines).answer("读一下", frame=make_frame())

    assert "后面还有字" in answer.text
    assert "text_truncated" in answer.degraded
    assert len(answer.text) < 400


def test_describe_uses_vlm_and_tags_frame():
    frame = make_frame(frame_id="f-desc")
    answer = _service([], vlm_reply="画面里是一个白色的纸盒。").answer("这是什么", frame=frame)

    assert answer.intent == "describe"
    assert answer.vlm_text == "画面里是一个白色的纸盒。"
    assert answer.text.startswith("画面里是一个白色的纸盒")
    assert {tag.kind for tag in answer.source_tags} == {"vlm"}
    assert answer.source_tags[0].label == "画面"


def test_purchase_calculation_needs_no_frame_and_no_models():
    answer = _service().answer("每件12块5，买3件一共多少钱")

    assert answer.intent == "calculate"
    assert answer.frame_id == ""
    assert "37元5角" in answer.text
    assert {tag.kind for tag in answer.source_tags} == {"spoken", "computed"}
    assert answer.timings.get("acquire", 0) == 0


def test_ocr_failure_degrades_without_crashing():
    answer = _service(ocr_error=RuntimeError("ocr down")).answer("这个多少钱", frame=make_frame())

    assert "ocr_failed" in answer.degraded
    assert "没读到" in answer.text
    assert "no_price" in answer.degraded


def test_vlm_failure_degrades_without_double_tag():
    answer = _service([], vlm_error=RuntimeError("vlm down")).answer("这是什么", frame=make_frame())

    assert "vlm_failed" in answer.degraded
    assert "vlm_empty" not in answer.degraded
    assert "没看清画面" in answer.text


def test_acquisition_failure_is_spoken():
    provider = CameraFrameProvider(
        camera=FakeCamera([CameraUnavailable("EBUSY") for _ in range(3)]),
        policy=FramePolicy(retry_delay_ms=0),
        sleep=lambda _s: None,
    )
    answer = SceneService(ocr=FakeOcr([]), vlm=FakeVlm(), provider=provider).answer("这个多少钱")

    assert "acquisition_failed" in answer.degraded
    assert "取不到画面" in answer.text
    assert answer.frame_id == ""


def test_low_quality_frame_is_partially_read():
    provider = CameraFrameProvider(
        camera=FakeCamera([make_frame("flat", frame_id=f"f-gray-{i}") for i in range(3)]),
        policy=FramePolicy(retry_delay_ms=0),
        sleep=lambda _s: None,
    )
    answer = SceneService(ocr=FakeOcr(["价格 8.80 元"]), vlm=FakeVlm(), provider=provider).answer("这个多少钱")

    assert "low_quality" in answer.degraded
    assert "8元8角" in answer.text
    assert answer.confidence <= 0.55


def test_direct_blurry_frame_is_degraded_but_bright_is_not():
    blurry = _service(["价格 8.80 元"]).answer("这个多少钱", frame=make_frame("flat", frame_id="f-flat"))
    assert "low_quality" in blurry.degraded

    bright = _service(["价格 8.80 元"]).answer("这个多少钱", frame=make_frame("white_noise", frame_id="f-white"))
    assert "low_quality" not in bright.degraded
