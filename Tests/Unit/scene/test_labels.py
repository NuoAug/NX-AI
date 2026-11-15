"""来源标注单元测试：口述/包装/画面/历史/计算。"""

from __future__ import annotations

import pytest

from aieyes.scene import SOURCE_LABELS, SceneService, SourceTag
from scene_fakes import FakeOcr, FakeVlm, make_frame


@pytest.mark.parametrize(
    ("kind", "label"),
    [("spoken", "口述"), ("ocr", "包装"), ("vlm", "画面"), ("history", "历史"), ("computed", "计算")],
)
def test_source_label_names(kind, label):
    assert SourceTag(kind=kind).label == label


def test_unknown_source_kind_falls_back_to_kind():
    assert SourceTag(kind="sensor").label == "sensor"
    assert set(SOURCE_LABELS) == {"spoken", "ocr", "vlm", "history", "computed"}


def test_price_tag_points_back_to_ocr_line():
    answer = SceneService(ocr=FakeOcr(["商品标题", "价格 12.50 元"]), vlm=FakeVlm("")).answer(
        "这个多少钱", frame=make_frame(frame_id="f-label")
    )

    tag = answer.source_tags[0]
    assert tag.kind == "ocr" and tag.label == "包装"
    assert tag.frame_id == "f-label"
    assert tag.line_index == 1
    assert answer.ocr_lines[tag.line_index].text == tag.raw_text
    assert answer.evidence


def test_spoken_purchase_tags_are_marked():
    answer = SceneService(ocr=FakeOcr([]), vlm=FakeVlm("")).answer("每件12块5，买3件一共多少钱")

    kinds = {tag.kind for tag in answer.source_tags}
    assert kinds == {"spoken", "computed"}
    assert {tag.label for tag in answer.source_tags} == {"口述", "计算"}
