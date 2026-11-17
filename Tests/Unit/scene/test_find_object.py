"""找物 find_object 单元测试：候选 ≤2、定位/去重、非法输出与降级。"""

from __future__ import annotations

import json

from aieyes.scene import SceneConfig, SceneService
from aieyes.scene.service import _find_target
from scene_fakes import FakeOcr, FakeVlm, make_frame

TARGET = "钥匙"


def _service(reply: str = "", *, error: Exception | None = None, config: SceneConfig | None = None):
    return SceneService(
        ocr=FakeOcr([]),
        vlm=FakeVlm(reply, error=error),
        config=config or SceneConfig(),
    )


def _reply(items) -> str:
    return "```json\n" + json.dumps(items, ensure_ascii=False) + "\n```"


def test_location_question_extracts_the_object_not_the_location_phrase():
    assert _find_target("键盘现在在画面中的哪个地方") == "键盘"
    assert _find_target("键盘在画面哪里") == "键盘"
    assert _find_target("我的钥匙在屏幕上的哪个位置") == "钥匙"


def test_three_candidates_are_limited_to_two_with_positions():
    items = [
        {"label": "钥匙", "position": "左上方", "box": [100, 100, 300, 300], "feature": "金属色", "confidence": 0.9},
        {"label": "钥匙", "position": "右上方", "box": [320, 90, 430, 220], "feature": "带挂绳", "confidence": 0.7},
        {"label": "钥匙", "position": "右侧", "box": [600, 300, 750, 450], "feature": "", "confidence": 0.6},
    ]
    frame = make_frame(frame_id="f-find")
    result = _service(_reply(items)).find_object(frame, TARGET)

    assert result.found
    assert len(result.candidates) == 2
    assert "有两个可能" in result.text
    assert "金属色" in result.text
    for candidate in result.candidates:
        x, y, w, h = candidate.box
        assert 0 <= x < frame.width and 0 <= y < frame.height
        assert w > 0 and h > 0
        assert candidate.frame_id == "f-find"
    assert {candidate.position for candidate in result.candidates} <= {"左上方", "正上方", "右上方", "左侧", "正中间", "右侧", "左下方", "正下方", "右下方"}


def test_overlapping_candidates_are_deduped():
    items = [
        {"label": "钥匙", "box": [100, 100, 400, 400], "confidence": 0.9},
        {"label": "钥匙", "box": [110, 105, 405, 402], "confidence": 0.8},  # 与第一个几乎重合
        {"label": "钥匙", "box": [600, 300, 750, 450], "confidence": 0.6},
    ]
    result = _service(_reply(items)).find_object(make_frame(), TARGET)

    assert len(result.candidates) == 2


def test_missing_box_gets_approximate_center_and_lower_confidence():
    items = [{"label": "钥匙", "feature": "金属色", "confidence": 0.8}]
    frame = make_frame(width=320, height=240)
    result = _service(_reply(items)).find_object(frame, TARGET)

    candidate = result.candidates[0]
    assert candidate.box == (80, 60, 160, 120)
    assert "近似" in candidate.feature
    assert candidate.confidence <= 0.4


def test_invalid_json_does_not_invent_position():
    result = _service("我不知道钥匙在哪里。").find_object(make_frame(), TARGET)

    assert not result.found
    assert result.candidates == ()
    assert "vlm_parse" in result.degraded
    assert result.text == "我没看清钥匙在哪儿。"


def test_empty_array_means_not_seen():
    result = _service("[]").find_object(make_frame(), TARGET)

    assert not result.found
    assert result.text == "画面里没看到钥匙。"
    assert "vlm_parse" not in result.degraded


def test_vlm_failure_degrades():
    result = _service(error=RuntimeError("vlm down")).find_object(make_frame(), TARGET)

    assert not result.found
    assert "vlm_failed" in result.degraded
    assert "失效" in result.text or "没看清" in result.text


def test_max_candidates_config_is_respected():
    items = [
        {"label": "钥匙", "box": [10, 10, 200, 200], "confidence": 0.9},
        {"label": "钥匙", "box": [400, 10, 600, 200], "confidence": 0.8},
    ]
    result = _service(_reply(items), config=SceneConfig(max_candidates=1)).find_object(make_frame(), TARGET)

    assert len(result.candidates) == 1


def test_answer_find_flow_reports_target_and_candidates():
    items = [{"label": "钥匙", "box": [400, 100, 550, 250], "feature": "金属色", "confidence": 0.8}]
    frame = make_frame(frame_id="f-answer-find")
    answer = _service(_reply(items)).answer("我的钥匙在哪", frame=frame)

    assert answer.intent == "find"
    assert answer.frame_id == "f-answer-find"
    assert answer.candidates is not None and answer.candidates.found
    assert "钥匙" in answer.text
    assert answer.source_tags and answer.source_tags[0].kind == "vlm"
