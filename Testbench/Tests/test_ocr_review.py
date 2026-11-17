from types import SimpleNamespace
from unittest.mock import Mock, patch
import numpy as np
from scene_guard import SceneService, ProjectSceneService
from aieyes.interfaces import OcrLine, OcrResult
from aieyes.scene.models import SceneAnswer, SourceTag

OLD = '原始学校招牌文字'
NEW = '原始学校标牌文字'


def inputs():
    frame = SimpleNamespace(rgb=np.zeros((80, 160, 3), np.uint8), frame_id='frame')
    line = OcrLine(OLD, (4, 5, 120, 35), .7)
    return frame, line, OcrResult(lines=(line,))


def test_two_views_agree_without_rewriting_raw_or_leaking_the_answer():
    frame, line, raw = inputs()
    vlm = Mock()
    vlm.complete.side_effect = [NEW, NEW]
    assert SceneService(vlm=vlm)._review_read_lines(frame, raw) == [(line, NEW)]
    assert raw.lines[0].text == OLD
    for call in vlm.complete.call_args_list:
        assert OLD not in call.args[0] and NEW not in call.args[0]


def test_second_view_returning_raw_is_disagreement():
    frame, line, raw = inputs()
    vlm = Mock()
    vlm.complete.side_effect = [NEW, OLD]
    assert SceneService(vlm=vlm)._review_read_lines(frame, raw) == [(line, None)]


def test_numbers_and_clear_text_are_not_rewritten():
    frame, _, _ = inputs()
    vlm = Mock()
    raw = OcrResult(lines=(
        OcrLine('生产日期2026年1月31日', (0, 0, 60, 20), .6),
        OcrLine('价格12.50元', (0, 0, 60, 20), .6),
        OcrLine(OLD, (0, 0, 60, 20), .99),
    ))
    assert SceneService(vlm=vlm)._review_read_lines(frame, raw) == []
    vlm.complete.assert_not_called()


def test_review_has_a_bounded_call_budget():
    frame, line, _ = inputs()
    vlm = Mock()
    vlm.complete.side_effect = [NEW] * 4
    result = SceneService(vlm=vlm)._review_read_lines(frame, OcrResult(lines=(line, line, line)))
    assert result == [(line, NEW), (line, NEW), (line, None)]
    assert vlm.complete.call_count == 4


def test_read_reply_preserves_raw_evidence_and_adds_vlm_provenance():
    frame, _, raw = inputs()
    vlm = Mock()
    vlm.complete.side_effect = [NEW, NEW]
    scene = SceneService(vlm=vlm)

    def original_answer(*args, **kwargs):
        scene._run_ocr(frame)
        return SceneAnswer('画面上写着：' + OLD, 'read', frame_id='frame', ocr_lines=raw.lines,
                           source_tags=(SourceTag(kind='ocr', frame_id='frame', raw_text=OLD, field='text'),),
                           confidence=.8)

    with patch.object(ProjectSceneService, 'answer', side_effect=original_answer), \
            patch.object(ProjectSceneService, '_run_ocr', return_value=raw):
        result = scene.answer('读一下牌子上的字', frame=frame)
    assert NEW in result.text and OLD not in result.text
    assert result.ocr_lines[0].text == OLD
    assert any(x.kind == 'ocr' and x.raw_text == OLD for x in result.source_tags)
    assert any(x.kind == 'vlm' and x.raw_text == NEW for x in result.source_tags)
    assert result.confidence == .8
