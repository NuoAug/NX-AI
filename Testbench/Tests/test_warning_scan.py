from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import Mock, patch

import numpy as np

from scene_guard import (SceneService, ProjectSceneService, _red_region_boxes,
                         _merge_row_boxes, _merge_column_boxes)
from aieyes.interfaces import OcrLine, OcrResult
from aieyes.scene.models import SceneAnswer
from aieyes.scene.service import _AnswerContext


def make_ctx(ocr=None, width=160, height=80):
    frame = SimpleNamespace(rgb=np.zeros((height, width, 3), np.uint8),
                            width=width, height=height, frame_id='f')
    quality = Mock(labels=())
    return _AnswerContext(frame=frame, quality=quality, ocr=ocr,
                          degraded=[], notes=[], timings={}, started=0)


class WarningScanTests(TestCase):
    def test_red_region_boxes_finds_text_sized_red_block(self):
        img = np.zeros((200, 400, 3), np.uint8)
        img[60:80, 100:140] = (255, 0, 0)
        boxes = _red_region_boxes(img)
        self.assertTrue(boxes)
        x1, y1, x2, y2 = boxes[0]
        self.assertLessEqual(abs(x1 - 100), 6)
        self.assertLessEqual(abs(y1 - 60), 6)
        self.assertLessEqual(abs(x2 - 140), 6)
        self.assertLessEqual(abs(y2 - 80), 6)

    def test_red_region_boxes_ignores_non_red_noise_and_large_blobs(self):
        img = np.zeros((200, 400, 3), np.uint8)
        img[60:80, 100:140] = (0, 0, 255)      # 蓝色文字块
        img[90:190, 200:390] = (255, 0, 0)     # 过大面积（整帧 9%）
        img[0, 0] = (255, 0, 0)               # 单像素噪声
        self.assertEqual(_red_region_boxes(img), [])

    def test_red_region_boxes_ignores_tall_or_wide_non_text_shapes(self):
        img = np.zeros((400, 200, 3), np.uint8)
        img[50:150, 80:100] = (255, 0, 0)    # 20×100 细长立柱
        img[300:310, 20:220] = (255, 0, 0)   # 200×10 细长横条
        self.assertEqual(_red_region_boxes(img), [])

    def test_merge_column_boxes_merges_vertical_neighbors_only(self):
        a, b = (100, 100, 140, 140), (102, 170, 138, 210)  # 同列垂直相邻
        far = (300, 100, 340, 140)                          # 同行但横向远离
        merged = _merge_column_boxes([a, b, far])
        self.assertIn((100, 100, 140, 210), merged)
        self.assertIn(far, merged)
        self.assertEqual(len(merged), 2)

    def test_merge_row_boxes_merges_same_line_chars_only(self):
        a, b = (100, 100, 140, 140), (160, 100, 200, 140)
        below = (100, 180, 140, 220)
        merged = _merge_row_boxes([a, b, below])
        self.assertIn((100, 100, 200, 140), merged)
        self.assertIn(below, merged)
        self.assertEqual(len(merged), 2)

    def test_column_merge_stops_after_three_rows(self):
        boxes = [(100, 100, 140, 140), (100, 170, 140, 210),
                 (100, 240, 140, 280), (100, 310, 140, 350)]
        merged = _merge_column_boxes(boxes)
        self.assertGreater(len(merged), 1)
        self.assertLess(len(merged), 4)

    def test_two_line_two_column_warning_block_is_merged_before_ocr(self):
        boxes = [(218, 506, 256, 544), (282, 506, 323, 544),
                 (224, 593, 262, 631), (285, 589, 314, 625)]
        merged = _merge_column_boxes(_merge_row_boxes(boxes))
        self.assertEqual(len(merged), 1)
        x1, y1, x2, y2 = merged[0]
        self.assertLessEqual(abs(x1 - 218), 2)
        self.assertLessEqual(abs(y1 - 506), 2)
        self.assertGreaterEqual(x2, 323)
        self.assertGreaterEqual(y2, 631)

    def test_warning_lines_keeps_high_score_warning_text_and_merges_vertical(self):
        scene = SceneService(vlm=Mock(), ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=(
            OcrLine('高压', (10, 10, 60, 40), .9),
            OcrLine('危险', (12, 90, 58, 40), .88),
            OcrLine('旷', (150, 10, 180, 40), .85),   # 高分但非警示词
            OcrLine('KK', (10, 150, 60, 180), .9),    # 非中文
            OcrLine('禁止入内', (10, 240, 130, 280), .6),  # 低分
        ))
        frame = make_ctx(width=400, height=300).frame
        with patch('scene_guard._red_region_boxes', return_value=[(100, 100, 140, 120)]):
            lines = scene._warning_lines(frame)
        texts = [line.text for line in lines]
        self.assertEqual(texts, ['高压危险'])

    def test_warning_lines_retries_small_block_at_2x_to_recover_missing_line(self):
        scene = SceneService(vlm=Mock(), ocr=Mock())
        scene.ocr.recognize.side_effect = [
            OcrResult(lines=(
                OcrLine('高丽', (10, 10, 60, 30), .713),
                OcrLine('危险', (12, 60, 58, 30), .949),
            )),
            OcrResult(lines=(
                OcrLine('高压', (12, 12, 60, 30), .855),
                OcrLine('危险', (14, 64, 58, 30), .954),
            )),
        ]
        frame = make_ctx(width=400, height=300).frame
        with patch('scene_guard._red_region_boxes', return_value=[(100, 100, 140, 120)]):
            lines = scene._warning_lines(frame)
        self.assertEqual([line.text for line in lines], ['高压危险'])
        self.assertEqual(scene.ocr.recognize.call_count, 2)

    def test_warning_lines_dedupes_identical_signs(self):
        scene = SceneService(vlm=Mock(), ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=(
            OcrLine('高压', (10, 10, 60, 40), .9),
            OcrLine('危险', (12, 90, 58, 40), .88),
        ))
        frame = make_ctx(width=400, height=300).frame
        with patch('scene_guard._red_region_boxes',
                   return_value=[(100, 100, 140, 120), (300, 100, 340, 120)]):
            lines = scene._warning_lines(frame)
        self.assertEqual([line.text for line in lines], ['高压危险'])

    def test_warning_lines_survives_ocr_engine_failure(self):
        scene = SceneService(vlm=Mock(), ocr=Mock())
        scene.ocr.recognize.side_effect = RuntimeError('engine down')
        frame = make_ctx(width=400, height=300).frame
        with patch('scene_guard._red_region_boxes', return_value=[(100, 100, 140, 120)]):
            self.assertEqual(scene._warning_lines(frame), [])


class ReadWarningRouteTests(TestCase):
    def test_warning_read_question_gets_local_ocr_evidence(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=()))
        warning = [OcrLine('高压危险', (1900, 484, 100, 130), .85)]
        with patch.object(SceneService, '_warning_lines', return_value=warning) as scan:
            answer = scene._answer_read('读一下前方红色警示字', ctx)
        scan.assert_called_once()
        self.assertIn('高压危险', answer.text)
        self.assertTrue(any(tag.kind == 'ocr' and tag.raw_text == '高压危险'
                            for tag in answer.source_tags))

    def test_plain_read_question_skips_warning_scan(self):
        scene = SceneService(vlm=Mock())
        scene.vlm.complete.return_value = '看不清。'
        ctx = make_ctx(ocr=OcrResult(lines=()))
        with patch.object(SceneService, '_warning_lines', return_value=[]) as scan:
            answer = scene._answer_read('读一下牌子上的字', ctx)
        scan.assert_not_called()
        self.assertEqual(answer.text, '画面里我没读到清楚的字。')

    def test_corner_read_question_skips_warning_scan(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=()))
        with patch.object(SceneService, '_warning_lines', return_value=[]) as scan:
            answer = scene._answer_read('读左上角的字', ctx)
        scan.assert_not_called()
        self.assertIn('左上角我没读到清楚的字', answer.text)

    def test_warning_read_dedupes_subsumed_warning_text(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=()))
        warning = [OcrLine('危险', (1900, 620, 90, 45), .92),
                   OcrLine('高压危险', (1933, 440, 125, 240), .96)]
        with patch.object(SceneService, '_warning_lines', return_value=warning):
            answer = scene._answer_read('读一下前方红色警示字', ctx)
        self.assertIn('高压危险', answer.text)
        self.assertNotIn('高压危险，危险', answer.text)
        self.assertEqual([t.text for t in answer.ocr_lines if '危险' in t.text],
                         ['高压危险'])

    def test_existing_warning_text_blocks_extra_scan(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=(OcrLine('禁止入内', (10, 10, 80, 30), .9),)))
        with patch.object(SceneService, '_warning_lines', return_value=[]) as scan:
            answer = scene._answer_read('读一下前方红色警示字', ctx)
        scan.assert_called_once()
        self.assertIn('禁止入内', answer.text)


    def test_partial_warning_does_not_suppress_local_evidence(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=(OcrLine('危险', (10, 50, 60, 30), .9),)))
        with patch.object(SceneService, '_warning_lines', return_value=[OcrLine('高压危险', (10, 10, 60, 70), .95)]) as scan:
            answer = scene._answer_read('读一下红色警示字', ctx)
        scan.assert_called_once()
        self.assertIn('高压危险', answer.text)
        self.assertEqual([line.text for line in answer.ocr_lines], ['高压危险'])


class SafetyDescribeTests(TestCase):
    def base_answer(self):
        return SceneAnswer('我看到的是行人、树木。', 'describe',
                           frame_id='f', vlm_text='行人、树木', confidence=.65)

    def test_safety_question_appends_red_warning_scan(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=()))
        warning = [OcrLine('高压危险', (1900, 484, 100, 130), .85)]
        with patch.object(ProjectSceneService, '_answer_describe', return_value=self.base_answer()), \
                patch.object(SceneService, '_warning_lines', return_value=warning) as scan:
            answer = scene._answer_describe('前面安全吗？', ctx)
        scan.assert_called_once()
        self.assertIn('我看到的是行人、树木。', answer.text)
        self.assertIn('画面中有红色警示文字：高压危险。', answer.text)
        self.assertTrue(any(tag.kind == 'ocr' and tag.field == 'warning'
                            for tag in answer.source_tags))
        self.assertTrue(any('高压危险' == line.text for line in answer.ocr_lines))

    def test_safety_question_uses_full_frame_ocr_warning_when_present(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=(OcrLine('禁止入内', (10, 10, 80, 30), .9),)))
        with patch.object(ProjectSceneService, '_answer_describe', return_value=self.base_answer()), \
                patch.object(SceneService, '_warning_lines', return_value=[]) as scan:
            answer = scene._answer_describe('前面安全吗？', ctx)
        scan.assert_not_called()
        self.assertIn('禁止入内', answer.text)
        self.assertNotIn('红色警示文字', answer.text)

    def test_safety_question_accepts_punctuated_full_frame_warning(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=(
            OcrLine('禁止摩托车、电动车、自行车', (10, 10, 220, 30), .95),)))
        with patch.object(ProjectSceneService, '_answer_describe', return_value=self.base_answer()), \
                patch.object(SceneService, '_warning_lines', return_value=[]) as scan:
            answer = scene._answer_describe('前面安全吗？', ctx)
        scan.assert_not_called()
        self.assertIn('禁止摩托车、电动车、自行车', answer.text)

    def test_safety_warning_dedupes_subsumed_and_repeated_lines(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=()))
        warning = [OcrLine('危险', (1900, 620, 90, 45), .92),
                   OcrLine('高压危险', (1933, 440, 125, 240), .96),
                   OcrLine('高压危险', (330, 440, 125, 240), .9)]
        with patch.object(ProjectSceneService, '_answer_describe', return_value=self.base_answer()), \
                patch.object(SceneService, '_warning_lines', return_value=warning):
            answer = scene._answer_describe('前面安全吗？', ctx)
        self.assertIn('画面中有红色警示文字：高压危险。', answer.text)
        self.assertNotIn('危险、高压', answer.text)
        self.assertEqual([t.text for t in answer.ocr_lines], ['高压危险'])

    def test_safety_question_without_warnings_keeps_base_answer(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=()))
        with patch.object(ProjectSceneService, '_answer_describe', return_value=self.base_answer()), \
                patch.object(SceneService, '_warning_lines', return_value=[]):
            answer = scene._answer_describe('前面安全吗？', ctx)
        self.assertEqual(answer.text, '我看到的是行人、树木。')

    def test_plain_describe_question_not_affected(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=()))
        with patch.object(ProjectSceneService, '_answer_describe', return_value=self.base_answer()), \
                patch.object(SceneService, '_warning_lines', return_value=[]) as scan:
            answer = scene._answer_describe('画面里有什么？', ctx)
        scan.assert_not_called()
        self.assertEqual(answer.text, '我看到的是行人、树木。')

    def test_safety_describe_runs_ocr_and_plain_describe_skips_it(self):
        scene = SceneService(vlm=Mock())
        seen = {}

        def fake_answer(question, **kwargs):
            seen[question] = getattr(scene._request_state, 'skip_ocr', None)
            return SceneAnswer('好。', 'describe', frame_id='f')

        with patch.object(ProjectSceneService, 'answer', side_effect=fake_answer):
            scene.answer('前面安全吗？')
            scene.answer('画面里有什么？')
        self.assertFalse(seen['前面安全吗？'])
        self.assertTrue(seen['画面里有什么？'])
