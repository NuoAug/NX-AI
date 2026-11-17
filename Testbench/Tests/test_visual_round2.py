"""2026-11-16 第二轮视觉修复的单元测试：读字重路由、读字噪声过滤、复合计数、
危险类找物证据路线、店面招牌 OCR 补充与 VLM 截断句过滤。"""
from types import SimpleNamespace
from dataclasses import replace as _dc_replace
from unittest.mock import Mock, patch
import numpy as np
import pytest

from scene_guard import SceneService, ProjectSceneService
from aieyes.interfaces import OcrLine, OcrResult
from aieyes.scene.models import SceneAnswer
from aieyes.scene.service import _AnswerContext
from aieyes.scene.prompts import describe_prompt


def make_ctx(ocr=None, width=160, height=80):
    frame = SimpleNamespace(rgb=np.zeros((height, width, 3), np.uint8),
                            width=width, height=height, frame_id='f')
    if ocr is not None and tuple(ocr.image_size) == (0, 0):
        # 角落过滤按 OcrResult.image_size 划分区域，测试须与帧尺寸一致
        ocr = _dc_replace(ocr, image_size=(width, height))
    quality = Mock(labels=())
    return _AnswerContext(frame=frame, quality=quality, ocr=ocr,
                          degraded=[], notes=[], timings={}, started=0)


class TestReroute:
    def test_warning_question_uses_text_evidence_not_scene_description(self):
        scene = SceneService()
        with patch.object(ProjectSceneService, 'answer',
                          return_value=SceneAnswer('警示文字', 'read')) as parent:
            scene.answer('前方门口上有什么危险警示？')
        assert parent.call_args.kwargs.get('intent_hint') == 'read'

    def test_notice_written_question_reroutes_to_read(self):
        scene = SceneService()
        with patch.object(ProjectSceneService, 'answer',
                          return_value=SceneAnswer('读到文字', 'read')) as parent:
            scene.answer('柱子上贴的告示写了什么？')
        assert parent.call_args.kwargs.get('intent_hint') == 'read'

    def test_phone_board_question_reroutes_to_read(self):
        scene = SceneService()
        with patch.object(ProjectSceneService, 'answer',
                          return_value=SceneAnswer('读到文字', 'read')) as parent:
            scene.answer('这块应急电话牌上有哪些电话？')
        assert parent.call_args.kwargs.get('intent_hint') == 'read'

    def test_plain_describe_and_safety_not_rerouted(self):
        scene = SceneService()
        for question in ('描述一下这条街的店面和环境。', '前面安全吗？', '画面里有什么？'):
            with patch.object(ProjectSceneService, 'answer',
                             return_value=SceneAnswer('答', 'describe')) as parent:
                scene.answer(question)
            assert parent.call_args.kwargs.get('intent_hint') != 'read'

    def test_signboard_describe_does_not_skip_ocr(self):
        scene = SceneService()
        seen = {}

        def fake_answer(question, **kwargs):
            seen[question] = getattr(scene._request_state, 'skip_ocr', None)
            return SceneAnswer('好。', 'describe', frame_id='f')

        with patch.object(ProjectSceneService, 'answer', side_effect=fake_answer):
            scene.answer('描述一下这条街的店面和环境。')
            scene.answer('画面里有什么？')
        assert seen['描述一下这条街的店面和环境。'] is False
        assert seen['画面里有什么？'] is True


class TestReadNoiseFilter:
    def test_low_score_and_single_char_noise_dropped(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=(
            OcrLine('FKEN', (1017, 5, 1265, 263), .51),
            OcrLine('十', (1925, 713, 58, 54), .72),
            OcrLine('高压危险', (1259, 598, 865, 191), .93),
            OcrLine('软件学院高配室', (2135, 838, 146, 39), .75),
        )))
        answer = scene._answer_read('这扇门上写了什么？旁边的牌子是什么？', ctx)
        assert '高压危险' in answer.text and '软件学院高配室' in answer.text
        assert 'FKEN' not in answer.text and '十' not in answer.text

    def test_clear_single_char_digit_is_kept(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=(
            OcrLine('1', (10, 10, 30, 30), .9),)))
        answer = scene._answer_read('门牌号是多少？', ctx)
        assert '1' in answer.text


class TestCompoundCount:
    def test_two_categories_counted_separately(self):
        vlm = Mock()
        vlm.complete.return_value = '{"井盖":2,"排水篦子":1}'
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        out = scene._vlm_complete(
            describe_prompt('斑马线附近地面上有几个井盖、几个排水篦子？'), frame)
        assert out == '井盖有2个，排水篦子有1个。'

    def test_unreliable_category_reports_uncertainty(self):
        vlm = Mock()
        vlm.complete.return_value = '{"井盖":2,"排水篦子":null}'
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        out = scene._vlm_complete(
            describe_prompt('斑马线附近地面上有几个井盖、几个排水篦子？'), frame)
        assert '井盖有2个' in out and '排水篦子我没数清' in out

    def test_all_categories_uncertain_reads_naturally(self):
        vlm = Mock()
        vlm.complete.return_value = '{"井盖":null,"排水篦子":null}'
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        out = scene._vlm_complete(
            describe_prompt('斑马线附近地面上有几个井盖、几个排水篦子？'), frame)
        assert out == '没数清井盖、排水篦子的数量。'

    def test_invalid_compound_payload_falls_back_honestly(self):
        vlm = Mock()
        vlm.complete.return_value = '我看不太清'
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        out = scene._vlm_complete(
            describe_prompt('斑马线附近地面上有几个井盖、几个排水篦子？'), frame)
        assert out == '暂时无法确认数量。'

    def test_single_count_keeps_old_format_and_gains_units(self):
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        for q, raw, expected in [
                ('这个消防点有几具灭火器？', '{"count":2}', '有2具。'),
                ('画面里有几辆白色的车？', '{"count":2}', '有2辆。'),
                ('画面里有几辆白色的车？', '{"count":null}', '暂时无法确认数量。')]:
            vlm = Mock(); vlm.complete.return_value = raw
            scene = SceneService(vlm=vlm)
            assert scene._vlm_complete(describe_prompt(q), frame) == expected


class TestDangerFind:
    def test_danger_find_uses_red_warning_evidence(self):
        scene = SceneService(vlm=Mock(), ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=())
        frame = SimpleNamespace(rgb=np.zeros((80, 160, 3), np.uint8),
                                width=160, height=80, frame_id='f')
        warning = [OcrLine('高压危险', (60, 10, 30, 50), .96)]
        with patch.object(SceneService, '_warning_lines', return_value=warning):
            result = scene.find_object(frame, '有危险')
        assert result.found and '红色警示文字：高压危险' in result.text

    def test_danger_find_uses_full_frame_warning_line_first(self):
        scene = SceneService(vlm=Mock(), ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=(
            OcrLine('危险！禁止攀爬', (20, 10, 90, 20), .91),))
        frame = SimpleNamespace(rgb=np.zeros((80, 160, 3), np.uint8),
                                width=160, height=80, frame_id='f')
        with patch.object(SceneService, '_warning_lines', return_value=[]) as scan:
            result = scene.find_object(frame, '有危险')
        scan.assert_called_once()
        assert result.found and '危险！禁止攀爬' in result.text and '警示文字' in result.text

    def test_fire_equipment_sign_is_not_danger(self):
        # 消防设施操作说明（如"灭火器使用方法"）不是危险警示，问"哪里有危险"不得据此报有危险
        scene = SceneService(vlm=Mock(), ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=(
            OcrLine('灭火器使用方法', (20, 10, 60, 20), .85),))
        frame = SimpleNamespace(rgb=np.zeros((80, 160, 3), np.uint8),
                                width=160, height=80, frame_id='f')
        with patch.object(SceneService, '_warning_lines', return_value=[]):
            result = scene.find_object(frame, '有危险')
        assert not result.found
        assert '没有在画面里读到明确的危险警示文字' in result.text
        assert '安全' not in result.text

    def test_danger_find_without_evidence_stays_honest(self):
        scene = SceneService(vlm=Mock(), ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=())
        frame = SimpleNamespace(rgb=np.zeros((80, 160, 3), np.uint8),
                                width=160, height=80, frame_id='f')
        with patch.object(SceneService, '_warning_lines', return_value=[]):
            result = scene.find_object(frame, '有危险')
        assert not result.found
        assert '没有在画面里读到明确的危险警示文字' in result.text
        assert '安全' not in result.text

    def test_normal_target_still_uses_vlm_find(self):
        from aieyes.scene.models import ObjectCandidates
        scene = SceneService(vlm=Mock())
        frame = SimpleNamespace(rgb=np.zeros((80, 160, 3), np.uint8),
                                width=160, height=80, frame_id='f')
        empty = ObjectCandidates('灭火器', False, '我没看清灭火器在哪儿。',
                                 frame_id='f')
        with patch.object(ProjectSceneService, 'find_object', return_value=empty) as parent:
            result = scene.find_object(frame, '灭火器')
        parent.assert_called_once()
        assert result is empty


class TestPhoneBoardRescan:
    """电话/号码牌读字：聚簇复扫 + 单位配对 + VLM 双尺度复核。"""

    FULL_BOARD = (
        OcrLine('交园应急电话指南', (1940, 583, 190, 50), .8682),
        OcrLine('应急电话', (2022, 635, 65, 36), .8017),
        OcrLine('6668110', (1983, 660, 156, 49), .8809),
        OcrLine('十', (1925, 713, 58, 54), .7236),
        OcrLine('6276711', (1983, 729, 151, 48), .7959),
        OcrLine('7797110', (1911, 776, 240, 95), .8517),
        OcrLine('学村派出', (2030, 788, 58, 19), .6277),
        OcrLine('EXIT', (3830, 318, 59, 42), .5458),
    )
    # 聚簇裁剪框为 (1839,497)-(2230,957)，以下为裁剪坐标系内的复扫结果
    RESCAN_BOARD = (
        OcrLine('校园应急电话指南', (82, 96, 209, 35), .9653),
        OcrLine('校园应急电话', (148, 141, 99, 24), .9393),
        OcrLine('6668110', (146, 170, 149, 38), .9671),
        OcrLine('医务室', (150, 213, 52, 22), .9784),
        OcrLine('6276711', (147, 241, 147, 38), .9581),
        OcrLine('集美字村派出所', (149, 287, 114, 24), .9480),
        OcrLine('7797110', (148, 318, 147, 35), .9579),
    )

    def _phone_ctx(self):
        # 复扫按整帧坐标聚簇裁剪，测试图幅须覆盖真实样例坐标范围
        return make_ctx(ocr=OcrResult(lines=self.FULL_BOARD),
                        width=4096, height=1000)

    def test_rescan_fixes_title_reads_labels_and_pairs_units(self):
        vlm = Mock()
        vlm.complete.return_value = '集美学村派出所\n7797110'
        scene = SceneService(vlm=vlm, ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=self.RESCAN_BOARD)
        ctx = self._phone_ctx()
        answer = scene._answer_read('这块应急电话牌上有哪些电话？', ctx)
        assert '校园应急电话指南' in answer.text
        assert '校园应急电话 6668110' in answer.text
        assert '医务室 6276711' in answer.text
        assert '集美学村派出所 7797110' in answer.text
        assert '交园' not in answer.text and '字村' not in answer.text
        # '学村派出' 是正确读法'集美学村派出所'的子串，不能整串 not in；
        # 改查带标点的截断残行形态，并确认完整归属读法已替代它
        assert '学村派出。' not in answer.text and '学村派出，' not in answer.text
        assert '有一行小字' not in answer.text
        assert 'EXIT' not in answer.text and '十' not in answer.text

    def test_pairing_vlm_disagreement_keeps_ocr_label(self):
        # 1x 与 OCR 一致时短路（pair1 只耗 1 次调用）；分歧/一字修正的对仍走双尺度
        vlm = Mock()
        vlm.complete.side_effect = [
            '校园应急电话 6668110',
            '医师室 6276711', '医务室 6276711',
            '集美学校派出所\n7797110', '集美学村派出所\n7797110']
        scene = SceneService(vlm=vlm, ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=self.RESCAN_BOARD)
        ctx = self._phone_ctx()
        answer = scene._answer_read('这块应急电话牌上有哪些电话？', ctx)
        assert '校园应急电话 6668110' in answer.text and '校园应急 ' not in answer.text
        assert '医务室 6276711' in answer.text and '医师室' not in answer.text
        assert '集美学村派出所 7797110' in answer.text

    def test_pairing_first_scale_agreement_skips_second_scale(self):
        # 1x 读法与 OCR 完全一致的两对短路省 2 次调用，共 4 次；第三对仍双尺度修正
        vlm = Mock()
        vlm.complete.side_effect = [
            '校园应急电话 6668110', '医务室\n6276711',
            '集美学校派出所\n7797110', '集美学村派出所\n7797110']
        scene = SceneService(vlm=vlm, ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=self.RESCAN_BOARD)
        ctx = self._phone_ctx()
        answer = scene._answer_read('这块应急电话牌上有哪些电话？', ctx)
        assert vlm.complete.call_count == 4
        assert '校园应急电话 6668110' in answer.text
        assert '医务室 6276711' in answer.text
        assert '集美学村派出所 7797110' in answer.text

    def test_pairing_adopts_one_char_fix_when_scales_differ(self):
        # 1x 小图固有形近字误读（学→校）、2x 正确：数字一致且中文仅差一字时采纳 2x 读法
        vlm = Mock()
        vlm.complete.side_effect = [
            '集美学村派出所\n7797110', '集美学村派出所\n7797110',
            '集美学村派出所\n7797110', '集美学村派出所\n7797110',
            '集美学校派出所\n7797110', '集美学村派出所\n7797110']
        scene = SceneService(vlm=vlm, ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=self.RESCAN_BOARD)
        ctx = self._phone_ctx()
        answer = scene._answer_read('这块应急电话牌上有哪些电话？', ctx)
        assert '集美学村派出所 7797110' in answer.text
        assert '学校' not in answer.text and '字村' not in answer.text

    def test_stale_review_substring_not_mangling_merged_line(self):
        # 复核基于复扫前的原始行；"学村派出"已是"集美学村派出所"的子串时，
        # 不能按独立行替换成"还没看清"，也不得引入 ocr_uncertain
        from aieyes.scene.service import SceneService as _Base

        low = OcrLine('学村派出', (2030, 788, 58, 19), .6277)

        def fake_answer(self, question, **kwargs):
            self._request_state.ocr_reviews = [(low, None)]
            return SceneAnswer('我看到上面写着：集美学村派出所 7797110。', 'read',
                               frame_id='f', confidence=.9)

        scene = SceneService(vlm=Mock(), ocr=Mock())
        with patch.object(_Base, 'answer', fake_answer):
            ans = scene.answer('这块应急电话牌上有哪些电话？', frame=Mock())
        assert '集美学村派出所 7797110' in ans.text
        assert '还没看清' not in ans.text and 'ocr_uncertain' not in ans.degraded
        assert ans.confidence == .9

    def test_rescan_rejects_low_score_new_lines(self):
        vlm = Mock()
        scene = SceneService(vlm=vlm, ocr=Mock())
        scene.ocr.recognize.return_value = OcrResult(lines=(
            OcrLine('6668110', (47, 15, 149, 38), .9671),
            OcrLine('区发', (200, 100, 40, 20), .71)))
        frame = SimpleNamespace(rgb=np.zeros((1000, 4096, 3), np.uint8),
                                width=4096, height=1000, frame_id='f')
        ctx = make_ctx(ocr=OcrResult(lines=(
            OcrLine('6668110', (1983, 660, 156, 49), .8809),)),
            width=4096, height=1000)
        answer = scene._answer_read('牌上有哪些电话？', ctx)
        assert '6668110' in answer.text and '区发' not in answer.text
        vlm.complete.assert_not_called()

    def test_rescan_not_triggered_for_non_phone_questions(self):
        scene = SceneService(vlm=Mock(), ocr=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=(
            OcrLine('危险！禁止攀爬', (2187, 554, 124, 40), .91),)))
        answer = scene._answer_read('柱子上贴的告示写了什么？', ctx)
        scene.ocr.recognize.assert_not_called()
        assert '危险！禁止攀爬' in answer.text

    def test_rescan_skipped_without_digit_lines(self):
        # 门牌类问题若整帧没有数字行，复扫无簇可扫，直接走原流程
        scene = SceneService(vlm=Mock(), ocr=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=(
            OcrLine('健康中心', (220, 67, 383, 199), .9117),)),
            width=4096, height=1000)
        answer = scene._answer_read('左上方招牌写的是什么？旁边的门牌号是多少？', ctx)
        scene.ocr.recognize.assert_not_called()
        assert '健康中心' in answer.text


class TestSignboardAndTruncation:
    def test_signboard_describe_appends_ocr_text(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=(
            OcrLine('Cotti Coffee', (1836, 671, 253, 70), .94),
            OcrLine('库迪咖啡', (1890, 720, 137, 48), .83),
            OcrLine('噪声', (10, 10, 20, 20), .3),
        )))
        base = SceneAnswer('我看到的是街道、树木。', 'describe',
                           frame_id='f', vlm_text='街道、树木', confidence=.65)
        with patch.object(ProjectSceneService, '_answer_describe', return_value=base):
            answer = scene._answer_describe('描述一下这条街的店面和环境。', ctx)
        assert '画面中有招牌文字：Cotti Coffee、库迪咖啡。' in answer.text
        assert '噪声' not in answer.text
        assert any(tag.field == 'signboard' for tag in answer.source_tags)

    def test_plain_describe_without_signboard_word_not_appended(self):
        scene = SceneService(vlm=Mock())
        ctx = make_ctx(ocr=OcrResult(lines=(
            OcrLine('库迪咖啡', (1890, 720, 137, 48), .83),)))
        base = SceneAnswer('我看到的是街道。', 'describe', frame_id='f', confidence=.65)
        with patch.object(ProjectSceneService, '_answer_describe', return_value=base):
            answer = scene._answer_describe('画面里有什么？', ctx)
        assert answer.text == '我看到的是街道。'

    def test_dangling_truncated_phrase_dropped(self):
        vlm = Mock()
        vlm.complete.return_value = (
            '["街道上有几棵高大的树木，枝叶茂密",'
            '"街道两侧有建筑物，部分建筑外墙为"]')
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        out = scene._vlm_complete(describe_prompt('描述一下这条街的店面和环境。'), frame)
        assert '外墙为' not in out and '枝叶茂密' in out

    def test_describe_accepts_four_short_phrases(self):
        # 场景描述放开到 4 实体：覆盖"整体+周围环境/路面要素"等问到的方面
        vlm = Mock()
        vlm.complete.return_value = '["篮球场","绿色围栏","树木","白色斑马线"]'
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        out = scene._vlm_complete(describe_prompt('画面里有什么？'), frame)
        assert out == '篮球场、绿色围栏、树木、白色斑马线'

    def test_describe_rejects_five_phrases(self):
        # 超过 schema 上限（4）的结构化回答无效，走既有失败路径
        from aieyes.scene.errors import SceneAnalysisError
        vlm = Mock()
        vlm.complete.return_value = '["a","b","c","d","e"]'
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        with pytest.raises(SceneAnalysisError):
            scene._vlm_complete(describe_prompt('画面里有什么？'), frame)

    def test_short_color_phrases_survive_dangling_filter(self):
        vlm = Mock()
        vlm.complete.return_value = '["红色的狗","白色"]'
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        out = scene._vlm_complete(describe_prompt('画面里有什么？'), frame)
        assert out == '红色的狗、白色'

    def test_counter_prefixed_duplicate_phrase_is_merged(self):
        # 模型同一实体换了个量词说法重复列举时，解析要合并而不是原样播两遍
        vlm = Mock()
        vlm.complete.return_value = '["一个篮球架","篮球架","树木"]'
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        out = scene._vlm_complete(describe_prompt('我面前是什么场地？周围环境怎么样？'), frame)
        assert out == '一个篮球架、树木'

    def test_distinct_nouns_are_never_merged(self):
        # 合并只对"同一短语加了数量前缀"生效，不同实体（人行道/人行横道）必须都保留
        vlm = Mock()
        vlm.complete.return_value = '["人行道","人行横道","篮球场","篮球架"]'
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        out = scene._vlm_complete(describe_prompt('画面里有什么？'), frame)
        assert out == '人行道、人行横道、篮球场、篮球架'


class TestDescribeKeepsModelWording:
    """解析层只做结构校验与去重，不改写模型给的物体名称（类名误判的根因不在解析）。"""

    def test_model_object_names_are_passed_through_unchanged(self):
        # 095239 原图实测原始输出（round4 RAW_LOG 表A 场景-3 A 变体）逐字回放
        vlm = Mock()
        vlm.complete.return_value = (
            '["场地为户外道路区域","路面为灰色水泥材质","路面上有白色斑马线",'
            '"路面上有黄色减速带和黑色警示块"]')
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((50, 80, 3), np.uint8))
        out = scene._vlm_complete(describe_prompt('现在看到的路面情况怎么样？'), frame)
        assert out == ('场地为户外道路区域、路面为灰色水泥材质、路面上有白色斑马线、'
                       '路面上有黄色减速带和黑色警示块')
        assert '篦子' not in out  # 解析层没有类名纠正能力，也没有硬编码答案

    def test_describe_prompt_does_not_seed_road_facility_names(self):
        prompt = describe_prompt('现在看到的路面情况怎么样？')
        assert not any(word in prompt for word in ('减速带', '警示块', '篦子', '井盖'))


class TestCompactSchemaSplit:
    """compact 分支实际有两个使用者：整景描述要给 4 实体，找物单体辨认只能 1 实体。"""

    def _payload(self, prompt):
        from vlm_adapter import VlmClient
        from aieyes.vision.vlm import VlmClient as ProjectVlmClient
        client = object.__new__(VlmClient)
        seen = []
        with patch.object(ProjectVlmClient, '_build_payload',
                          side_effect=lambda p, images, **kw: seen.append(p) or
                          {'messages': [{'role': 'user', 'content': 'x'}]}):
            payload = client._build_payload(prompt, [])
        return payload, seen

    def test_describe_allows_four_entities(self):
        from vlm_adapter import VISUAL_PREFIX
        payload, seen = self._payload(VISUAL_PREFIX + '描述这条街。')
        schema = payload['response_format']['schema']
        assert schema['maxItems'] == 4 and payload['max_tokens'] == 96
        assert seen == ['描述这条街。']  # 前缀剥掉后再交给项目实现

    def test_single_object_allows_only_one_entity(self):
        from vlm_adapter import SINGLE_PREFIX
        payload, seen = self._payload(SINGLE_PREFIX + '图中最主要的物体是什么？')
        schema = payload['response_format']['schema']
        assert schema['maxItems'] == 1 and schema['minItems'] == 1
        assert payload['max_tokens'] == 48
        assert seen == ['图中最主要的物体是什么？']

    def test_find_object_verification_uses_single_prefix(self):
        from vlm_adapter import SINGLE_PREFIX
        from aieyes.scene.models import ObjectCandidates
        vlm = Mock()
        vlm.complete.return_value = '["灭火器"]'
        scene = SceneService(vlm=vlm)
        frame = SimpleNamespace(rgb=np.zeros((80, 160, 3), np.uint8),
                                width=160, height=80, frame_id='f')
        candidate = Mock(box=(10, 10, 40, 40), label='灭火器', position='左侧',
                         confidence=.8, feature='', frame_id='f')
        base = ObjectCandidates('灭火器', True, '灭火器在左边。', frame_id='f',
                                candidates=(candidate,))
        with patch.object(ProjectSceneService, 'find_object', return_value=base):
            scene.find_object(frame, '灭火器')
        assert vlm.complete.call_args.args[0].startswith(SINGLE_PREFIX)


class TestResolutionRouting:
    """分辨率路由：640 压缩只作用于 compact/single，数量/颜色等属性路径按 VlmConfig 走。

    M4 A/B（deepseek_acceptance6/logs/M4_ab_run.py）依赖这条路由事实：
    拿 describe 与 count 当同一档分辨率比较会得出错误结论。
    """

    def _sent_size(self, prompt):
        import base64
        import io
        from PIL import Image
        from vlm_adapter import VlmClient as BenchVlmClient
        from aieyes.config import LlmConfig, VlmConfig

        client = BenchVlmClient(llm_config=LlmConfig(),
                                vlm_config=VlmConfig(image_max_side_px=1024))
        payload = client._build_payload(prompt, [np.zeros((600, 1200, 3), np.uint8)],
                                        max_tokens=None, temperature=None)
        # compact 分支会先插入一条 system 消息，图像在最后一条用户消息里
        content = payload['messages'][-1]['content']
        url = content[1]['image_url']['url']
        return Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[-1]))).size

    def test_describe_path_still_downsamples_to_640(self):
        from vlm_adapter import VISUAL_PREFIX
        width, height = self._sent_size(VISUAL_PREFIX + '描述这条街。')
        assert max(width, height) == 640  # 产品默认 max_side_px=640 不变

    def test_count_path_keeps_config_resolution(self):
        from vlm_adapter import COUNT_PREFIX
        width, height = self._sent_size(COUNT_PREFIX + '用户问：有几具灭火器？')
        assert max(width, height) == 1024  # 不吃 640 compact，按 VlmConfig
