from unittest import TestCase
from unittest.mock import patch, Mock
import numpy as np
from scene_guard import SceneService, ProjectSceneService, confirmed_result
from aieyes.scene.models import ObjectCandidates, ObjectCandidate, SceneAnswer
from aieyes.interfaces import ImageFrame

class GuardTests(TestCase):
    def test_strict_confirmation(self):
        self.assertTrue(confirmed_result('["灭火器"]','灭火器'))
        for s in ['["篮球架底座"]','["灭火器标牌"]','["不是灭火器"]','["疑似灭火器"]','["{name:灭火器}"]','["灭火器","门"]','{}','broken']:
            self.assertFalse(confirmed_result(s,'灭火器'))

    def test_positive_negative_and_failure(self):
        from types import SimpleNamespace
        frame=SimpleNamespace(rgb=np.zeros((120,160,3),np.uint8),width=160,height=120)
        c=ObjectCandidate('目标',(30,30,40,40),'中间')
        r=ObjectCandidates('目标',True,'旧答案',(c,))
        for answer,passed in [('["目标"]',True),('["其他物体"]',False),('broken',False), (RuntimeError('timeout'),False)]:
            vlm=Mock()
            if isinstance(answer,Exception):vlm.complete.side_effect=answer
            else:vlm.complete.return_value=answer
            scene=SceneService(vlm=vlm)
            with patch.object(ProjectSceneService,'find_object',return_value=r):
                result=scene.find_object(frame,'目标')
            self.assertNotIn('目标',vlm.complete.call_args.args[0])
            self.assertEqual(result.found,passed)
            self.assertEqual(bool(result.candidates),passed)
            if not passed:self.assertIn('candidate_unverified',result.degraded)

    def test_voice_look_hint_still_routes_explicit_read_question_to_ocr(self):
        scene=SceneService()
        with patch.object(ProjectSceneService,'answer',return_value=SceneAnswer('读到文字','read')) as parent:
            scene.answer('读一下牌子上的字',intent_hint='describe')
        self.assertEqual(parent.call_args.kwargs['intent_hint'],'read')
