from unittest.mock import Mock,patch
from chat_adapter import make_chat
from scene_guard import SceneService,ProjectSceneService
from aieyes.scene.models import SceneAnswer

def test_missing_view_is_grounded_in_both_response_paths():
 v=Mock();c=make_chat(v)
 for question in ['你看看我面前有什么。','眼前是什么？']:
  assert '还没看到画面' in c.ask(question).text
  assert '还没看到画面' in ''.join(c.ask_stream(question))
 v.complete.assert_not_called()

def test_unmeasured_status_does_not_declare_success():
 c=make_chat(Mock())
 assert '不能确定' in c.ask('测试通过了吗？').text

def test_visual_companion_wording_preserves_facts_and_evidence():
 source=SceneAnswer('红色。','describe',vlm_text='红色。',confidence=.8,evidence=('原始证据',))
 with patch.object(ProjectSceneService,'answer',return_value=source):
  result=SceneService().answer('是什么颜色？')
 assert result.text=='我看到是红色的。'
 assert result.evidence==source.evidence and result.confidence==source.confidence
