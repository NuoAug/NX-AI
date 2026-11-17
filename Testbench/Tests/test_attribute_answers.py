from unittest.mock import Mock
from types import SimpleNamespace
import numpy as np
from scene_guard import SceneService
from aieyes.scene.prompts import describe_prompt
from vlm_adapter import VlmClient,COLOR_PREFIX,COUNT_PREFIX

def test_typed_attribute_payload_excludes_free_position_text():
 v=VlmClient()
 for prefix,key in [(COLOR_PREFIX,'color'),(COUNT_PREFIX,'count')]:
  p=v._build_payload(prefix+'问题',[],max_tokens=64,temperature=0)
  schema=p['response_format']['schema'];assert key in schema['properties'];assert not schema['additionalProperties']
 v.close()

def test_attribute_reply_contains_only_the_requested_fact():
 frame=SimpleNamespace(rgb=np.zeros((50,80,3),np.uint8))
 for q,raw,expected in [('是什么颜色？','{"count":2,"color":"红色"}','红色。'),('有几个？','{"count":2}','有2个。'),('有几个？','{"count":0}','没有看到这个目标。')]:
  v=Mock();v.complete.return_value=raw;s=SceneService(vlm=v)
  assert s._vlm_complete(describe_prompt(q),frame)==expected

def test_absent_target_cannot_borrow_another_objects_color():
 v=Mock();v.complete.return_value='{"count":0,"color":"蓝色"}';s=SceneService(vlm=v)
 assert '无法判断颜色' in s._vlm_complete(describe_prompt('灭火器是什么颜色？'),SimpleNamespace(rgb=np.zeros((50,80,3),np.uint8)))
