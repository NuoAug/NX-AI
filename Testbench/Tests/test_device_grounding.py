import json
import numpy as np
from aieyes.interfaces import ImageFrame,OcrLine,OcrResult
from scene_guard import SceneService
from vlm_adapter import LOCATE_PREFIX,VISUAL_PREFIX

class FakeVlm:
    def __init__(self,located=True):self.located=located;self.calls=[]
    def complete(self,prompt,**kw):
        self.calls.append(prompt)
        if prompt.startswith(LOCATE_PREFIX):return json.dumps({'found':self.located,'box':[250,250,750,750]})
        assert '归还' in prompt
        return '带插槽的租借设备。'
class FakeOcr:
    def recognize(self,rgb,**kw):return OcrResult(lines=(OcrLine('归还',(2,3,10,10),.95),))

def test_device_answer_keeps_crop_evidence_and_coordinates():
    s=SceneService(vlm=FakeVlm(),ocr=FakeOcr());f=ImageFrame(np.full((200,300,3),128,np.uint8),0,'fixture','file')
    a=s.answer('这个设备是什么？',frame=f,intent_hint='describe')
    assert a.text=='我看到的是带插槽的租借设备。'
    assert any(t.field=='device_detail' and '归还' in t.raw_text for t in a.source_tags)
    assert a.ocr_lines[-1].box[0]>2

def test_unlocated_device_does_not_invent_identity():
    s=SceneService(vlm=FakeVlm(False),ocr=FakeOcr());f=ImageFrame(np.full((200,300,3),128,np.uint8),0,'fixture','file')
    a=s.answer('这个设备是什么？',frame=f,intent_hint='describe')
    assert 'device_unverified' in a.degraded
    assert '不能确认' in a.text
    assert len(s.vlm.calls)==1
