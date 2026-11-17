"""A stop during setup must not launch a later expensive inference."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from debug import DebugRunner


@pytest.mark.parametrize('module',['vision','ocr','compare','chat'])
def test_stop_during_setup_does_not_start_inference(module):
    events=[]; finished=[]
    ctx=SimpleNamespace(stopped=False,emit=lambda *a:events.append(a),
                        log=lambda *a:None,error=lambda *a:events.append(('error',a)),
                        finished=lambda *a:finished.append(a))
    runner=DebugRunner(ctx);runner._module=module
    runner._input={'question':'问题','source':'上传图片 A','text':'问题'}
    runner._acquire_frame=lambda *a:(object(),{})
    scene=Mock()
    def setup(*args):ctx.stopped=True;return scene
    runner._make_scene=setup
    runner._make_vlm=lambda:None
    runner._make_chat=setup
    runner._compare_state={'a':object(),'b':object()}
    if module=='compare':runner._run_compare=lambda:runner._compare_frames(object(),object(),'比较')
    runner._thread_main()
    assert finished==[('interrupted','')]
    scene.answer.assert_not_called();scene.compare.assert_not_called()
    scene.ask_stream.assert_not_called();scene.read_text_result.assert_not_called()
    assert not any(e[0] in ('error','assistant.final') for e in events)


def test_stop_during_ocr_does_not_launch_answer_inference():
    ctx=SimpleNamespace(stopped=False,emit=lambda *a:None,log=lambda *a:None,
                        timing=lambda **kw:None,error=Mock(),finished=Mock())
    runner=DebugRunner(ctx);runner._module='ocr';runner._input={}
    runner._acquire_frame=lambda *a:(object(),{})
    scene=Mock()
    def read(frame):ctx.stopped=True;return SimpleNamespace(lines=[],t_ms=1)
    scene.read_text_result=read;runner._make_scene=lambda:scene
    runner._thread_main()
    scene.answer.assert_not_called();ctx.error.assert_not_called()
    ctx.finished.assert_called_once_with('interrupted','')
