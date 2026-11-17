from intent_adapter import classify
from aieyes.intent import IntentKind

def test_recorded_visual_questions():
    for text in ['请告诉我前面有什么东西','前方有什么','眼前有啥','我面前有什么物体','黄色设备是什么？']:
        assert classify(text).kind == IntentKind.LOOK

def test_control_and_other_intents_keep_priority():
    assert classify('前面有什么，停止').kind == IntentKind.STOP
    assert classify('今天天气不错我准备出去走一走').kind == IntentKind.CHAT
    assert classify('我之前提到的故事有趣吗').kind == IntentKind.CHAT
    assert classify('这件东西多少钱').kind == IntentKind.PRICE


def test_world_location_question_is_not_physical_object_search():
    assert classify('法国首都在哪里？').kind==IntentKind.CHAT
    assert classify('钥匙在哪里？').kind==IntentKind.FIND


def test_recorded_visual_homophones_route_to_look():
    for text in ('原前是什么吧', '健钱有什么', '前眼有什么', '这个是什么', '它有什么'):
        assert classify(text).kind == IntentKind.LOOK, text
    assert classify('这样对不对').kind != IntentKind.CORRECT


def test_live_location_question_is_find_not_compare():
    for text in ('键盘现在在画面中的哪个地方','键盘在画面哪里','键盘在屏幕上的哪个位置'):
        assert classify(text).kind==IntentKind.FIND, text
    assert classify('这两个哪个贵').kind==IntentKind.COMPARE
    assert classify('画面里有什么').kind==IntentKind.LOOK
    assert classify('你看到什么了').kind==IntentKind.LOOK


def test_direct_visual_attribute_questions_do_not_fall_back_to_chat():
    assert classify('灭火器是什么颜色').kind==IntentKind.LOOK
    assert classify('有几个灭火器').kind==IntentKind.LOOK
    assert classify('我有几个问题想问你').kind==IntentKind.CHAT


def test_recorded_device_homophone_keeps_asr_text_unchanged():
    # 清单第 1 项红线：判定可以用同音归一化，但识别文本必须原样进入下游与历史。
    result=classify('黄色设备是神魔')
    assert result.kind==IntentKind.LOOK
    assert result.text=='黄色设备是神魔' and result.raw=='黄色设备是神魔'
    assert classify('游戏里的这个设备是神魔').text=='游戏里的这个设备是神魔'
