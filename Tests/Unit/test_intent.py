"""意图路由单测（纯函数，无设备依赖）。"""

from aieyes.intent import IntentKind, classify, strip_wake_word


def test_strip_wake_word_variants() -> None:
    assert strip_wake_word("小冰，你好呀") == "你好呀"
    assert strip_wake_word("小冰") == ""
    assert strip_wake_word("  小冰 我在吗！ ") == "我在吗"


def test_classify_chat() -> None:
    intent = classify("小冰，给我讲个笑话")
    assert intent.kind is IntentKind.CHAT
    assert intent.text == "给我讲个笑话"


def test_classify_stop_has_priority() -> None:
    for text in ("别说了", "小冰，停一下", "闭嘴", "取消"):
        assert classify(text).kind is IntentKind.STOP


def test_classify_vision_intents() -> None:
    assert classify("帮我看看前面是什么").kind is IntentKind.LOOK
    assert classify("这两个商品哪个便宜").kind is IntentKind.COMPARE
    assert classify("这个多少钱").kind is IntentKind.PRICE
    assert classify("保质期到什么时候").kind is IntentKind.DATE
    assert classify("我的钥匙在哪").kind is IntentKind.FIND


def test_classify_repeat_and_empty() -> None:
    assert classify("再说一遍").kind is IntentKind.REPEAT
    assert classify("小冰").kind is IntentKind.UNKNOWN
    assert classify("").kind is IntentKind.UNKNOWN


def test_stop_negation_is_not_stop() -> None:
    for text in ("不是让你停", "别停", "不要停", "我没有让你停"):
        assert classify(text).kind is not IntentKind.STOP, text


def test_backchannel_and_correct() -> None:
    for text in ("嗯", "好", "好的", "对", "行", "可以"):
        assert classify(text).kind is IntentKind.BACKCHANNEL, text
    assert classify("不对，我说的是那个").kind is IntentKind.CORRECT


def test_mute_unmute_intents() -> None:
    assert classify("关麦").kind is IntentKind.MUTE
    assert classify("静音").kind is IntentKind.MUTE
    assert classify("取消静音").kind is IntentKind.UNMUTE
    assert classify("取消").kind is IntentKind.STOP


def test_compare_asr_variants_and_resume() -> None:
    # ASR 常把“哪个”听成“拿个/那个”，仍应走双图比较
    for text in ("这两个拿个贵", "这两个那个贵", "这两瓶哪个更划算", "比较一下这两样"):
        assert classify(text).kind is IntentKind.COMPARE, text
    for text in ("继续", "接着看", "继续比较一下", "我们继续"):
        assert classify(text).kind is IntentKind.RESUME, text
    # ASR 曾把“继续比较这两样东西的价格”识别成“继需比较这两样东西的价格”，
    # 句首“继…”必须走恢复而不是新开一个比较任务
    for text in ("继需比较这两样东西的价格", "接续刚才的比较", "继续看"):
        assert classify(text).kind is IntentKind.RESUME, text
