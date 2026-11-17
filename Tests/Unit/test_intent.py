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
    # 真人自然问法（录音 03/06/09）不能掉成聊天。
    assert classify("请告诉我前面有什么东西").kind is IntentKind.LOOK
    natural = classify("嗯告诉我前面有什么东西")
    assert natural.kind is IntentKind.LOOK
    assert natural.text == "嗯告诉我前面有什么东西", "判意不能改写 ASR 原文"
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


def test_explicit_correction_beats_compare_words() -> None:
    # 纠正信号必须先于“左边/右边/哪个”等比较词，否则会新开一个双图比较任务
    for text in (
        "不是这个，我说的是右边那件",
        "不是这个，是右边那个",
        "我说的是左边那件",
        "我说错了，是右边",
        "搞错了，不是这个",
        "不是左边，是右边",
    ):
        got = classify(text)
        assert got.kind is IntentKind.CORRECT, (text, got.kind)
        assert got.text == text, "纠正句不得改写识别原文"
        assert got.raw == text


def test_compare_phrases_without_correction_still_compare() -> None:
    # 反向保护：没有纠正信号的合法比较句不得被纠正规则吞掉
    for text in ("左边和右边哪个贵", "比较一下左边和右边", "这两个有什么区别"):
        assert classify(text).kind is IntentKind.COMPARE, text
    assert classify("右边那件多少钱").kind in (IntentKind.COMPARE, IntentKind.PRICE)


def test_question_tag_is_not_correction() -> None:
    # “对不对/是不是”是疑问尾缀，不能被当成“不对”纠正，否则会回“好的，你说。”
    for text in ("这样对不对", "对不对呀", "这个对不对呢"):
        assert classify(text).kind is not IntentKind.CORRECT, text
    for text in ("这个不对", "不对，我说的是右边", "搞错了"):
        assert classify(text).kind is IntentKind.CORRECT, text


def test_location_question_is_find_not_compare() -> None:
    # “哪个地方/哪个位置”是定位问题，不能因为出现“哪个”就开双图任务
    for text in ("键盘现在在画面中的哪个地方", "键盘在画面哪里", "键盘在屏幕上的哪个位置"):
        assert classify(text).kind is IntentKind.FIND, text
    assert classify("键盘在哪个地方").kind is IntentKind.FIND
    assert classify("哪个地方有超市").kind is IntentKind.FIND


def test_which_plus_attribute_still_compares() -> None:
    for text in ("这两个哪个贵", "哪个便宜", "这两瓶哪个更划算", "这两个拿个贵"):
        assert classify(text).kind is IntentKind.COMPARE, text


def test_pair_question_with_dropped_attribute_still_compares() -> None:
    # 实测 ASR 把“这两个哪个贵”的尾字吞掉，只剩“这两个哪个”；
    # 二选一问句没有形容词也必须走双图比较，不能掉到聊天。
    for text in ("这两个哪个", "这两件哪个", "这两个哪个好"):
        assert classify(text).kind is IntentKind.COMPARE, text
    # 反向保护：位置提问仍是找物，不因“两个/哪个”误开双图
    for text in ("画面中的哪个地方", "图里哪个位置"):
        assert classify(text).kind is IntentKind.FIND, text


def test_short_stop_homophones_are_control_not_chat() -> None:
    # 流式 ASR 把短促“停止”识成“停直/亭止”：整句白名单判 STOP，原文不改写
    for text in ("停直", "亭止", "请停直", "先停直！"):
        got = classify(text)
        assert got.kind is IntentKind.STOP, (text, got.kind)
        assert got.raw == text
        assert got.text == strip_wake_word(text)
    for text in ("停直了吗", "他一直停直着", "别停直", "不要停直"):
        assert classify(text).kind is not IntentKind.STOP, text


def test_bare_stop_does_not_fire_on_nouns_or_statements() -> None:
    # 裸“停”只在命令句里算停止；“停车场/停职/车停了/停了吗”不能被误判
    for text in ("停车场在哪里", "停车方便吗", "停职是什么意思", "我被停职了",
                 "车停了", "停了吗", "公交车停在哪里", "他停了一下"):
        assert classify(text).kind is not IntentKind.STOP, text
    for text in ("停", "停！", "停一下", "停停停", "你先停一下", "停住"):
        assert classify(text).kind is IntentKind.STOP, text


def test_flip_side_is_not_a_compare_task() -> None:
    # “翻面/翻个面”是看另一面，不是比较两件东西；无对应场景 kind 时落 CHAT
    for text in ("翻面", "翻个面", "转过来"):
        assert classify(text).kind is IntentKind.CHAT, text
    assert classify("看看背面").kind is IntentKind.LOOK
