import pytest
from intent_adapter import classify, control_feedback
from aieyes.intent import IntentKind


@pytest.mark.parametrize('text',[
    '停职是什么意思','我被停职了','公交车停在哪里','车停了',
    '不要关闭麦克风','不要取消静音','别再取消静音',
    '我没有让你关闭麦克风','我想继续上学','继续学习',
    '怎么清空资料','清空资料是什么意思','不是让你清空资料',
    '怎样清空资料','怎么才能清空资料','清空资料怎么操作',
    '我该如何删除这轮对话','怎样关闭麦克风','如何才能取消静音',
    '请解释一下关闭麦克风是什么意思',
    '我不想关闭麦克风','我不希望你取消静音','不要让麦克风静音',
    '静音了吗','你刚才是不是关麦了',
    # 状态询问/客观陈述，不是停止命令（放开重复“停”后仍需保持）
    '停了吗','车停了','停职是什么意思','公交车停在哪里',
    # acceptance5 补充：名词“停车/停车场”、量词“一下”、陈述“我停直了”
    '他停了一下','停车场在哪里','我停直了','停车方便吗',
])
def test_control_mentions_do_not_execute_actions(text):
    result=classify(text)
    assert result.kind==IntentKind.CHAT
    assert result.raw==text


@pytest.mark.parametrize(('text','expected'),[
    ('别说了',IntentKind.STOP),('请停一下',IntentKind.STOP),
    ('停止播放',IntentKind.STOP),('暂停',IntentKind.STOP),
    ('请你现在停',IntentKind.STOP),
    # 真机上连续喊停的口语形式（叠字/重述），此前被降级成 CHAT
    ('停停停',IntentKind.STOP),('停一停',IntentKind.STOP),('停停停！',IntentKind.STOP),
    ('停一停吧',IntentKind.STOP),('我们先停一下',IntentKind.STOP),('能不能停止播放',IntentKind.STOP),
    ('关麦',IntentKind.MUTE),('请关闭麦克风',IntentKind.MUTE),
    ('别听了',IntentKind.MUTE),('不用听了',IntentKind.MUTE),
    ('请取消静音',IntentKind.UNMUTE),
    ('请清空资料',IntentKind.CLEAR),
    ('继续说刚才的回答',IntentKind.RESUME),('请继续比较',IntentKind.RESUME),
    ('继续',IntentKind.RESUME),
    ('请关闭麦克风，静音是什么意思',IntentKind.MUTE),
])
def test_real_control_commands_keep_their_behavior(text,expected):
    assert classify(text).kind==expected


@pytest.mark.parametrize(('text','feedback'),[
    ('不要关或开麦克风','好，我不更改麦克风设置。'),
    ('别关或开麦克风','好，我不更改麦克风设置。'),
    ('别开或关麦克风','好，我不更改麦克风设置。'),
    ('不要开或关麦克风','好，我不更改麦克风设置。'),
    ('不要关闭或打开麦克风','好，我不更改麦克风设置。'),
    ('不要开关麦克风','好，我不更改麦克风设置。'),
])
def test_negated_conjoined_control_words_do_not_execute(text,feedback):
    result=classify(text)
    assert result.kind==IntentKind.CHAT and result.raw==text
    assert control_feedback(text)==feedback


@pytest.mark.parametrize('text',['关麦克风有什么作用','静音有什么用',
                                 '开麦干什么用的','取消静音有啥用'])
def test_purpose_question_about_control_words_does_not_execute(text):
    result=classify(text)
    assert result.kind==IntentKind.CHAT and result.text==text


# 否定优先级（C 组复核补充）：否定词+完整动作短语必须先于正向动作匹配，
# 否则“不要打开静音”“别关掉静音”会被正向匹配落到 MUTE，方向相反地真的执行静音。
@pytest.mark.parametrize(('text','feedback'),[
    ('不要打开静音','好，我不更改静音设置。'),
    ('别关掉静音','好，我不更改静音设置。'),
    ('别把麦克风关了','好，我不更改麦克风设置。'),
    ('不要把麦克风关掉','好，我不更改麦克风设置。'),
    ('不要关声音','好，我不更改静音设置。'),
    ('别关声音','好，我不更改静音设置。'),
    ('别把音量关了','好，我不更改静音设置。'),
])
def test_negated_action_phrase_takes_priority_over_positive_match(text,feedback):
    result=classify(text)
    assert result.kind==IntentKind.CHAT
    assert result.text==text and result.raw==text
    assert control_feedback(text)==feedback


# 成对反例：同一批正向控制短语在否定优先级生效后仍是可执行动作。
@pytest.mark.parametrize(('text','kind'),[
    ('关掉麦克风',IntentKind.MUTE),
    ('关闭麦克风',IntentKind.MUTE),
    ('静音',IntentKind.MUTE),
    ('关麦',IntentKind.MUTE),
    ('打开麦克风',IntentKind.UNMUTE),
    ('开麦',IntentKind.UNMUTE),
    ('取消静音',IntentKind.UNMUTE),
    ('请关闭麦克风',IntentKind.MUTE),
])
def test_positive_control_phrases_are_not_swallowed_by_negation(text,kind):
    assert classify(text).kind==kind
    assert control_feedback(text) is None
