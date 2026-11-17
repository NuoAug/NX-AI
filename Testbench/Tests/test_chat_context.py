"""聊天历史写入时机与“无图像事实”分支的结构检查。

只断言消息结构和程序化话术，不涉及模型回答质量。
"""
import json

import pytest
from unittest.mock import Mock
from chat_adapter import make_chat
from vlm_adapter import CHAT_PREFIX

FIXED_VIEW_ANSWER = '我还没看到画面，让我看一眼再告诉你，好吗？'


def _messages(prompt):
    """把 build_prompt 的输出还原成 [(role, content)]，判定只看实际收到的结构。"""
    assert prompt.startswith(CHAT_PREFIX)
    return [(m['role'], m['content']) for m in json.loads(prompt[len(CHAT_PREFIX):])]


class RecordingVlm:
    """记录每次收到的 prompt 全文，回答可预期，便于判定跨轮串答。"""

    def __init__(self):
        self.prompts = []

    def complete(self, prompt, images=None, **kwargs):
        self.prompts.append(prompt)
        return '答。'

    def complete_stream(self, prompt, images=None, **kwargs):
        self.prompts.append(prompt)

        def gen():
            yield '半'
            yield '分'

        return gen()


class NoModel:
    def complete(self, *args, **kwargs):
        raise AssertionError('无图现场问题不得交给模型自由描述')

    def complete_stream(self, *args, **kwargs):
        raise AssertionError('无图现场问题不得交给模型自由描述')


class CapturingVlm:
    def __init__(self):
        self.calls = []

    def complete(self, prompt, images=None, **kwargs):
        self.calls.append(list(images or []))
        return '好的。'

    def complete_stream(self, prompt, images=None, **kwargs):
        self.calls.append(list(images or []))
        return iter(['好的。'])


@pytest.mark.parametrize('text', [
    '那边有人吗？',
    '这盏灯亮着吗',
    '画面里是什么',
    '你看见我手里的东西了吗',
    '对面有没有台阶',
])
def test_no_image_view_question_is_answered_without_model(text):
    chat = make_chat(NoModel())
    assert '还没看到画面' in chat.ask(text).text
    assert '还没看到画面' in ''.join(chat.ask_stream(text))


@pytest.mark.parametrize('text', [
    # E 组复审给出的 2 条漏拦
    '墙上有什么',
    '地上有东西吗',
    # 同类处所/实物问句
    '包里有没有钥匙',
    '门上挂着什么',
    '架子上是什么',
    '地上有水吗',
    '身上有字吗',
    '屋里有动静吗',
    '车里有没有人',
    '窗上写着什么',
])
def test_place_and_object_question_needs_picture(text):
    chat = make_chat(NoModel())
    assert chat._grounded_answer(text) == '我还没看到画面，让我看一眼再告诉你，好吗？'
    assert '还没看到画面' in chat.ask(text).text


@pytest.mark.parametrize('text', ['今天天气怎么样', '我想出门', '这是什么意思', '这盏灯怎么开', '我手上有点冷',
                                  '我手上有点疼', '你刚才看到我说的那个做法了吗', '画面里的人物是谁',
                                  # 比喻/身体类不属现场，不得被处所词表误拦
                                  '我脑子里有什么想法', '我身上有什么毛病', '哪里有好吃的', '有什么好听的歌'])
def test_non_view_question_still_reaches_model(text):
    chat = make_chat(NoModel())
    assert chat._grounded_answer(text) is None


@pytest.mark.parametrize('text', [
    # Lead 复审给出的 7 条误拦句：问的是消息/文件/知识/推荐，不是当前现场
    '这里有什么好吃的',
    '你看见我发的消息了吗',
    '你看到我简历了吗',
    '桌上有什么成语',
    '前面有什么好吃的推荐',
    '这边有什么公交',
    '这里有什么好玩的地方',
    # 同类句式的其它说法：豁免要能覆盖，不是把 7 句写死
    '你看到我发的文件了吗',
    '我发的记录你看到了吗',
    '这边有什么推荐的餐厅',
    '桌上有什么典故',
])
def test_view_exempt_question_is_not_treated_as_missing_picture(text):
    chat = make_chat(NoModel())
    assert chat._grounded_answer(text) is None


@pytest.mark.parametrize('text', ['对面有没有台阶', '旁边有人吗', '那边有什么', '你看见我手里的东西了吗'])
def test_view_question_still_blocked_without_picture(text):
    chat = make_chat(NoModel())
    assert '还没看到画面' in chat.ask(text).text


def test_previous_image_is_not_reused_for_text_question():
    vlm = CapturingVlm()
    chat = make_chat(vlm)
    frame = object()
    chat.ask('这是什么？', images=[frame])
    chat.ask('那现在呢？')
    assert [len(images) for images in vlm.calls] == [1, 0]


def test_grounded_stream_answer_is_recorded_once_by_caller():
    chat = make_chat(NoModel())
    question = '12元买两件各5元的东西，还剩几元？'
    text = ''.join(chat.ask_stream(question))
    assert text == '还剩2元。'
    # 流式契约：本方法不落历史，由调用方（引擎）读完流后 record，避免多记。
    assert chat.turns == []
    chat.record(question, text)
    assert [(turn.user, turn.assistant) for turn in chat.turns] == [(question, text)]
    chat.ask(question)
    assert len(chat.turns) == 2


def test_model_failure_records_one_empty_answer_not_two():
    chat = make_chat(Mock())
    chat.vlm.complete.return_value = ''
    assert chat.ask('讲个故事').text == ''
    assert len(chat.turns) == 1
    assert chat.turns[0].assistant == ''


# ---------------------------------------------------------------- 跨轮序列（S1/S3/S4/S5）
def test_unfinished_first_round_does_not_leak_into_second_prompt():
    # S1 快速连问：第一轮只出了半句就被第二轮打断，第二轮 prompt 里不能出现第一轮的问句或半截回答。
    vlm = RecordingVlm()
    chat = make_chat(vlm)
    stream = chat.ask_stream('第一轮问题：讲个故事')
    assert next(stream) == '半'
    chat.ask('第二轮问题：现在几点')
    messages = _messages(vlm.prompts[-1])
    assert [role for role, _ in messages] == ['system', 'user']
    assert messages[-1][1] == '第二轮问题：现在几点'
    assert '第一轮问题' not in vlm.prompts[-1]
    assert '半' not in messages[-1][1]


def test_abandoned_stream_is_not_written_to_history():
    # S3 取消一轮后立刻问新问题：被放弃的尝试不得进历史，也不得作为下一轮上下文。
    vlm = RecordingVlm()
    chat = make_chat(vlm)
    stream = chat.ask_stream('放弃的问题：讲个故事')
    next(stream)
    stream.close()
    assert chat.turns == []
    chat.ask('新问题：你还在吗')
    messages = _messages(vlm.prompts[-1])
    assert [role for role, _ in messages] == ['system', 'user']
    assert messages[-1][1] == '新问题：你还在吗'
    assert '放弃的问题' not in vlm.prompts[-1]


def test_stream_pieces_are_not_recorded_until_caller_records():
    # record 落盘时机：流式片段一律不落，只有调用方拿到定稿全文才记一条。
    chat = make_chat(RecordingVlm())
    stream = chat.ask_stream('讲个故事')
    assert chat.turns == []
    assert ''.join(stream) == '半分'
    assert chat.turns == []
    chat.record('讲个故事', '半分')
    assert [(turn.user, turn.assistant) for turn in chat.turns] == [('讲个故事', '半分')]


def test_blocked_view_answer_is_only_what_was_actually_spoken():
    # S4 无画面拦截：历史里记的就是播出去的固定答句，不含任何“模型回答”痕迹。
    chat = make_chat(NoModel())
    chat.ask('那边有人吗？')
    assert [(turn.user, turn.assistant) for turn in chat.turns] == [('那边有人吗？', FIXED_VIEW_ANSWER)]
    messages = _messages(chat.build_prompt('我有点无聊'))
    assert [m for m in messages if m == ('assistant', FIXED_VIEW_ANSWER)] == [('assistant', FIXED_VIEW_ANSWER)]
    assert [m for m in messages if m[0] == 'assistant'] == [('assistant', FIXED_VIEW_ANSWER)]
    assert messages[-1] == ('user', '我有点无聊')


def test_name_correction_overrides_and_history_keeps_chronological_order():
    # S2 用户纠正：问名字直接答最新称呼且不调模型；历史按原顺序保留，纠正句在旧声明之后。
    chat = make_chat(NoModel())
    chat.record('我叫陈晨，请记住。', '模型回答')
    chat.record('我叫李敏，不叫陈晨。', '模型回答')
    assert chat._grounded_answer('我现在叫什么？') == '你叫李敏。'
    assert chat.ask('我现在叫什么？').text == '你叫李敏。'
    messages = _messages(chat.build_prompt('随便聊两句'))
    users = [content for role, content in messages if role == 'user']
    assert users.index('我叫李敏，不叫陈晨。') > users.index('我叫陈晨，请记住。')
    assert messages[-1] == ('user', '随便聊两句')


# ---------------------------------------------------------------- 严格交替（S5）
def test_empty_answer_turn_is_skipped_so_roles_stay_alternating():
    # 模型失败留下空回答的旧轮次不重放：否则 prompt 里会出现连续两条 user，并把旧问句塞到当前问题前面。
    chat = make_chat(NoModel())
    chat.record('问题一', '')
    chat.record('问题二', '答二')
    messages = _messages(chat.build_prompt('现在的问题'))
    assert [role for role, _ in messages] == ['system', 'user', 'assistant', 'user']
    assert messages[1] == ('user', '问题二')
    assert messages[2] == ('assistant', '答二')
    assert messages[3] == ('user', '现在的问题')
    assert '问题一' not in [content for _, content in messages]


def test_build_prompt_alternates_roles_after_system():
    chat = make_chat(NoModel())
    for index in range(3):
        chat.record(f'问题{index}', '' if index == 1 else f'答{index}')
    roles = [role for role, _ in _messages(chat.build_prompt('当前'))]
    assert roles[0] == 'system'
    assert roles[-1] == 'user'
    for left, right in zip(roles[1:], roles[2:]):
        assert left != right, roles


def test_system_prompt_is_always_first_message():
    chat = make_chat(NoModel())
    chat.record('你好', '你好呀')
    messages = _messages(chat.build_prompt('在吗'))
    assert messages[0][0] == 'system'
    assert '小冰' in messages[0][1]


# ---------------------------------------------------------------- 无画面拦截三组判定探针
MUST_BLOCK = (
    '那边有人吗？', '这盏灯亮着吗', '画面里是什么', '你看见我手里的东西了吗', '对面有没有台阶',
    '墙上有什么', '地上有东西吗', '包里有没有钥匙', '门上挂着什么', '架子上是什么',
    '地上有水吗', '身上有字吗', '屋里有动静吗', '车里有没有人', '窗上写着什么',
    '旁边有人吗', '那边有什么', '桌上放着什么', '手里拿的是什么', '镜头里有没有人',
    '照片上是什么颜色', '门后有什么', '楼上有声音吗', '房间里有没有人', '屋里头有人吗',
    '这儿开着灯吗', '你能看见我的杯子吗', '你看清牌子上的字了吗', '衣服上有什么图案',
    '后面有什么', '旁边是什么', '墙上挂着什么', '地上有没有水', '架子上放着什么',
    '画面上有什么文字', '对面是什么', '背后有什么',
    # E 组 7 条结构性缺口：旧词表按“处所词枚举”写，句子换一个处所词就漏拦
    '柜子上有什么', '抽屉里有什么', '窗外有什么', '桌子上有什么', '床上有什么',
    '这附近有人吗', '周围有人吗',
    # 同类里同样没出现在旧词表的处所：结构正则必须覆盖，不是把这 7 句写死
    '柜子上面有什么', '鞋柜上面有什么', '茶几上放着什么', '冰箱里有什么', '沙发下面有东西吗',
    '床头柜上摆着什么', '阳台外面有人吗', '楼道里有没有人', '天花板上有东西吗',
    '走廊尽头有什么', '门框上贴着什么', '袋子里是什么', '枕头旁边有什么', '窗外有人吗',
    '桌子下面有东西吗', '床头上放着什么',
)

MUST_EXEMPT = (
    '这里有什么好吃的', '你看见我发的消息了吗', '你看到我简历了吗', '桌上有什么成语',
    '前面有什么好吃的推荐', '这边有什么公交', '这里有什么好玩的地方', '你看到我发的文件了吗',
    '我发的记录你看到了吗', '这边有什么推荐的餐厅', '桌上有什么典故', '这边有没有推荐的餐厅',
    '前面有什么好玩的', '桌上有什么故事', '你看见我说的那句话了吗', '前面有什么超市',
    '屋里的文件你看到了吗', '照片上有什么故事', '桌上有什么好喝的', '这边有什么医院',
    '面前有什么通知', '你看到我发的语音了吗', '墙上有什么通知',
    # 结构正则收紧后的反向保护：处所+询问词命中，但问的是知识/推荐/消息，必须先豁免
    '桌子有什么讲究', '墙上这幅画有什么寓意', '柜子有什么说法', '窗外有什么好玩的',
    '门口有什么好吃的', '车里有什么消息', '抽屉里有什么文件', '背后有什么原因',
)

MUST_ALLOW = (
    '今天天气怎么样', '我想出门', '这是什么意思', '这盏灯怎么开', '我手上有点冷',
    '我手上有点疼', '你刚才看到我说的那个做法了吗', '我脑子里有什么想法', '我身上有什么毛病',
    '哪里有好吃的', '有什么好听的歌', '讲个故事', '我有点无聊', '你叫什么名字',
    '现在几点了', '明天要带伞吗', '你觉得我应该先做什么', '谢谢', '我累了',
    '我们聊点别的吧', '你能不能陪我说话', '帮我把灯关掉',
)

# 身份类问句（“谁”）沿用 E 组结论交给模型，不当作无图现场问句；列在这里是为了边界可查。
KNOWN_UNBLOCKED = ('画面里的人物是谁', '前面是谁')


@pytest.mark.parametrize('text', MUST_BLOCK)
def test_view_probe_must_block(text):
    chat = make_chat(NoModel())
    assert chat._grounded_answer(text) == FIXED_VIEW_ANSWER


@pytest.mark.parametrize('text', MUST_EXEMPT)
def test_view_probe_must_exempt(text):
    chat = make_chat(NoModel())
    # 这些句子本身命中现场问句句式，靠豁免词放行；豁免一旦被吞掉就会变成误拦。
    assert chat._view_missing.search(text) is not None
    assert chat._grounded_answer(text) is None


@pytest.mark.parametrize('text', MUST_ALLOW)
def test_view_probe_must_allow(text):
    chat = make_chat(NoModel())
    assert chat._grounded_answer(text) is None


@pytest.mark.parametrize('text', KNOWN_UNBLOCKED)
def test_identity_question_is_left_to_model(text):
    chat = make_chat(NoModel())
    assert chat._grounded_answer(text) is None


def test_view_probe_group_counts():
    chat = make_chat(NoModel())
    blocked = [text for text in MUST_BLOCK if chat._grounded_answer(text) == FIXED_VIEW_ANSWER]
    exempted = [text for text in MUST_EXEMPT if chat._grounded_answer(text) is None]
    allowed = [text for text in MUST_ALLOW if chat._grounded_answer(text) is None]
    total = len(MUST_BLOCK) + len(MUST_EXEMPT) + len(MUST_ALLOW)
    assert total >= 60
    assert (len(blocked), len(exempted), len(allowed)) == (len(MUST_BLOCK), len(MUST_EXEMPT), len(MUST_ALLOW))
    assert len({*MUST_BLOCK, *MUST_EXEMPT, *MUST_ALLOW}) == total
