"""B 组定向回归：意图层只做判定，不得按预期台词改写 ASR 原文。

对应《视觉语言交互子系统待修复清单》第 1、6、7 项红线：
- ASR 原文必须原样进入下游与历史，不能为了命中“标准短语”而回写识别文本；
- 短控制词的近音误识别只能走显式、可审计的判定表，判定结果同样不得篡改文本。

本轮把语料扩到 59 条（近音/叠字/疑问/短控制/中英混/数字/标点空白），并按
“ASR 适配器 -> 意图层”的真实链路再验一遍：ASR 文本一字不改地到达引擎，
意图层只允许做唤醒词与标点归一（strip_wake_word），内容不得替换。
"""
import pytest
import intent_adapter
from intent_adapter import classify, control_feedback
from aieyes.intent import IntentKind, classify as project_classify, strip_wake_word

CORPUS = [
    # 近音（ASR 把“什么”识别成“神魔”、把“停止”识别成“停直/亭止”）
    '柜子里是神魔？',
    '黄色设备是神魔',
    '这台机器是神魔',
    '屏幕上是神魔',
    '前面那个装置是神魔',
    '游戏里的这个设备是神魔',
    '设备是什么？',
    '前面有什么东西',
    '小冰一加一等于几',
    '小斌小斌',
    '小兵你好',
    '继需比较',
    '接续刚才的',
    '停直',
    '亭止',
    # 叠字 / 重复喊
    '停停停',
    '好好好',
    '嗯嗯',
    '慢慢说',
    '看看这个',
    '开开开',
    '小冰小冰小冰',
    # 短控制词
    '停止',
    '停',
    '停一下',
    '别说了',
    # 标点 / 空白 / 引号
    '停止！',
    '停，',
    '停止。。。',
    '什么…',
    '“这是一句话。”',
    '  前面有什么  ',
    # 中英混
    '帮我打开 wifi',
    'OK 停止',
    '小冰 hello',
    'hello 小冰',
    '把 LED 关掉',
    '这个 APP 是什么',
    '看一下这个 API 文档',
    # 数字
    '第二个抽屉里有什么',
    '价格 19.9 吗',
    '保质期多久',
    # 疑问
    '你还有什么要说的吗',
    # 否定 / 处所旁询问
    '不要关或开麦克风',
    '关麦克风有什么作用',
    '别把麦克风关了',
    '不要打开静音',
    '怎么清空资料',
    # 极短应答
    '谢谢',
    '好的',
    '嗯',
    '啊？',
    # 不该被当成停止命令的口语
    '我停直了',
    '停职是什么意思',
    '公交车停在哪里',
    '停了吗',
    # acceptance5 补充：唤醒词前后缀、停顿、数字、逗号并列、末字
    '小冰。',
    '小冰，停止',
    '亭止了吗',
    '停一停',
    '停一停吧',
    '他停了一下',
    '停车场在哪里',
    '第二排第三个是什么',
    '价格是 19.9 元吗',
    '打开 Wi-Fi 然后停止',
    '嗯……让我想想',
    '这是第 3 个设备，对吧？',
    # 真实 paraformer 输出（离线捕获，见 B_speech2.md 证据表）
    '今好你好我正试测试我智能眼睛',
    '请告诉我前面有什么东西',
    '今天天气不错我准备出去走一走',
]


@pytest.mark.parametrize('text', CORPUS)
def test_intent_adapter_never_rewrites_recognized_text(text):
    mine = classify(text)
    project = project_classify(text)
    assert mine.raw == project.raw
    # 适配器可以改 kind，但识别文本必须与正式分类器对同一段原文的取值一致
    assert mine.text == project.text


@pytest.mark.parametrize('text', CORPUS)
def test_only_wake_word_and_punctuation_normalization_touches_the_text(text):
    # text 与 raw 的差异来源必须是且仅是 strip_wake_word（唤醒词/标点归一），
    # 不能是“把近音词替换成标准台词”这类内容改写。
    mine = classify(text)
    assert mine.text == strip_wake_word(text)


# 判定用的近音归一化只能发生在判定副本上：原文里的“神魔/停直/亭止”必须留着。
@pytest.mark.parametrize('text', ['柜子里是神魔？', '黄色设备是神魔', '这台机器是神魔',
                                  '停直', '亭止', '亭止。'])
def test_homophone_probe_never_writes_back_to_the_text(text):
    result = classify(text)
    assert result.raw == text.strip()
    assert result.text == strip_wake_word(text)


def test_installed_classifier_keeps_asr_text_for_downstream_question():
    intent_adapter.install()
    import aieyes.interaction.engine as engine

    result = engine.classify('柜子里是神魔？')
    assert result.kind == IntentKind.LOOK
    # engine._do_scene 用 intent.text 作为视觉提问；这里必须仍是 ASR 原文
    assert result.text == project_classify('柜子里是神魔？').text == '柜子里是神魔'


@pytest.mark.parametrize('text', ['柜子里是神魔？', '停直', '亭止', '停止！', '停停停',
                                  '小冰一加一等于几', '  前面有什么  ', 'OK 停止',
                                  '今好你好我正试测试我智能眼睛', '不要关或开麦克风'])
def test_engine_chain_receives_the_same_text(text):
    intent_adapter.install()
    import aieyes.interaction.engine as engine

    judged = engine.classify(text)
    assert judged.raw == text.strip()
    assert judged.text == project_classify(text).text


class SilentStream:
    def accept_waveform(self, rate, samples):
        pass

    def input_finished(self):
        pass


class EchoRecognizer:
    """把固定的 ASR 输出原样交回，用来隔离“识别”只验“文本通路”。"""

    def __init__(self, text):
        self.text = text

    def is_ready(self, stream):
        return False

    def get_result(self, stream):
        return self.text


# 真实 paraformer 对真实 wav 的输出（离线捕获，路径见 B_speech2.md）
ASR_OUTPUTS = [
    '小冰你好',
    '小冰一加一等于几',
    '小冰',
    '请告诉我前面有什么东西',
    '今天天气不错我准备出去走一走',
    '你好我正在测试我的智能眼镜',
    '今好你好我正试测试我智能眼睛',
    '小冰 hello',
]


@pytest.mark.parametrize('asr_text', ASR_OUTPUTS)
def test_asr_text_reaches_the_intent_layer_unchanged(asr_text):
    from asr_adapter import StreamingAsr

    asr = StreamingAsr()
    asr._stream = SilentStream()
    asr._recognizer = EchoRecognizer(asr_text)
    asr._t_start_ms, asr._t_end_ms = 100, 1000
    result = asr.finalize()
    # 1) ASR 适配器自身不加工文本（只有 0.3s 解码填充与时间戳刷新）
    assert result.text == asr_text

    # 2) 引擎侧拿到的 text 只允许唤醒词/标点归一，raw 必须是完整原文
    intent_adapter.install()
    import aieyes.interaction.engine as engine

    judged = engine.classify(result.text)
    assert judged.raw == asr_text
    assert judged.text == strip_wake_word(asr_text)


def test_recorded_short_stop_homophone_is_judged_without_rewriting():
    result = classify('停直')
    assert result.kind == IntentKind.STOP
    assert (result.text, result.raw) == ('停直', '停直')
    assert classify('请停直').kind == IntentKind.STOP
    assert control_feedback('停直') is None


@pytest.mark.parametrize('text', ['我停直了', '停直了吗', '别停直', '他一直停直着'])
def test_stop_homophone_requires_standalone_short_word(text):
    assert classify(text).kind != IntentKind.STOP


@pytest.mark.parametrize('text', ['停止', '停止播放', '请停一下', '暂停'])
def test_longer_stop_commands_keep_working(text):
    result = classify(text)
    assert result.kind == IntentKind.STOP
    assert result.text == project_classify(text).text


# 叠字“停停停”“停一停”是真机上连续喊停的口语形式，基础分类器给 STOP，
# 但过去的“非命令句降级”把第二个“停”也算成了非命令，降级成 CHAT（已复现并修复）。
@pytest.mark.parametrize('text', ['停停停', '停一停', '停停', '你停停停', '请停一停',
                                  '停停停！', '停一停吧'])
def test_repeated_stop_is_still_a_stop_command(text):
    result = classify(text)
    assert result.kind == IntentKind.STOP
    # 修复只动 kind，不动文本
    assert result.text == project_classify(text).text
    assert result.raw == text.strip()


# 成对反例：放开重复“停”的同时不能把非命令句一起放进来。
@pytest.mark.parametrize('text', ['停职是什么意思', '公交车停在哪里', '车停了', '停了吗',
                                  '我停直了', '停车场在这边', '他停了一下'])
def test_repeated_stop_widening_does_not_capture_non_commands(text):
    result = classify(text)
    assert result.kind != IntentKind.STOP
    assert result.text == project_classify(text).text
    assert result.raw == text.strip()
