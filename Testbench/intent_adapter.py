"""补足实际录音中常见的视觉问法，保留项目控制/价格等意图优先级。"""
import re
from aieyes.intent import classify as project_classify, Intent, IntentKind

_VISUAL = re.compile(r'(?:前面|前方|眼前|原前|前眼|面前|周围|画面里|画面中|视野里|健钱|见钱|前钱).{0,8}(?:有什么|有啥|是什么|什么东西|什么物体|什么情况)|(?:这个|那个|它).{0,4}(?:是什么|有什么)|(?:看到|看见)(?:了)?(?:什么|啥)|(?:设备|机器|柜子|装置|标牌|物体).{0,3}(?:是什么|做什么|干什么)|(?:什么)(?:设备|机器|柜子|装置)')

_READ = re.compile(r'(?:读|念)(?:一下|一遍|出来|给我|上面|这个|这段|这些|文字)|(?:上面|牌子|标签).{0,8}(?:写|字)|什么字')
_ATTRIBUTE = re.compile(r'(?:画面|图中|这个|这些|这把|这只|这台).{0,12}(?:颜色|什么色|几个|多少[个只件辆人])')

_CONTROL_KINDS = {IntentKind.STOP, IntentKind.MUTE, IntentKind.UNMUTE,
                  IntentKind.RESUME, IntentKind.CLEAR, IntentKind.CONTRACT}
_POLITE = r'(?:(?:请|你|先|现在|帮我|麻烦|马上|立刻|赶紧|能不能|可以|给我)\s*)*'
_BARE_STOP = re.compile(r'^' + _POLITE + r'(?:暂停|停)(?:$|[，,。！？?!]|一下|下来|一会儿|会儿|住|播放|播报|说话|回答|语音)')
# 真机上“停停停”“停一停”是连续喊停的口语形式。基础分类器靠子串“停”给 STOP，
# 但 _BARE_STOP 的允许后缀只有 一下/下来/播放…，第二个“停”落不进任何分支，会被
# 这里降级成 CHAT（复现：classify('停停停') 返回 CHAT）。只收“重复的停”，
# 不放开“停职/停车场/停了吗/车停了”这类非命令句。
_REPEAT_STOP = re.compile(r'^' + _POLITE + r'停+(?:一停+)?[吧了。！!？?，,\s]*$')
_RESUME_COMMAND = re.compile(r'^' + _POLITE + r'(?:继续|接着|接续|继需|继|恢复)(?:[吧好吗呢了\s]*$|(?:说|讲|回答|播报|播放|比|看|读|识别|找|刚才|之前|上次|当前|这个任务).*)')
_METHOD_QUESTION = re.compile(r'怎么|怎样|如何')
_EXPLANATION = re.compile(r'什么意思|是什么意思|是啥意思|含义|解释|怎么|怎样|如何|为什么|有何区别|有什么作用|有什么用|有啥用|有什么用途|什么用途|干什么用|干啥用|干嘛用|是不是|(?:静音|关麦|开麦).*(?:了吗|了没|了没有)')
_DIRECT_REQUEST = re.compile(r'^' + _POLITE + r'(?:关闭麦克风|关掉麦克风|打开麦克风|取消静音|关麦|开麦|静音|停止|停下|停一下|暂停|别说了|清空|清除|删掉|删除|忘掉|忘记|继续|接着|恢复)')
# 裸动词否定句（别关麦克风/别开麦克风）原本靠子串"关麦/开麦"命中降级为聊天，
# 但匹配在"关麦"处截断、句尾残留"克风"，control_feedback 的句尾为空判定失败，
# 拿不到"不更改设置"的程序化反馈。增加"关麦克风|开麦克风"备选让否定完整吞掉宾语。
# 否定句里并列/连写的两个控制动词（"不要关或开麦克风""别开关麦克风"）原先只被
# 子串命中其中一个动词，降级失败后直接执行 mute/unmute，等于反向动作。这里把并列
# 动词短语整体吞进否定匹配，才能命中 _mentioned_control 的"只提及不执行"。
_NEGATED_PAIR = (r'(?:关闭|关掉|打开|取消|关|开)(?:麦克风|麦|静音)?\s*(?:或|和|与|、|/)\s*'
                 r'(?:关闭|关掉|打开|取消|关|开)(?:麦克风|麦|静音)?'
                 r'|开关(?:麦克风|麦)')
# 否定优先级（清单第 7 项，C 组复核补充）：否定词 + 完整动作短语必须整体匹配，
# 否则"不要打开静音""别关掉静音"会由正向匹配落到 MUTE（方向相反，真的执行静音），
# "别把麦克风关了""不要关声音"则否定了却拿不到"不更改设置"的反馈。
_NEGATED_OBJECT = r'(?:麦克风|声音|音量|静音)'
_NEGATED_VERB = r'(?:关闭掉|关闭|关掉|关上|关了|关|打开|开启|开开|开了|开|取消|静音)'
_NEGATED_PHRASE = (r'|'.join((
    r'(?:打开|开启|关掉|关上|关闭|取消|关|开|静音)' + _NEGATED_OBJECT,
    r'把' + _NEGATED_OBJECT + r'(?:给|都|也)?\s*' + _NEGATED_VERB + r'(?:了|掉|上)?',
    _NEGATED_OBJECT + r'(?:给|都|也)?\s*' + _NEGATED_VERB + r'(?:了|掉|上)?',
)))
_NEGATED_ACTION = re.compile(r'(?:不要|别|不用|不必|不想|不需要|不希望你?|(?:并)?不是(?:让你|要你?|叫你)|(?:并)?没(?:有)?(?:让|叫)你)(?:再|去|再去|让麦克风)?\s*(?:' + _NEGATED_PHRASE + '|' + _NEGATED_PAIR + '|关闭麦克风|关掉麦克风|打开麦克风|取消静音|关麦克风|开麦克风|关麦|开麦|静音|停止|停下|停一下|停|别说了|清空|清除|删掉|删除|抹掉|忘掉|忘记|继续|接着|恢复|取消|打住)')

# 已复现的同音错词只用于“判定副本”，绝不回写识别文本（清单第 1 项红线）。
_DEVICE_HOMOPHONE = re.compile(r'(?:设备|机器|装置|柜子).{0,3}是神魔[？?。]?$')
_DEVICE_HOMOPHONE_EXCLUDE = re.compile(r'游戏|小说|故事|神话|动漫|角色')

# 短控制词的近音误识别（清单第 6 项：“停止”被识别成“停直”）。只允许下面显式
# 列出的词条参与判定，且只在整句就是该短词（可带礼貌前缀）时判为停止；判定结果
# 只改 kind，text/raw 仍是 ASR 原文，不能据此宣称短控制词识别问题已解决。
_CONTROL_HOMOPHONES = ('停直', '亭止')
_HOMOPHONE_STOP = re.compile(
    r'^' + _POLITE + r'(?:' + '|'.join(map(re.escape, _CONTROL_HOMOPHONES)) + r')'
    r'[吧了。！!？?，,\s]*$'
)


def _homophone_probe(text):
    """返回只供判定使用的副本；对外的识别文本保持 ASR 原文。"""
    text = text or ''
    if _DEVICE_HOMOPHONE.search(text) and not _DEVICE_HOMOPHONE_EXCLUDE.search(text):
        return re.sub(r'是神魔([？?。]?)$', r'是什么\1', text)
    return text


def _mentioned_control(result):
    """A mentioned, negated or explained control word is not an action."""
    if result.kind not in _CONTROL_KINDS:
        return False
    text = result.text
    if _NEGATED_ACTION.search(text):
        return True
    first_clause = re.split(r'[，,。！？?!；;]', text, maxsplit=1)[0]
    if _EXPLANATION.search(text) and (_EXPLANATION.search(first_clause) or not _DIRECT_REQUEST.match(first_clause)):
        return True
    if result.kind == IntentKind.STOP:
        from aieyes.intent import _STOP_WORDS
        if (not any(word in text for word in _STOP_WORDS if word != '停')
                and not _BARE_STOP.match(text)
                and not _REPEAT_STOP.match(text)):
            return True
    if result.kind == IntentKind.RESUME and not _RESUME_COMMAND.match(first_clause):
        return True
    return False


def control_feedback(text):
    """Give factual help/no-change feedback without inventing device state."""
    if (_METHOD_QUESTION.search(text)
            and re.search(r'(?:清空|清除|删除)(?:这轮|本轮|当前)?(?:的)?(?:资料|对话|记忆)', text)
            and not re.search(r'[，,；;。].*\S', text)
            and not _NEGATED_ACTION.search(text)):
        return '直接对我说“清空资料”就可以了。'
    if _EXPLANATION.search(text):
        return None
    negated = _NEGATED_ACTION.search(text)
    if negated and text[negated.end():].strip('吧了。！!？? \t') == '':
        if re.search(r'麦克风|关麦|开麦', text):
            return '好，我不更改麦克风设置。'
        if re.search(r'静音|声音|音量', text):
            return '好，我不更改静音设置。'
    if re.fullmatch(r'(?:请|你)?(?:不要|别|不用|不必)(?:再)?取消[^，,。！？?!；;]{1,20}[。！!？?]?', text):
        return '好，我不取消。'
    return None

def classify(text):
    probe=_homophone_probe(text)
    result=_judged(probe)
    if probe!=(text or ''):
        # 同音错词只在判定副本里归一化；对外的 text/raw 必须回到 ASR 原文。
        original=project_classify(text)
        result=Intent(result.kind,original.text,original.raw)
    if result.kind in (IntentKind.CHAT,IntentKind.UNKNOWN) and _HOMOPHONE_STOP.match(result.text):
        result=Intent(IntentKind.STOP,result.text,result.raw)
    return result

def _judged(text):
    result = project_classify(text)
    if _mentioned_control(result):
        result = Intent(IntentKind.CHAT, result.text, result.raw)
    if result.kind==IntentKind.FIND and re.search(r'首都|省会|国家|城市|大洲|大陆|出生地',result.text) and not re.search(r'画面|图中|眼前|面前|前面|找一下',result.text):
        return Intent(IntentKind.CHAT,result.text,result.raw)
    if result.kind in (IntentKind.CHAT,IntentKind.PRICE):
        from calculator import answer_for
        if answer_for(result.text) is not None:
            return Intent(IntentKind.CHAT,result.text,result.raw)
        direct_color=re.search(r'.+(?:是什么颜色|什么颜色|什么色)(?:的|呢)?[？?。]?$',result.text)
        direct_count=re.search(r'^(?:请问)?(?:有|能看到|看见)(?:几个|多少[个只件辆人])',result.text)
        if _READ.search(result.text) or _ATTRIBUTE.search(result.text) or direct_color or direct_count:
            return Intent(IntentKind.LOOK,result.text,result.raw)
    if result.kind == IntentKind.CHAT and _VISUAL.search(result.text):
        return Intent(IntentKind.LOOK, result.text, result.raw)
    return result

def install():
    import aieyes.interaction.engine as engine
    engine.classify = classify
