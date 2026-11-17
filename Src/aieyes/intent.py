"""语音指令的轻量意图路由（Lead 维护）。

阶段 1–2 只需区分「聊天 / 停止 / 重说 / 唤醒词残留」；阶段 3–6 的视觉意图
（看、比较、读价格/日期、找物、合同控制）先在这里给出稳定分类，供状态机
逐步接入，不提前实现能力本身。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class IntentKind(str, Enum):
    CHAT = "chat"
    STOP = "stop"
    REPEAT = "repeat"
    BACKCHANNEL = "backchannel"
    CORRECT = "correct"
    RESUME = "resume"
    CLEAR = "clear"
    MUTE = "mute"
    UNMUTE = "unmute"
    LOOK = "look"
    COMPARE = "compare"
    PRICE = "price"
    DATE = "date"
    FIND = "find"
    CONTRACT = "contract"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Intent:
    kind: IntentKind
    text: str
    raw: str

    @property
    def is_control(self) -> bool:
        return self.kind in (
            IntentKind.STOP,
            IntentKind.REPEAT,
            IntentKind.MUTE,
            IntentKind.UNMUTE,
            IntentKind.CONTRACT,
        )


# 停止类：注意先长后短，“停”必须放在“停止/停下/停一下”之后
_STOP_WORDS = (
    "别说了",
    "别说话",
    "不用说了",
    "停一下",
    "停止",
    "停下",
    "打住",
    "闭嘴",
    "安静",
    "够了",
    "取消",
    "停",
)
#: 不含裸“停”的明确停止短语；裸“停”单独用句式判断，
#: 否则“停车场/停职/车停了/停一下车”都会被当成停止命令。
_STOP_PHRASES = tuple(word for word in _STOP_WORDS if word != "停")
_POLITE_PREFIX = r"(?:(?:请|你|先|现在|帮我|麻烦|马上|立刻|赶紧|能不能|可以|给我)\s*)*"
#: “停”必须处在命令句里：单独成句、后接标点或“一下/下来/播放/播报/说话/回答/语音”。
_BARE_STOP_RE = re.compile(
    r"^" + _POLITE_PREFIX
    + r"(?:暂停|停)(?:$|[，,。！？?!]|一下|下来|一会儿|会儿|住|播放|播报|说话|回答|语音)"
)
#: 真机连续喊停“停停停/停一停”；只收重复的停，不放开“停职/停车场/停了吗”。
_REPEAT_STOP_RE = re.compile(r"^" + _POLITE_PREFIX + r"停+(?:一停+)?[吧了。！!？?，,\s]*$")
# “不是让你停 / 别停 / 不要停 / 我没有让你停”属于否定或纠正，不能当停止
_NEG_STOP_RE = re.compile(
    r"(?:不是|并没有|没有|没|别|不要|不用|甭)\s*"
    r"(?:再|让你|叫你|让我|让|叫|我让|我叫)?\s*"
    r"(?:停止|停下|停一下|停|闭嘴|安静|取消|打住)"
)
# 短控制词“停止”被流式 ASR 识成“停直/亭止”时，只按整句白名单判意；
# 不改写识别原文，也不放开“停直了吗/别停直”这类陈述或否定。
_CONTROL_HOMOPHONES = ("停直", "亭止")
_STOP_HOMOPHONE_RE = re.compile(
    r"^(?:请|你|先|现在|帮我|麻烦|马上|立刻|赶紧|能不能|可以|给我)*\s*"
    r"(?:停直|亭止)[吧了。！!？?，,\s]*$"
)
_MUTE_WORDS = ("关麦", "关闭麦克风", "关掉麦克风", "静音", "别听了", "不用听了")
_UNMUTE_WORDS = ("开麦", "打开麦克风", "取消静音", "可以听了", "继续听")
_CORRECT_WORDS = ("不是这个", "不是那个", "我说的是", "搞错了", "弄错了", "重新来", "换个")
# “对不对/是不是”是疑问尾缀，不是纠正；“不对”单独出现才算纠正。
_CORRECT_NOT_RE = re.compile(r"(?<!对)不对(?!对)")
_REPEAT_WORDS = ("再说一遍", "重复一遍", "再说一次", "没听清", "什么")
# 恢复被打断的图片任务（“继续/接着”；“继续听”已先被 UNMUTE 命中）
# ASR 常把“继续”识别成“继需/接续”，这里按句首前缀判定，避免又开一个新比较任务
_RESUME_WORDS = ("继续", "接着", "恢复", "继续比")
_RESUME_PREFIX_RE = re.compile(r"^(?:继续|接着|接续|继需|继)")
# 清除本轮对话/图片资料（阶段 5 资料权限；执行前还要过声纹身份门）
_CLEAR_WORDS = (
    "清空对话",
    "清空记忆",
    "清空资料",
    "清除资料",
    "清除记录",
    "清除对话",
    "清除缓存",
    "删掉资料",
    "删除资料",
    "抹掉资料",
    "忘掉之前",
    "忘掉刚才",
    "忘记之前",
    "忘记刚才",
)
_LOOK_WORDS = (
    "看看",
    "看一下",
    "看下",
    "这是什么",
    "前面是什么",
    "拍一下",
    "拍照",
    "取图",
    "识别",
    # 真人录音里的自然问法（如“请告诉我前面有什么东西”），不能掉成聊天。
    "前面有什么",
    "前面有啥",
    "眼前有什么",
    "眼前是什么",
    "周围有什么",
    "周围是什么",
    "身边有什么",
    "看到什么",
    "看见什么",
    "画面里有什么",
    "视野里有什么",
)
# “XX在哪里”默认是找物，但城市/车站/停车场这类不是画面内目标；
# 只有带明确画面语境（画面/眼前/找一下）时才继续走 FIND。
_NON_VISUAL_FIND_RE = re.compile(
    r"首都|省会|国家|城市|大洲|大陆|出生地|停车场|公交站|公交车停"
)
_FIND_CONTEXT_RE = re.compile(r"画面|图中|眼前|面前|前面|找一下")
_COMPARE_WORDS = (
    "比较",
    "对比",
    "那个更",
    "那个贵",
    "那个便宜",
    "那个好",
    "谁更",
    "有什么区别",
    "差异",
)
# “哪个/拿个/那个”只有后接比较形容词时才是双图比较；
# “键盘在画面哪个地方”是定位问题，不能因为出现“哪个”就开双图任务。
_COMPARE_WHICH_RE = re.compile(
    r"(?:哪个|哪一个|哪件|哪瓶|哪样|拿个|那个)"
    r"(?:更|最|比较|比|要|会)?"
    r"(?:贵|便宜|好|划算|重|轻|大|小|新|旧|远|近|多|少|高|低|长|短|强|弱|合适|值得|实惠|耐用)"
)
# “这两个哪个”即使形容词被 ASR 吞掉（实测“这两个哪个贵”→“这两个哪个”）也仍是
# 二选一比较；但“画面中的哪个地方/位置”是定位提问，不能算比较。
_COMPARE_PAIR_RE = re.compile(
    r"(?:这|那)?\s*(?:两|二|俩)\s*(?:个|件|样|种|瓶|盒|包|只|台|部|张|袋)?"
    r"[^，。！？?!]{0,4}?(?:哪个|哪一个|哪件|哪样|拿个)"
    r"(?!地方|位置|区域|角落)"
)
# 画面/图片内的位置提问：走找物定位，不进入聊天或双图比较
_LOCATION_RE = re.compile(
    r"(?:在|位于)?(?:画面|图里|图中|屏幕|照片|视野|眼前|前面)"
    r"(?:里|中|上)?(?:的)?(?:哪里|哪儿|哪个地方|哪个位置|什么位置|何处)"
    r"|(?:在哪里|在哪儿|在哪个地方|在哪个位置|在什么位置)"
)
_PRICE_WORDS = ("多少钱", "价格", "价钱", "几块", "售价", "多少", "标价", "单价", "打折", "优惠")
_DATE_WORDS = ("保质期", "生产日期", "到期", "有效期", "日期")
_FIND_WORDS = ("找", "在哪", "放在哪", "在哪里", "哪个地方", "哪个位置", "什么位置")
_CONTRACT_WORDS = ("紧急", "紧急停止", "抢占")

# 明确纠正信号：出现时先判 CORRECT，不能被“左边/右边/哪个”等比较词抢走
# （否则“不是这个，我说的是右边那件”会新开一个双图比较任务）
_CORRECTION_RE = re.compile(
    r"(?:不是|并不是)(?:这个|那个|这件|那件|这里|那里|左边|右边)"
    r"|我说的是|我说错了|我说错|搞错了|弄错了"
    r"|不是[^，。？！]{0,8}?(?:是|而是)"
)

_BACKCHANNEL_RE = re.compile(
    r"^(?:嗯+|啊+|哦+|噢+|对+|好+|行+|可以|是+|哎+|唉+|嘿+)[啊呀呢的了]?$"
)

_WAKE_RE = re.compile(r"^(小冰|小兵|小斌|小斌)[，,。.\s]*")
_PUNCT_RE = re.compile(r"^[\s，。！？、,.!?;；:：\"'“”‘’()（）\[\]【】]+|[\s，。！？、,.!?;；:：\"'“”‘’()（）\[\]【】]+$")


def strip_wake_word(text: str) -> str:
    """去掉句首的唤醒词与空白，返回可用于意图判断的文本。"""
    cleaned = _PUNCT_RE.sub("", text or "")
    cleaned = _WAKE_RE.sub("", cleaned)
    return _PUNCT_RE.sub("", cleaned)


def _contains(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def classify(text: str) -> Intent:
    raw = (text or "").strip()
    cleaned = strip_wake_word(raw)
    lowered = cleaned.lower()
    if not cleaned:
        return Intent(IntentKind.UNKNOWN, cleaned, raw)
    # 控制类优先于聊天（“别说了”不应被当成聊天内容）；“取消静音”要先于“静音”判
    if _contains(cleaned, _UNMUTE_WORDS):
        return Intent(IntentKind.UNMUTE, cleaned, raw)
    if _contains(cleaned, _MUTE_WORDS):
        return Intent(IntentKind.MUTE, cleaned, raw)
    if _RESUME_PREFIX_RE.match(cleaned) or _contains(cleaned, _RESUME_WORDS):
        return Intent(IntentKind.RESUME, cleaned, raw)
    if (lowered in ("stop", "cancel")
            or _contains(cleaned, _STOP_PHRASES)
            or _BARE_STOP_RE.match(cleaned)
            or _REPEAT_STOP_RE.match(cleaned)
            or _STOP_HOMOPHONE_RE.match(cleaned)):
        if not _NEG_STOP_RE.search(cleaned):
            return Intent(IntentKind.STOP, cleaned, raw)
    if _contains(cleaned, _CLEAR_WORDS):
        return Intent(IntentKind.CLEAR, cleaned, raw)
    if _contains(cleaned, _CONTRACT_WORDS):
        return Intent(IntentKind.CONTRACT, cleaned, raw)
    if _CORRECTION_RE.search(cleaned):
        return Intent(IntentKind.CORRECT, cleaned, raw)
    if (_LOCATION_RE.search(cleaned)
            and not (_NON_VISUAL_FIND_RE.search(cleaned)
                     and not _FIND_CONTEXT_RE.search(cleaned))):
        return Intent(IntentKind.FIND, cleaned, raw)
    if (_contains(cleaned, _COMPARE_WORDS)
            or _COMPARE_WHICH_RE.search(cleaned)
            or _COMPARE_PAIR_RE.search(cleaned)):
        return Intent(IntentKind.COMPARE, cleaned, raw)
    if _contains(cleaned, _PRICE_WORDS):
        return Intent(IntentKind.PRICE, cleaned, raw)
    if _contains(cleaned, _DATE_WORDS):
        return Intent(IntentKind.DATE, cleaned, raw)
    if (_contains(cleaned, _FIND_WORDS)
            and not (_NON_VISUAL_FIND_RE.search(cleaned)
                     and not _FIND_CONTEXT_RE.search(cleaned))):
        return Intent(IntentKind.FIND, cleaned, raw)
    if _contains(cleaned, _LOOK_WORDS):
        return Intent(IntentKind.LOOK, cleaned, raw)
    if cleaned in _REPEAT_WORDS or _contains(cleaned, ("再说一遍", "重复一遍", "再说一次")):
        return Intent(IntentKind.REPEAT, cleaned, raw)
    if _contains(cleaned, _CORRECT_WORDS) or _CORRECT_NOT_RE.search(cleaned):
        return Intent(IntentKind.CORRECT, cleaned, raw)
    if _BACKCHANNEL_RE.match(cleaned):
        return Intent(IntentKind.BACKCHANNEL, cleaned, raw)
    return Intent(IntentKind.CHAT, cleaned, raw)
