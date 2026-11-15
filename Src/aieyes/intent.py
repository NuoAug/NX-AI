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
# “不是让你停 / 别停 / 不要停 / 我没有让你停”属于否定或纠正，不能当停止
_NEG_STOP_RE = re.compile(
    r"(?:不是|并没有|没有|没|别|不要|不用|甭)\s*"
    r"(?:再|让你|叫你|让我|让|叫|我让|我叫)?\s*"
    r"(?:停止|停下|停一下|停|闭嘴|安静|取消|打住)"
)
_MUTE_WORDS = ("关麦", "关闭麦克风", "关掉麦克风", "静音", "别听了", "不用听了")
_UNMUTE_WORDS = ("开麦", "打开麦克风", "取消静音", "可以听了", "继续听")
_CORRECT_WORDS = ("不是这个", "不是那个", "不对", "我说的是", "搞错了", "弄错了", "重新来", "换个")
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
_LOOK_WORDS = ("看看", "看一下", "看下", "这是什么", "前面是什么", "拍一下", "拍照", "取图", "识别")
_COMPARE_WORDS = (
    "比较",
    "对比",
    "哪个",
    "哪一个",
    "哪件",
    "哪瓶",
    "哪样",
    "拿个",
    "那个更",
    "那个贵",
    "那个便宜",
    "那个好",
    "谁更",
    "有什么区别",
    "差异",
    "左边",
    "右边",
    "翻面",
)
_PRICE_WORDS = ("多少钱", "价格", "价钱", "几块", "售价", "多少", "标价", "单价", "打折", "优惠")
_DATE_WORDS = ("保质期", "生产日期", "到期", "有效期", "日期")
_FIND_WORDS = ("找", "在哪", "放在哪", "在哪里")
_CONTRACT_WORDS = ("紧急", "紧急停止", "抢占")

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
    if lowered in ("stop", "cancel") or _contains(cleaned, _STOP_WORDS):
        if not _NEG_STOP_RE.search(cleaned):
            return Intent(IntentKind.STOP, cleaned, raw)
    if _contains(cleaned, _CLEAR_WORDS):
        return Intent(IntentKind.CLEAR, cleaned, raw)
    if _contains(cleaned, _CONTRACT_WORDS):
        return Intent(IntentKind.CONTRACT, cleaned, raw)
    if _contains(cleaned, _COMPARE_WORDS):
        return Intent(IntentKind.COMPARE, cleaned, raw)
    if _contains(cleaned, _PRICE_WORDS):
        return Intent(IntentKind.PRICE, cleaned, raw)
    if _contains(cleaned, _DATE_WORDS):
        return Intent(IntentKind.DATE, cleaned, raw)
    if _contains(cleaned, _FIND_WORDS):
        return Intent(IntentKind.FIND, cleaned, raw)
    if _contains(cleaned, _LOOK_WORDS):
        return Intent(IntentKind.LOOK, cleaned, raw)
    if cleaned in _REPEAT_WORDS or _contains(cleaned, ("再说一遍", "重复一遍", "再说一次")):
        return Intent(IntentKind.REPEAT, cleaned, raw)
    if _contains(cleaned, _CORRECT_WORDS):
        return Intent(IntentKind.CORRECT, cleaned, raw)
    if _BACKCHANNEL_RE.match(cleaned):
        return Intent(IntentKind.BACKCHANNEL, cleaned, raw)
    return Intent(IntentKind.CHAT, cleaned, raw)
