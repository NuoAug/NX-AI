"""用户引导控制意图 → ``control.intent``（合同 §ControlIntent）。

合同（``Docs/MCPT/主控与交互接口合同.md``）规定：

* 交互发送 ``control.intent``，主控返回 ``control.result``；
* v0.1 允许的动作只有 ``start_guidance`` / ``pause_guidance`` /
  ``resume_guidance`` / ``end_guidance`` / ``repeat_guidance``；
* 这些是**意图请求**，主控按任务状态判断能否执行；
* 只有确认完成且模式匹配时才能说"已经开始/暂停"；
* 超时先查同一请求状态，不新建请求盲目重复执行；
* 暂停聊天/停止普通播报由交互侧自己处理，不必一律暂停行走任务；
  "停一下"存在歧义时先澄清；
* 交互侧不能发出移动/通行/空间指令。

本模块只做两件事：把**明确**的话语映射成允许动作，以及按 ``control.result``
如实措辞。它不实现状态机，也不决定是否真的开始引导——那是主控的事。

动作映射要求整句里出现"引导"，否则像"再说一遍"这类本地重播请求会被截走：
``repeat`` 是交互自己的重播，``repeat_guidance`` 才是请主控重发引导。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .schema import CONTROL_ACTIONS, Message

__all__ = [
    "GUIDANCE_ACTIONS",
    "GUIDANCE_PHRASES",
    "GuidanceOutcome",
    "build_control_intent",
    "classify_guidance_action",
    "send_guidance_intent",
]

#: 合同许可的引导动作（与 schema.CONTROL_ACTIONS 同一集合，测试里断言不越界）
GUIDANCE_ACTIONS = CONTROL_ACTIONS

#: 明确话语 → 动作。顺序即优先级：先具体的"重复引导"，再开始/暂停/继续/结束。
GUIDANCE_PHRASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "repeat_guidance",
        ("重复引导", "重复一下引导", "再播一遍引导", "再说一遍引导", "再报一遍引导", "再念一遍引导", "重复刚才的引导"),
    ),
    ("pause_guidance", ("暂停引导", "先暂停引导", "把引导暂停一下")),
    ("resume_guidance", ("继续引导", "恢复引导", "接着引导")),
    ("end_guidance", ("结束引导", "停止引导", "取消引导", "不用引导了", "别再引导了")),
    ("start_guidance", ("开始引导", "开始语音引导")),
)

#: 句子里必须出现"引导"才认为这是引导控制（否则"再说一遍""停一下"等本地动作会被截走）
GUIDANCE_TOKEN = "引导"

#: 否定/纠正：包含这些结构时不提交控制意图（合同：否定句不得因含关键词而执行）
_NEGATED_VERBS = ("开始", "暂停", "继续", "结束", "停止", "取消", "重复", "播", "报", "念")
_NEGATION_PREFIXES = ("不是", "并不是", "我不是", "没让你", "没有让你", "别", "不要", "不用", "甭")

#: 每个动作在"已完成"时可以说的话，以及未完成时的中性说明
_DONE_SPEECH = {
    "start_guidance": "好的，已开始引导。",
    "pause_guidance": "好的，已暂停引导。",
    "resume_guidance": "好的，已接着引导。",
    "end_guidance": "好的，已结束引导。",
    "repeat_guidance": "好的，已请主控重播这条引导。",
}


def _is_negated(text: str) -> bool:
    for prefix in _NEGATION_PREFIXES:
        index = text.find(prefix)
        while index >= 0:
            tail = text[index + len(prefix) : index + len(prefix) + 8]
            if any(verb in tail for verb in _NEGATED_VERBS):
                return True
            index = text.find(prefix, index + 1)
    return False


def classify_guidance_action(text: str) -> str | None:
    """把明确话语映射成合同动作；不确定就返回 ``None``（交给本地流程/澄清）。

    * 不含"引导"→ ``None``：避免抢走"再说一遍""停一下""继续"这些本地动作；
    * 含否定/纠正结构 → ``None``：合同明确否定句不得执行；
    * 命中多个动作时按 :data:`GUIDANCE_PHRASES` 顺序取第一个（重复优先）。
    """
    cleaned = (text or "").strip()
    if not cleaned or GUIDANCE_TOKEN not in cleaned:
        return None
    if _is_negated(cleaned):
        return None
    for action, phrases in GUIDANCE_PHRASES:
        if any(phrase in cleaned for phrase in phrases):
            return action
    return None


def build_control_intent(
    action: str,
    *,
    utterance_id: str,
    final_transcript: str,
    producer: Mapping[str, str],
    target_ref: str | Mapping[str, Any] | None = None,
    user_confirmed: bool = False,
    **kwargs: Any,
) -> Message:
    """构造并校验一条 ``control.intent``（越界动作在这里就被 schema 拒绝）。"""
    from .schema import control_intent

    if action not in CONTROL_ACTIONS:
        raise ValueError(f"引导动作不在合同 v0.1 允许范围: {action!r}")
    return control_intent(
        action=action,
        utterance_id=utterance_id,
        final_transcript=final_transcript,
        target_ref=target_ref,
        user_confirmed=user_confirmed,
        producer=producer,
        **kwargs,
    )


@dataclass(frozen=True, slots=True)
class GuidanceOutcome:
    """一次``control.intent``的结果，``speech`` 已经按合同口径分级。"""

    action: str
    accepted: bool
    operation_state: str
    reason_code: str | None
    speech: str
    result: Message | None = None

    @property
    def completed(self) -> bool:
        return self.accepted and self.operation_state == "completed"


def _speech_for(action: str, state: str, reason_code: str | None) -> str:
    if state == "completed":
        return _DONE_SPEECH.get(action, "好的，已按你说的处理引导。")
    if state == "pending":
        return "主控已经收到，还在处理，暂时不能确认是否完成。"
    if state == "rejected":
        why = f"（{reason_code}）" if reason_code else ""
        return f"主控没有执行这次引导操作{why}，我按原来的状态继续。"
    if state == "unavailable":
        why = f"（{reason_code}）" if reason_code else ""
        return f"主控返回了异常{why}，我还不能确认引导操作是否完成。"
    # unknown：超时等拿不到结论的情况，不能猜测执行结果
    return "我没能确认引导操作是否已经执行，先不重复发送，请先确认主控当前的引导状态。"


async def send_guidance_intent(
    contract: Any,
    *,
    action: str,
    utterance_id: str,
    final_transcript: str,
    target_ref: str | Mapping[str, Any] | None = None,
    user_confirmed: bool = False,
) -> GuidanceOutcome:
    """发送 ``control.intent`` 并把 ``control.result`` 解读成可播报的结论。

    ``contract`` 只需暴露 ``send_control_intent(**kwargs)``（
    :class:`aieyes.contract.handlers.InteractionContract` 与
    :class:`aieyes.interaction.contract_link.ContractLink` 都满足）。

    合同约束在这里落地：

    * ``accepted=false`` / ``operation_state=rejected`` 不得说成已执行；
    * ``pending`` 只能说"已收到、在处理"；
    * 超时/发送失败 → ``operation_state="unknown"``，措辞明确"没确认、不重复发送"，
      不自动重发（合同要求先查同一请求状态，v0.1 未定义查询接口）。
    """
    from .schema import is_completed_ok

    if action not in CONTROL_ACTIONS:
        raise ValueError(f"引导动作不在合同 v0.1 允许范围: {action!r}")
    if not utterance_id or not final_transcript:
        raise ValueError("control.intent 需要 utterance_id 与最终话语 final_transcript")

    try:
        result = await contract.send_control_intent(
            action=action,
            utterance_id=utterance_id,
            final_transcript=final_transcript,
            target_ref=target_ref,
            user_confirmed=user_confirmed,
        )
    except Exception as exc:  # noqa: BLE001 - 通道失败按"未确认"报，不猜结果
        code = getattr(exc, "code", None) or type(exc).__name__
        state = "unknown"
        return GuidanceOutcome(action, False, state, str(code), _speech_for(action, state, str(code)), None)

    data = result.data if isinstance(getattr(result, "data", None), Mapping) else {}
    state = str(data.get("operation_state") or "")
    reason_code = data.get("reason_code")
    reason_code = None if reason_code is None else str(reason_code)

    if getattr(result, "status", None) != "ok":
        err = getattr(result, "error", None) or {}
        code = str(err.get("code") or "DEPENDENCY_UNAVAILABLE") if isinstance(err, Mapping) else "DEPENDENCY_UNAVAILABLE"
        return GuidanceOutcome(action, False, "unavailable", code, _speech_for(action, "unavailable", code), result)

    completed = is_completed_ok(result)
    accepted = bool(data.get("accepted"))
    if not completed and state not in ("pending", "rejected"):
        # 合同里 operation_state 只有三态；出现未知值时按"未确认完成"处理
        return GuidanceOutcome(action, accepted, "unknown", reason_code, _speech_for(action, "unknown", reason_code), result)
    final_state = "completed" if completed else state
    return GuidanceOutcome(action, accepted, final_state, reason_code, _speech_for(action, final_state, reason_code), result)
