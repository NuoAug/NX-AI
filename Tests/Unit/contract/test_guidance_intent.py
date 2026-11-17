"""引导控制意图 → control.intent 的软件回归（合同 §ControlIntent）。

不接触主控或任何设备：用 FakeContract 扮演 ``send_control_intent``，
只验证"允许动作 + 如实措辞"这两条合同要求。
"""

from __future__ import annotations

import asyncio
import inspect

import pytest

from aieyes.contract import schema
from aieyes.contract.guidance_intent import (
    GUIDANCE_ACTIONS,
    GUIDANCE_PHRASES,
    build_control_intent,
    classify_guidance_action,
    send_guidance_intent,
)

PRODUCER = schema.producer_of("interaction", "test-voice-1")


class FakeContract:
    """只实现 send_control_intent 的替身；记录调用参数，返回构造的 control.result。"""

    def __init__(self, *, data=None, status="ok", error=None, raises=None):
        self.data = data if data is not None else _result_data("completed")
        self.status = status
        self.error = error
        self.raises = raises
        self.calls: list[dict] = []

    async def send_control_intent(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        request = schema.control_intent(
            action=kwargs["action"],
            utterance_id=kwargs["utterance_id"],
            final_transcript=kwargs["final_transcript"],
            target_ref=kwargs.get("target_ref"),
            user_confirmed=kwargs.get("user_confirmed", False),
            producer=PRODUCER,
        )
        return schema.control_result(
            request, producer=PRODUCER, data=self.data if self.status == "ok" else None,
            status=self.status, error=self.error,
        )


def _result_data(state: str, *, accepted: bool = True, reason: str | None = None) -> dict:
    return {
        "accepted": accepted,
        "operation_state": state,
        "actual_mode": "guidance",
        "reason_code": reason,
        "state_id": "state-1",
    }


class FakeTimeout(Exception):
    code = "DEADLINE_EXCEEDED"


def test_phrase_table_stays_inside_the_contract() -> None:
    assert set(GUIDANCE_ACTIONS) == set(schema.CONTROL_ACTIONS)
    mapped = [action for action, _phrases in GUIDANCE_PHRASES]
    assert set(mapped) == set(GUIDANCE_ACTIONS), "合同允许的动作都要有明确话语入口"
    for _action, phrases in GUIDANCE_PHRASES:
        assert phrases, "每个动作至少一条明确话语"
        for phrase in phrases:
            assert "引导" in phrase, f"{phrase!r} 不含“引导”，可能抢走本地动作"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("重复引导", "repeat_guidance"),
        ("再播一遍引导", "repeat_guidance"),
        ("再说一遍引导", "repeat_guidance"),
        ("开始引导", "start_guidance"),
        ("暂停引导", "pause_guidance"),
        ("继续引导", "resume_guidance"),
        ("结束引导", "end_guidance"),
        ("不用引导了", "end_guidance"),
    ],
)
def test_unambiguous_phrases_map_to_a_contract_action(text: str, expected: str) -> None:
    assert classify_guidance_action(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "再说一遍",       # 本地重播
        "重复一遍",       # 本地重播
        "停一下",         # 合同要求歧义时先澄清
        "暂停",           # 停普通播报，不必暂停行走任务
        "继续",           # 本地恢复看图任务
        "开始导航",       # 不是引导动作，也不能被当成移动指令
        "向左走",         # 禁止自造移动指令
        "不是让你结束引导",
        "不要开始引导",
        "别暂停引导",
        "我没让你继续引导",
        "昨天说的引导路线是什么",  # 只是聊天里提到“引导”
    ],
)
def test_local_commands_negations_and_chat_are_not_guidance_intents(text: str) -> None:
    assert classify_guidance_action(text) is None


def test_build_control_intent_validates_against_schema() -> None:
    msg = build_control_intent(
        "repeat_guidance",
        utterance_id="utt-1",
        final_transcript="重复引导",
        producer=PRODUCER,
    )
    roundtrip = schema.parse_message(msg.to_dict())
    assert roundtrip.type == "control.intent"
    assert roundtrip.data["action"] == "repeat_guidance"
    assert roundtrip.data["final_transcript"] == "重复引导"


def test_build_control_intent_rejects_actions_outside_v01() -> None:
    for action in ("emergency_stop", "walk_forward", "turn_left", "开始"):
        with pytest.raises(ValueError):
            build_control_intent(
                action, utterance_id="utt-1", final_transcript="x", producer=PRODUCER
            )


def test_completed_result_allows_done_wording() -> None:
    contract = FakeContract(data=_result_data("completed"))

    async def run():
        return await send_guidance_intent(
            contract, action="start_guidance", utterance_id="utt-1", final_transcript="开始引导"
        )

    outcome = asyncio.run(run())
    assert outcome.completed is True
    assert outcome.speech == "好的，已开始引导。"
    assert contract.calls[0]["action"] == "start_guidance"


def test_pending_result_never_claims_completion() -> None:
    contract = FakeContract(data=_result_data("pending"))

    async def run():
        return await send_guidance_intent(
            contract, action="pause_guidance", utterance_id="utt-2", final_transcript="暂停引导"
        )

    outcome = asyncio.run(run())
    assert outcome.completed is False
    assert "已暂停" not in outcome.speech
    assert "不能确认" in outcome.speech


def test_rejected_result_is_reported_with_reason_and_no_done_wording() -> None:
    contract = FakeContract(data=_result_data("rejected", accepted=False, reason="BUSY"))

    async def run():
        return await send_guidance_intent(
            contract, action="end_guidance", utterance_id="utt-3", final_transcript="结束引导"
        )

    outcome = asyncio.run(run())
    assert outcome.accepted is False
    assert outcome.operation_state == "rejected"
    assert "BUSY" in outcome.speech
    assert "已结束" not in outcome.speech


def test_transport_failure_reports_unknown_without_retrying() -> None:
    contract = FakeContract(raises=FakeTimeout("deadline"))

    async def run():
        return await send_guidance_intent(
            contract, action="repeat_guidance", utterance_id="utt-4", final_transcript="重复引导"
        )

    outcome = asyncio.run(run())
    assert outcome.operation_state == "unknown"
    assert "DEADLINE_EXCEEDED" in (outcome.reason_code or "")
    assert "没能确认" in outcome.speech and "不重复发送" in outcome.speech
    assert len(contract.calls) == 1, "超时不得自动重发（合同要求先查同一请求状态）"


def test_error_status_maps_to_unavailable() -> None:
    contract = FakeContract(
        status="unavailable", error=schema.error_payload("DEPENDENCY_UNAVAILABLE", "主控未连接", True)
    )

    async def run():
        return await send_guidance_intent(
            contract, action="start_guidance", utterance_id="utt-5", final_transcript="开始引导"
        )

    outcome = asyncio.run(run())
    assert outcome.operation_state == "unavailable"
    assert "DEPENDENCY_UNAVAILABLE" in outcome.speech
    assert "已开始" not in outcome.speech
    assert "没有送达" not in outcome.speech
    assert "不能确认" in outcome.speech


def test_module_never_emits_movement_instructions() -> None:
    """sanity：本模块只产生引导控制措辞，不产生行走/转向指令。"""
    import aieyes.contract.guidance_intent as module

    source = inspect.getsource(module)
    for word in ("向前走", "左转", "右转", "迈步", "通行授权"):
        assert word not in source
