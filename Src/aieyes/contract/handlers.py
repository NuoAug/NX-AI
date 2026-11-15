"""合同处理器协议、测试替身与交互侧门面。

``handlers.py`` 是合同层与 interaction 会话状态机之间的边界：

* :class:`ContractHandler` —— 状态机实现该协议；合同层把**入站**请求
  （主控发来的 ``capabilities.get`` / ``interaction.status.get`` /
  ``output.cancel``）交给它，返回响应 ``data``（或抛
  :class:`HandlerFailure` / :class:`~aieyes.contract.errors.ProtocolError`）。
* :class:`NullHandler` —— 没有真实设备时的最小实现：如实声明能力缺失，
  设备相关请求返回 ``DEPENDENCY_UNAVAILABLE`` / ``rejected``，绝不假装完成。
* :class:`FakeHandler` —— 可脚本化响应、可注入延迟/错误的测试替身。
* :class:`InteractionContract` —— 状态机使用的出站门面（typed 方法 +
  自动填充 producer/clock/deadline/evidence_mode），与
  ``server.ContractServer`` 组合。

本模块不实现状态机；``guidance.event`` 的入站处理只负责交给状态机
（``ContractHandler.guidance_event``），去重/过期/替代判断由状态机调用
``schema.GuidanceDeduper`` 完成。
"""

from __future__ import annotations

import asyncio
import logging
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Mapping, Protocol, Sequence, runtime_checkable

from aieyes.contract.errors import (
    ERROR_CODES,
    RETRYABLE_DEFAULT,
    SCHEMA_VERSION,
    EVIDENCE_MODES,
    ProtocolError,
    as_error_payload,
)
from aieyes.contract.schema import (
    CONTROL_ACTIONS,
    Message,
    capabilities_get,
    control_intent,
    frame_acquire,
    guidance_event,
    interaction_status_get,
    now_ms,
    playback_status,
    scene_query,
    target_observation,
)

if TYPE_CHECKING:  # pragma: no cover
    from aieyes.contract.server import ContractServer

logger = logging.getLogger(__name__)

#: 请求类型 → ContractHandler 方法名
HANDLER_METHODS: dict[str, str] = {
    "capabilities.get": "capabilities_get",
    "frame.acquire": "frame_acquire",
    "scene.query": "scene_query",
    "control.intent": "control_intent",
    "target.observation": "target_observation",
    "output.cancel": "output_cancel",
    "interaction.status.get": "interaction_status_get",
}

#: 只有这些请求允许由主控发往交互（其余方向由交互发起）
INBOUND_REQUEST_TYPES = frozenset({"capabilities.get", "interaction.status.get", "output.cancel"})

#: 错误码 → 响应 ``status`` 的建议映射（合同只约束集合，不固定逐码对应）
_STATUS_FOR_CODE: dict[str, str] = {
    "DEPENDENCY_UNAVAILABLE": "unavailable",
    "NO_FRAME": "unavailable",
    "STALE_FRAME": "unavailable",
    "OUTPUT_UNAVAILABLE": "unavailable",
    "BUSY": "unavailable",
    "DEADLINE_EXCEEDED": "unavailable",
    "INVALID_FRAME_REF": "rejected",
    "UNSUPPORTED_VERSION": "rejected",
    "CLOCK_UNVERIFIED": "rejected",
    "STALE_STATE": "rejected",
    "INVALID_TARGET": "rejected",
    "INVALID_PAYLOAD": "rejected",
    "CANCELLED": "rejected",
}


def status_for_error_code(code: str) -> str:
    return _STATUS_FOR_CODE.get(code, "error")


# ---------------------------------------------------------------------------
# 状态机 → 合同层的返回/异常类型
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class HandlerReply:
    """处理器返回的非 ok 响应；正常路径直接返回 ``data`` dict 即可。"""

    status: str = "ok"
    data: Mapping[str, Any] | None = None
    error: Any = None

    def __post_init__(self) -> None:
        if self.status not in ("ok", "unavailable", "rejected", "error"):
            raise ValueError(f"invalid handler status: {self.status!r}")
        if self.status == "ok" and self.error is not None:
            raise ValueError("ok HandlerReply must not carry error")
        if self.status != "ok" and self.error is None:
            raise ValueError(f"{self.status} HandlerReply requires error")


class HandlerFailure(Exception):
    """处理器拒绝/无法完成请求；由合同层转换为合同错误响应。"""

    def __init__(
        self,
        code: str,
        message: str | None = None,
        *,
        status: str | None = None,
        retryable: bool | None = None,
    ) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown interaction.v0.1 error code: {code!r}")
        self.code = code
        self.message = message or code
        self.status = status or status_for_error_code(code)
        self.retryable = RETRYABLE_DEFAULT[code] if retryable is None else bool(retryable)
        super().__init__(f"{code}: {self.message}")

    @classmethod
    def from_protocol_error(
        cls, exc: ProtocolError, *, status: str | None = None
    ) -> "HandlerFailure":
        return cls(exc.code, exc.message, status=status, retryable=exc.retryable)


# ---------------------------------------------------------------------------
# 处理器协议
# ---------------------------------------------------------------------------
@runtime_checkable
class ContractHandler(Protocol):
    """interaction 状态机接入点。

    入站请求的响应数据必须符合 ``schema`` 对应 ``*.result.data`` 结构；
    需要回 ``unavailable``/``rejected`` 时抛 :class:`HandlerFailure`（带
    status），或返回 :class:`HandlerReply`。
    """

    async def capabilities_get(
        self, request: Message
    ) -> Mapping[str, Any] | HandlerReply: ...

    async def frame_acquire(
        self, request: Message
    ) -> Mapping[str, Any] | HandlerReply: ...

    async def scene_query(
        self, request: Message
    ) -> Mapping[str, Any] | HandlerReply: ...

    async def control_intent(
        self, request: Message
    ) -> Mapping[str, Any] | HandlerReply: ...

    async def target_observation(
        self, request: Message
    ) -> Mapping[str, Any] | HandlerReply: ...

    async def output_cancel(
        self, request: Message
    ) -> Mapping[str, Any] | HandlerReply: ...

    async def interaction_status_get(
        self, request: Message
    ) -> Mapping[str, Any] | HandlerReply: ...

    async def guidance_event(self, event: Message) -> None: ...


# ---------------------------------------------------------------------------
# data 构造辅助（供处理器/夹具/状态机复用）
# ---------------------------------------------------------------------------
def capabilities_result_data(
    *,
    supported_versions: Sequence[str] = (SCHEMA_VERSION,),
    capabilities: Mapping[str, bool] | None = None,
    supported_evidence_modes: Sequence[str] = EVIDENCE_MODES,
    host_boot_id: str | None = None,
    unimplemented: Sequence[str] = (),
    resources: Mapping[str, Any] | None = None,
    service: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "supported_versions": list(supported_versions),
        "capabilities": dict(capabilities or {}),
        "supported_evidence_modes": list(supported_evidence_modes),
        "host_boot_id": host_boot_id,
        "unimplemented": list(unimplemented),
        "resources": None if resources is None else dict(resources),
        "service": None if service is None else dict(service),
    }


def frame_result_data(
    *,
    frame_id: str,
    captured_mono_ms: int | None,
    width_px: int,
    height_px: int,
    sha256: str,
    asset_id: str,
    path: str,
    encoding: str = "jpeg",
    source_kind: str = "camera",
    is_historical: bool = False,
    color_space: str = "RGB8",
    parent_frame_id: str | None = None,
    transform_to_parent: Sequence[float] | None = None,
    quality: Mapping[str, str] | None = None,
    available_until_mono_ms: int = 0,
) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "captured_mono_ms": captured_mono_ms,
        "source_kind": source_kind,
        "is_historical": is_historical,
        "width_px": width_px,
        "height_px": height_px,
        "color_space": color_space,
        "parent_frame_id": parent_frame_id,
        "transform_to_parent": None if transform_to_parent is None else list(transform_to_parent),
        "quality": dict(quality or {"focus": "unknown", "exposure": "unknown"}),
        "asset": {
            "asset_id": asset_id,
            "transport": "file",
            "path": path,
            "encoding": encoding,
            "sha256": sha256,
            "available_until_mono_ms": available_until_mono_ms,
        },
    }


def measurement_data(
    *,
    distance_m: float | None = None,
    valid: bool = False,
    valid_until_mono_ms: int | None = None,
    reason_code: str | None = None,
) -> dict[str, Any]:
    return {
        "distance_m": distance_m,
        "valid": valid,
        "valid_until_mono_ms": valid_until_mono_ms,
        "reason_code": reason_code,
    }


def scene_object_data(
    *,
    track_id: str,
    class_name: str,
    measurement: Mapping[str, Any] | None = None,
    bearing: str | None = None,
) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "class_name": class_name,
        "bearing": bearing,
        "measurement": dict(
            measurement
            or measurement_data(valid=False, reason_code="DEPTH_INVALID")
        ),
    }


def scene_result_data(
    *,
    state_id: str,
    sequence: int,
    observed_mono_ms: int,
    valid_until_mono_ms: int,
    objects: Sequence[Mapping[str, Any]] = (),
    navigation: Mapping[str, Any] | None = None,
    sensor_status: Mapping[str, Any] | None = None,
    coordinate_frame: str = "camera_depth_optical",
    measurement: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "state_id": state_id,
        "sequence": sequence,
        "observed_mono_ms": observed_mono_ms,
        "valid_until_mono_ms": valid_until_mono_ms,
        "coordinate_frame": coordinate_frame,
        "navigation": dict(
            navigation
            or {
                "mode": "follow",
                "decision": "unknown",
                "reason_code": "NONE",
                "speech_fact": None,
            }
        ),
        "objects": [dict(obj) for obj in objects],
        "sensor_status": dict(
            sensor_status or {"rgb": "ok", "depth": "ok", "heading": "ok"}
        ),
    }
    if measurement is not None:
        data["measurement"] = dict(measurement)
    return data


def control_result_data(
    *,
    accepted: bool,
    operation_state: str,
    actual_mode: str | None = None,
    reason_code: str | None = None,
    state_id: str | None = None,
) -> dict[str, Any]:
    return {
        "accepted": accepted,
        "operation_state": operation_state,
        "actual_mode": actual_mode,
        "reason_code": reason_code,
        "state_id": state_id,
    }


def target_result_data(
    *,
    match_status: str,
    track_id: str | None = None,
    valid_until_mono_ms: int | None = None,
    reason_code: str | None = None,
) -> dict[str, Any]:
    return {
        "match_status": match_status,
        "track_id": track_id,
        "valid_until_mono_ms": valid_until_mono_ms,
        "reason_code": reason_code,
    }


def cancel_result_data(
    *,
    device_stopped: bool,
    queue_cleared: bool,
    compute_ended: bool,
    reason_code: str | None = None,
    basis: str | None = "not_applicable",
) -> dict[str, Any]:
    return {
        "device_stopped": device_stopped,
        "queue_cleared": queue_cleared,
        "compute_ended": compute_ended,
        "reason_code": reason_code,
        "basis": basis,
    }


def playback_status_data(
    *,
    state: str,
    occurred_mono_ms: int,
    source_event_id: str | None = None,
    source_request_id: str | None = None,
    utterance_id: str | None = None,
    segment_index: int = 0,
    generation: int | None = None,
    priority: str | None = None,
    start_basis: str | None = None,
    reason_code: str | None = None,
) -> dict[str, Any]:
    return {
        "source_event_id": source_event_id,
        "source_request_id": source_request_id,
        "utterance_id": utterance_id,
        "segment_index": segment_index,
        "generation": generation,
        "priority": priority,
        "state": state,
        "occurred_mono_ms": occurred_mono_ms,
        "start_basis": start_basis,
        "reason_code": reason_code,
    }


def interaction_status_data(
    *,
    mode: str = "degraded",
    active_request_id: str | None = None,
    audio_input_state: Mapping[str, Any] | str = "unknown",
    audio_output_state: Mapping[str, Any] | str = "unknown",
    services: Mapping[str, Mapping[str, Any]] | None = None,
    queues: Mapping[str, int] | None = None,
    memory: Mapping[str, Any] | None = None,
    metrics_window: Mapping[str, Any] | None = None,
    last_error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """最小 ``interaction.status.data``；未测项按合同写 null/unknown，不猜数值。"""

    def _service(version: str | None = None, state: str = "unavailable") -> dict[str, Any]:
        return {"version": version, "state": state}

    return {
        "mode": mode,
        "active_request_id": active_request_id,
        "audio_input_state": audio_input_state,
        "audio_output_state": audio_output_state,
        "services": dict(
            services
            or {
                "asr": _service(),
                "vlm": _service(),
                "ocr": _service(),
                "tts": _service(),
                "kws": _service(),
            }
        ),
        "queues": dict(queues or {}),
        "memory": dict(
            memory
            or {
                "metric": "process_rss",
                "bytes": None,
                "processes": None,
                "sampling": None,
            }
        ),
        "metrics_window": dict(metrics_window or {"window_ms": None, "samples": 0}),
        "last_error": None if last_error is None else dict(last_error),
    }


# ---------------------------------------------------------------------------
# NullHandler：所有设备能力缺失时的诚实最小实现
# ---------------------------------------------------------------------------
class NullHandler:
    """无设备/降级模式下的最小处理器。

    * ``capabilities.get``：如实在 ``capabilities`` 中声明各能力为 false；
    * 依赖相机/深度/场景状态的请求：``DEPENDENCY_UNAVAILABLE``
      （``status=unavailable``，可重试）；
    * ``control.intent``：不假装执行，回 ``accepted=false`` +
      ``operation_state=rejected``；
    * ``output.cancel``：没有输出设备，三个布尔位如实为 false，
      ``basis=not_applicable``；
    * ``interaction.status.get``：最小状态（模式 degraded，服务 unavailable）；
    * ``guidance.event``：只记录，不播放。
    """

    def __init__(self, *, host_boot_id: str | None = None) -> None:
        self.host_boot_id = host_boot_id
        self.guidance_events: list[Message] = []

    async def capabilities_get(self, request: Message) -> Mapping[str, Any]:
        return capabilities_result_data(
            capabilities={
                "frame_provider": False,
                "derived_image": False,
                "scene_state": False,
                "navigation_decision": False,
                "track_id_match": False,
                "guidance_events": False,
                "interaction_status": True,
                "output_cancel": True,
            },
            unimplemented=[
                "frame.acquire",
                "scene.query",
                "target.observation",
                "control.intent",
                "guidance.event",
            ],
            host_boot_id=self.host_boot_id,
        )

    async def frame_acquire(self, request: Message) -> Mapping[str, Any]:
        raise HandlerFailure("DEPENDENCY_UNAVAILABLE", "no frame provider attached")

    async def scene_query(self, request: Message) -> Mapping[str, Any]:
        raise HandlerFailure("DEPENDENCY_UNAVAILABLE", "no scene state source attached")

    async def control_intent(self, request: Message) -> Mapping[str, Any]:
        return control_result_data(
            accepted=False,
            operation_state="rejected",
            actual_mode=None,
            reason_code="NO_OUTPUT_DEVICE",
            state_id=None,
        )

    async def target_observation(self, request: Message) -> Mapping[str, Any]:
        raise HandlerFailure("DEPENDENCY_UNAVAILABLE", "no main_control target matcher attached")

    async def output_cancel(self, request: Message) -> Mapping[str, Any]:
        return cancel_result_data(
            device_stopped=False,
            queue_cleared=False,
            compute_ended=False,
            reason_code="NO_OUTPUT_DEVICE",
            basis="not_applicable",
        )

    async def interaction_status_get(self, request: Message) -> Mapping[str, Any]:
        return interaction_status_data(mode="degraded")

    async def guidance_event(self, event: Message) -> None:
        self.guidance_events.append(event)
        logger.info(
            "NullHandler received guidance.event %r (not played: no output device)",
            event.data.get("event_id") if isinstance(event.data, Mapping) else None,
        )


# ---------------------------------------------------------------------------
# FakeHandler：可脚本化测试替身
# ---------------------------------------------------------------------------
_ScriptValue = Mapping[str, Any] | Callable[[Message], Mapping[str, Any] | None] | HandlerReply


class FakeHandler:
    """可脚本化处理器：按消息类型预设 ``data``/失败/延迟，并记录调用。

    用法::

        fake = FakeHandler()
        fake.script("interaction.status.get", interaction_status_data(mode="idle"))
        fake.fail("frame.acquire", "DEPENDENCY_UNAVAILABLE", "camera off")
        fake.delay("scene.query", 0.05)
        server = ContractServer(config, handler=fake)

    ``strict=True``（默认）时未脚本化的请求回 ``DEPENDENCY_UNAVAILABLE``；
    ``strict=False`` 时回 :class:`NullHandler` 的诚实默认值。
    """

    def __init__(self, *, strict: bool = True) -> None:
        self.strict = strict
        self._scripts: dict[str, _ScriptValue] = {}
        self._failures: dict[str, HandlerFailure | ProtocolError] = {}
        self._delays: dict[str, float] = {}
        self.requests: list[Message] = []
        self.guidance_events: list[Message] = []
        self.counts: Counter[str] = Counter()
        self.guidance_handler: Callable[[Message], Awaitable[None]] | None = None
        self._null = NullHandler()

    # --- 脚本 -----------------------------------------------------------
    def script(
        self,
        msg_type: str,
        data: Mapping[str, Any] | Callable[[Message], Mapping[str, Any] | None] | HandlerReply | None = None,
    ) -> "FakeHandler":
        self._scripts[msg_type] = data if data is not None else {}
        return self

    def fail(
        self,
        msg_type: str,
        code: str,
        message: str | None = None,
        *,
        status: str | None = None,
        retryable: bool | None = None,
    ) -> "FakeHandler":
        self._failures[msg_type] = HandlerFailure(
            code, message, status=status, retryable=retryable
        )
        return self

    def delay(self, msg_type: str, seconds: float) -> "FakeHandler":
        self._delays[msg_type] = float(seconds)
        return self

    def reset(self) -> None:
        self._scripts.clear()
        self._failures.clear()
        self._delays.clear()
        self.requests.clear()
        self.guidance_events.clear()
        self.counts.clear()

    # --- 通用分发 -------------------------------------------------------
    async def _handle(self, msg_type: str, request: Message) -> Mapping[str, Any] | HandlerReply:
        self.requests.append(request)
        self.counts[msg_type] += 1
        delay = self._delays.get(msg_type)
        if delay:
            await asyncio.sleep(delay)
        failure = self._failures.get(msg_type)
        if failure is not None:
            raise failure
        if msg_type in self._scripts:
            value = self._scripts[msg_type]
            if callable(value):
                value = value(request)
            if isinstance(value, HandlerReply):
                return value
            return {} if value is None else dict(value)
        if self.strict:
            raise HandlerFailure(
                "DEPENDENCY_UNAVAILABLE", f"FakeHandler has no scripted response for {msg_type}"
            )
        return await getattr(self._null, HANDLER_METHODS[msg_type])(request)

    # --- ContractHandler 实现 ------------------------------------------
    async def capabilities_get(self, request: Message) -> Mapping[str, Any] | HandlerReply:
        return await self._handle("capabilities.get", request)

    async def frame_acquire(self, request: Message) -> Mapping[str, Any] | HandlerReply:
        return await self._handle("frame.acquire", request)

    async def scene_query(self, request: Message) -> Mapping[str, Any] | HandlerReply:
        return await self._handle("scene.query", request)

    async def control_intent(self, request: Message) -> Mapping[str, Any] | HandlerReply:
        return await self._handle("control.intent", request)

    async def target_observation(self, request: Message) -> Mapping[str, Any] | HandlerReply:
        return await self._handle("target.observation", request)

    async def output_cancel(self, request: Message) -> Mapping[str, Any] | HandlerReply:
        return await self._handle("output.cancel", request)

    async def interaction_status_get(
        self, request: Message
    ) -> Mapping[str, Any] | HandlerReply:
        return await self._handle("interaction.status.get", request)

    async def guidance_event(self, event: Message) -> None:
        self.guidance_events.append(event)
        self.counts["guidance.event"] += 1
        if self.guidance_handler is not None:
            await self.guidance_handler(event)


# ---------------------------------------------------------------------------
# 状态机门面：交互侧主动请求/事件
# ---------------------------------------------------------------------------
class InteractionContract:
    """状态机出站调用的 typed 门面。

    ``ContractServer`` 管理连接；本类只构造合同消息并委托
    ``server.send_request`` / ``server.send_event``，自动填充：

    * ``producer`` = {role: interaction, instance_id: 配置实例};
    * ``clock_id`` = 本地单调时钟域；``timestamp_mono_ms`` = 当前单调毫秒；
    * ``deadline_mono_ms`` = now + timeout（默认 T15=10s）；
    * ``evidence_mode`` = 配置（live/replay/synthetic）。

    返回值是 ``*.result`` 的 ``Message``：调用方必须先检查 ``msg.ok`` 与具体
    ``data``，不能把“请求发出”当作“执行完成”。
    """

    def __init__(self, server: "ContractServer", *, default_timeout_ms: int | None = None) -> None:
        self._server = server
        self.default_timeout_ms = (
            server.default_timeout_ms if default_timeout_ms is None else default_timeout_ms
        )

    @property
    def server(self) -> "ContractServer":
        return self._server

    def _envelope_kwargs(self) -> dict[str, Any]:
        return {
            "producer": self._server.producer,
            "clock_id": self._server.clock_id,
            "evidence_mode": self._server.evidence_mode,
        }

    def _deadline(self, timeout_ms: int | None) -> int:
        timeout = self.default_timeout_ms if timeout_ms is None else int(timeout_ms)
        return now_ms() + timeout

    async def _request(self, msg: Message, timeout_ms: int | None) -> Message:
        return await self._server.send_request(msg, timeout_ms=timeout_ms)

    # --- 出站请求 -------------------------------------------------------
    async def query_capabilities(
        self, *, requested: Sequence[str] | None = None, timeout_ms: int | None = None
    ) -> Message:
        msg = capabilities_get(
            producer=self._server.producer,
            requested=requested,
            clock_id=self._server.clock_id,
            evidence_mode=self._server.evidence_mode,
            deadline_mono_ms=self._deadline(timeout_ms),
        )
        return await self._request(msg, timeout_ms)

    async def acquire_frame(
        self,
        *,
        mode: str = "latest",
        purpose: str = "vqa",
        frame_id: str | None = None,
        max_age_ms: int | None = None,
        max_side_px: int | None = None,
        roi_xyxy_px: Sequence[int] | None = None,
        timeout_ms: int | None = None,
    ) -> Message:
        msg = frame_acquire(
            mode=mode,
            purpose=purpose,
            frame_id=frame_id,
            max_age_ms=max_age_ms,
            max_side_px=max_side_px,
            roi_xyxy_px=roi_xyxy_px,
            producer=self._server.producer,
            clock_id=self._server.clock_id,
            evidence_mode=self._server.evidence_mode,
            deadline_mono_ms=self._deadline(timeout_ms),
        )
        return await self._request(msg, timeout_ms)

    async def query_scene(
        self,
        *,
        track_id: str | None = None,
        fields: Sequence[str] | None = None,
        timeout_ms: int | None = None,
    ) -> Message:
        msg = scene_query(
            producer=self._server.producer,
            track_id=track_id,
            fields=fields,
            clock_id=self._server.clock_id,
            evidence_mode=self._server.evidence_mode,
            deadline_mono_ms=self._deadline(timeout_ms),
        )
        return await self._request(msg, timeout_ms)

    async def send_control_intent(
        self,
        *,
        action: str,
        utterance_id: str,
        final_transcript: str,
        target_ref: str | Mapping[str, Any] | None = None,
        user_confirmed: bool = False,
        timeout_ms: int | None = None,
    ) -> Message:
        if action not in CONTROL_ACTIONS:
            raise ProtocolError("INVALID_PAYLOAD", f"control action not allowed in v0.1: {action!r}")
        msg = control_intent(
            action=action,
            utterance_id=utterance_id,
            final_transcript=final_transcript,
            target_ref=target_ref,
            user_confirmed=user_confirmed,
            producer=self._server.producer,
            clock_id=self._server.clock_id,
            evidence_mode=self._server.evidence_mode,
            deadline_mono_ms=self._deadline(timeout_ms),
        )
        return await self._request(msg, timeout_ms)

    async def observe_target(
        self,
        *,
        visual_target_id: str,
        frame_id: str,
        roi_xyxy_px: Sequence[int],
        name: str,
        source: str,
        text: str | None = None,
        user_selection_basis: str | None = None,
        timeout_ms: int | None = None,
    ) -> Message:
        msg = target_observation(
            visual_target_id=visual_target_id,
            frame_id=frame_id,
            roi_xyxy_px=roi_xyxy_px,
            name=name,
            source=source,
            text=text,
            user_selection_basis=user_selection_basis,
            producer=self._server.producer,
            clock_id=self._server.clock_id,
            evidence_mode=self._server.evidence_mode,
            deadline_mono_ms=self._deadline(timeout_ms),
        )
        return await self._request(msg, timeout_ms)

    # --- 出站事件 -------------------------------------------------------
    async def publish_playback_status(
        self,
        *,
        state: str,
        source_event_id: str | None = None,
        source_request_id: str | None = None,
        utterance_id: str | None = None,
        segment_index: int = 0,
        generation: int | None = None,
        priority: str | None = None,
        start_basis: str | None = None,
        reason_code: str | None = None,
        occurred_mono_ms: int | None = None,
    ) -> None:
        msg = playback_status(
            state=state,
            occurred_mono_ms=now_ms() if occurred_mono_ms is None else occurred_mono_ms,
            source_event_id=source_event_id,
            source_request_id=source_request_id,
            utterance_id=utterance_id,
            segment_index=segment_index,
            generation=generation,
            priority=priority,
            start_basis=start_basis,
            reason_code=reason_code,
            producer=self._server.producer,
            clock_id=self._server.clock_id,
            evidence_mode=self._server.evidence_mode,
        )
        await self._server.send_event(msg)


__all__ = [
    "ContractHandler",
    "FakeHandler",
    "HANDLER_METHODS",
    "HandlerFailure",
    "HandlerReply",
    "INBOUND_REQUEST_TYPES",
    "InteractionContract",
    "NullHandler",
    "cancel_result_data",
    "capabilities_result_data",
    "control_result_data",
    "frame_result_data",
    "interaction_status_data",
    "measurement_data",
    "playback_status_data",
    "scene_object_data",
    "scene_result_data",
    "status_for_error_code",
    "target_result_data",
]
