"""interaction.v0.1 合同层。

模块划分：

* :mod:`aieyes.contract.schema` —— 9 组消息的构造/严格校验、错误码、合同语义辅助
  （测量有效性、播放去重、序列防回退、握手指纹）。
* :mod:`aieyes.contract.transport` —— TCP 长度前缀分帧、请求-响应关联、超时/断线。
* :mod:`aieyes.contract.server` —— 交互侧 TCP 服务端、入站分发、重连 epoch。
* :mod:`aieyes.contract.handlers` —— ``ContractHandler`` 协议、``NullHandler``、
  ``FakeHandler``、状态机门面 ``InteractionContract``。
* :mod:`aieyes.contract.fixture_client` —— 扮演 main_control 的测试夹具。
* :mod:`aieyes.contract.cases` —— C01–C12 合同案例（``python -m aieyes.contract.cases``）。

合同本身的权威文本是 ``Docs/MCPT/主控与交互接口合同.md``；本包只实现合同，
不拥有交互状态机。
"""

from aieyes.contract.errors import (
    EVIDENCE_MODES,
    ERROR_CODES,
    RESPONSE_STATUSES,
    ROLES,
    RETRYABLE_DEFAULT,
    SCHEMA_VERSION,
    ProtocolError,
    as_error_payload,
    error_payload,
)
from aieyes.contract.schema import (
    ALL_MESSAGE_TYPES,
    CONTROL_ACTIONS,
    DEFAULT_RESPONSE_TIMEOUT_MS,
    EVENT_TYPES,
    GuidanceDeduper,
    Message,
    REPLY_TYPE,
    REQUEST_TYPES,
    RESPONSE_TYPES,
    SequenceGuard,
    capabilities_get,
    capabilities_result,
    control_intent,
    control_result,
    error_response,
    frame_acquire,
    frame_result,
    guidance_event,
    guidance_is_fresh,
    handshake_fingerprint,
    interaction_status,
    interaction_status_get,
    is_completed_ok,
    measurement_distance_m,
    new_message_id,
    new_request_id,
    now_ms,
    output_cancel,
    output_cancel_result,
    parse_message,
    playback_key,
    playback_status,
    response_for,
    scene_measurement,
    scene_query,
    scene_result,
    target_observation,
    target_result,
    validate_message,
)
from aieyes.contract.transport import (
    Connection,
    ConnectionClosed,
    DEFAULT_MAX_MESSAGE_BYTES,
    FramingError,
    RequestTimeout,
    decode_payload,
    encode_header,
    encode_message,
    read_frame,
)
from aieyes.contract.handlers import (
    ContractHandler,
    FakeHandler,
    HANDLER_METHODS,
    HandlerFailure,
    HandlerReply,
    INBOUND_REQUEST_TYPES,
    InteractionContract,
    NullHandler,
    status_for_error_code,
)
from aieyes.contract.server import ContractServer, NotConnected, read_host_boot_id
from aieyes.contract.fixture_client import (
    DEFAULT_CAPABILITIES,
    FixtureClient,
    FixtureResponder,
)

__all__ = [
    # errors
    "SCHEMA_VERSION",
    "ROLES",
    "EVIDENCE_MODES",
    "RESPONSE_STATUSES",
    "ERROR_CODES",
    "RETRYABLE_DEFAULT",
    "ProtocolError",
    "error_payload",
    "as_error_payload",
    # schema constants
    "ALL_MESSAGE_TYPES",
    "REQUEST_TYPES",
    "RESPONSE_TYPES",
    "EVENT_TYPES",
    "REPLY_TYPE",
    "CONTROL_ACTIONS",
    "DEFAULT_RESPONSE_TIMEOUT_MS",
    # schema core
    "Message",
    "parse_message",
    "validate_message",
    "new_message_id",
    "new_request_id",
    "now_ms",
    "response_for",
    "error_response",
    # schema constructors
    "capabilities_get",
    "capabilities_result",
    "frame_acquire",
    "frame_result",
    "scene_query",
    "scene_result",
    "control_intent",
    "control_result",
    "target_observation",
    "target_result",
    "guidance_event",
    "playback_status",
    "output_cancel",
    "output_cancel_result",
    "interaction_status_get",
    "interaction_status",
    # schema semantics helpers
    "measurement_distance_m",
    "scene_measurement",
    "guidance_is_fresh",
    "is_completed_ok",
    "playback_key",
    "handshake_fingerprint",
    "SequenceGuard",
    "GuidanceDeduper",
    # transport
    "Connection",
    "ConnectionClosed",
    "FramingError",
    "RequestTimeout",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "encode_header",
    "encode_message",
    "read_frame",
    "decode_payload",
    # handlers
    "ContractHandler",
    "HandlerReply",
    "HandlerFailure",
    "NullHandler",
    "FakeHandler",
    "InteractionContract",
    "HANDLER_METHODS",
    "INBOUND_REQUEST_TYPES",
    "status_for_error_code",
    # server
    "ContractServer",
    "NotConnected",
    "read_host_boot_id",
    # fixtures
    "FixtureClient",
    "FixtureResponder",
    "DEFAULT_CAPABILITIES",
]
