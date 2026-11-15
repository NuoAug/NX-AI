"""interaction.v0.1 传输层（TCP + 长度前缀分帧 + 请求关联）。

合同没有固定“传输帧格式/连接方向”，只要求请求-响应通道与常驻事件通道。
本适配器选择：

* 交互进程作为 TCP **服务端**，绑定 ``ContractConfig.host:port``；
  主控作为客户端连接。这是 ``accept_timeout_s`` 存在的原因，也避免交互侧
  在无主控时反复重连。
* 单连接**双向多路复用**：主控发来的请求（frame/scene/control/...）与
  交互发来的请求（capabilities/scene/...）走同一条连接；事件
  （guidance.event / playback.status）也复用该连接，保证“事件顺序/去重”
  在一条流上可判定。合同允许把紧急事件拆到独立通道，后续可加，语义不变。
* 分帧：4 字节大端无符号长度 + UTF-8 JSON。长度含 JSON 字节数，不含头。
  ``0 < length <= max_message_bytes``；超限/坏头/坏 JSON → 关闭连接。
* 关联：请求 ``message_id`` ↔ 响应 ``reply_to``；``request_id`` 相同才接受。
  ``reply_to``/``request_id`` 不匹配或没有对应 pending 的响应只记诊断，不打断
  连接（合同要求不匹配/迟到响应不得推进当前交互）。

本模块不依赖具体业务处理器：``Connection`` 把合法 ``Message`` 交给回调，
由 ``server.ContractServer`` 决定分发/回错误码。
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Mapping

from aieyes.contract.errors import ProtocolError
from aieyes.contract.schema import Message, parse_message, peek_envelope

logger = logging.getLogger(__name__)

#: 4 字节大端长度前缀
HEADER_SIZE = 4

#: 合同/配置允许的最大消息体（JSON 字节数，不含 4 字节头）
DEFAULT_MAX_MESSAGE_BYTES = 4 * 1024 * 1024

#: 单条 JSON 允许的嵌套深度上限（防止恶意消息耗尽栈）
MAX_JSON_DEPTH = 64


class FramingError(Exception):
    """帧头/长度/JSON 编码非法；调用方应关闭连接。"""


class ConnectionClosed(ConnectionError):
    """连接已断开；未完成请求全部失败。"""


class RequestTimeout(TimeoutError):
    """请求在期限内没有收到关联响应。"""


async def _maybe_await(result: Any) -> None:
    """回调既可以是普通函数也可以是协程函数。"""
    if inspect.isawaitable(result):
        await result


def encode_header(payload_size: int) -> bytes:
    if not isinstance(payload_size, int) or payload_size <= 0:
        raise FramingError(f"invalid frame length: {payload_size!r}")
    return payload_size.to_bytes(HEADER_SIZE, "big")


def encode_message(msg: Message) -> bytes:
    """``Message`` → 长度前缀 + UTF-8 JSON 帧。"""
    payload = json.dumps(
        msg.to_dict(),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return encode_header(len(payload)) + payload


async def read_frame(reader: asyncio.StreamReader, *, max_message_bytes: int) -> bytes | None:
    """读取一个完整帧的 JSON 字节；对端正常关闭返回 None。"""
    try:
        header = await reader.readexactly(HEADER_SIZE)
    except asyncio.IncompleteReadError as exc:
        if not exc.partial:
            return None
        raise FramingError("truncated frame header") from exc
    size = int.from_bytes(header, "big")
    if size <= 0:
        raise FramingError(f"invalid frame length: {size}")
    if size > max_message_bytes:
        raise FramingError(f"frame too large: {size} > {max_message_bytes}")
    try:
        return await reader.readexactly(size)
    except asyncio.IncompleteReadError as exc:
        raise FramingError("truncated frame payload") from exc


def decode_payload(payload: bytes) -> dict[str, Any]:
    """JSON 字节 → dict；拒绝非对象/过深/非 UTF-8。"""
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FramingError(f"invalid JSON frame: {exc}") from exc
    if not isinstance(raw, dict):
        raise FramingError("frame payload must be a JSON object")
    _check_depth(raw)
    return raw


def _check_depth(value: Any, depth: int = 0) -> None:
    if depth > MAX_JSON_DEPTH:
        raise FramingError(f"JSON nesting deeper than {MAX_JSON_DEPTH}")
    if isinstance(value, dict):
        for item in value.values():
            _check_depth(item, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_depth(item, depth + 1)


def encode_error_frame(exc: ProtocolError) -> dict[str, Any]:
    """无法关联到请求时最少可用的诊断帧（contract 无匹配请求时仅诊断）。"""
    return {"schema_version": "interaction.v0.1", "type": "error.diagnostic",
            "code": exc.code, "message": exc.message}


class Connection:
    """一条已完成 TCP 握手（分帧）的连接，负责收发与请求-响应关联。

    ``on_message``：收到合法、非“响应”的消息（请求或事件）时回调；
    ``on_invalid``：收到 JSON 对象但不符合 schema 时回调，由上层决定是否回
    合同错误码（能识别 request 就回 result，否则诊断）；
    ``on_unmatched_response``：响应找不到 pending 请求时回调（诊断）；
    ``on_closed``：连接关闭（含异常）后回调一次。
    """

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
        on_message: Callable[[Message], Any] | None = None,
        on_invalid: Callable[[dict[str, Any], ProtocolError], Any] | None = None,
        on_unmatched_response: Callable[[Message], Any] | None = None,
        on_closed: Callable[[], None] | None = None,
        name: str = "peer",
        logger_: logging.Logger | None = None,
    ) -> None:
        self._reader = reader
        self._writer = writer
        self.max_message_bytes = max_message_bytes
        self.name = name
        self._log = logger_ or logger
        self._on_message = on_message
        self._on_invalid = on_invalid
        self._on_unmatched = on_unmatched_response
        self._on_closed = on_closed
        self._pending: dict[str, tuple[asyncio.Future[Message], Message]] = {}
        self._recv_task: asyncio.Task[None] | None = None
        self._closed_event = asyncio.Event()
        self._closed = False
        self.close_reason: str = ""
        self.framing_error: FramingError | None = None
        self.received_count = 0
        self.sent_count = 0

    # --- 状态 -----------------------------------------------------------
    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def peer(self) -> str:
        try:
            info = self._writer.get_extra_info("peername")
        except Exception:  # pragma: no cover - 关闭后 hint 可能不可用
            info = None
        return self.name if info is None else f"{self.name}:{info}"

    def start(self) -> None:
        if self._recv_task is not None:
            return
        self._recv_task = asyncio.get_running_loop().create_task(
            self._recv_loop(), name=f"contract-recv-{self.name}"
        )

    async def wait_closed(self) -> None:
        """等待接收循环/关闭流程结束。"""
        await self._closed_event.wait()

    # --- 发送 -----------------------------------------------------------
    async def send(self, msg: Message) -> None:
        if self._closed:
            raise ConnectionClosed("connection closed")
        frame = encode_message(msg)
        if len(frame) - HEADER_SIZE > self.max_message_bytes:
            raise FramingError(
                f"outbound message too large: {len(frame) - HEADER_SIZE} > {self.max_message_bytes}"
            )
        self._writer.write(frame)
        await self._writer.drain()
        self.sent_count += 1

    async def request(self, msg: Message, *, timeout_ms: int | None = None) -> Message:
        """发送请求并按 ``reply_to`` 等待关联响应。

        ``timeout_ms`` 必须来自请求的 ``deadline_mono_ms``（T15 默认 10s）；
        超时抛 :class:`RequestTimeout`，连接断开抛 :class:`ConnectionClosed`。
        """
        if not msg.is_request:
            raise ValueError(f"{msg.type} is not a request")
        loop = asyncio.get_running_loop()
        future: asyncio.Future[Message] = loop.create_future()
        if msg.message_id in self._pending:
            raise ProtocolError(
                "INVALID_PAYLOAD", f"duplicate pending message_id: {msg.message_id}"
            )
        self._pending[msg.message_id] = (future, msg)
        try:
            await self.send(msg)
        except BaseException:
            self._pending.pop(msg.message_id, None)
            raise
        timeout_s = None if timeout_ms is None else max(0.0, timeout_ms / 1000.0)
        try:
            return await asyncio.wait_for(future, timeout=timeout_s)
        except asyncio.TimeoutError as exc:
            raise RequestTimeout(
                f"no {msg.type} response within {timeout_ms} ms"
            ) from exc
        finally:
            self._pending.pop(msg.message_id, None)

    # --- 接收 -----------------------------------------------------------
    async def _recv_loop(self) -> None:
        try:
            while True:
                payload = await read_frame(self._reader, max_message_bytes=self.max_message_bytes)
                if payload is None:
                    self.close_reason = "peer closed"
                    break
                raw = decode_payload(payload)
                self.received_count += 1
                try:
                    msg = parse_message(raw, strict=False)
                except ProtocolError as exc:
                    envelope = peek_envelope(raw)
                    self._log.warning(
                        "invalid contract message (type=%r): %s", envelope.get("type"), exc
                    )
                    if self._on_invalid is not None:
                        await _maybe_await(self._on_invalid(raw, exc))
                    continue
                if msg.is_response:
                    if not self._resolve_response(msg):
                        self._log.warning(
                            "unmatched/late response reply_to=%r request_id=%r",
                            msg.reply_to,
                            msg.request_id,
                        )
                        if self._on_unmatched is not None:
                            await _maybe_await(self._on_unmatched(msg))
                    continue
                if self._on_message is not None:
                    await _maybe_await(self._on_message(msg))
        except asyncio.CancelledError:
            raise
        except FramingError as exc:
            self.framing_error = exc
            self.close_reason = f"framing error: {exc}"
            self._log.warning("closing connection %s: %s", self.peer, exc)
        except (ConnectionError, asyncio.IncompleteReadError) as exc:
            self.close_reason = f"connection error: {exc!r}"
        except Exception as exc:  # pragma: no cover - 防御回调异常
            self.close_reason = f"unexpected error: {exc!r}"
            self._log.exception("unexpected error in contract receive loop")
        finally:
            await self._teardown(self.close_reason or "closed")

    def _resolve_response(self, msg: Message) -> bool:
        if not msg.reply_to:
            return False
        entry = self._pending.get(msg.reply_to)
        if entry is None:
            return False
        future, request = entry
        if future.done():
            return False
        if msg.request_id != request.request_id:
            # 合同：reply_to/request_id 不匹配的响应只能诊断，不能推进交互
            self._log.warning(
                "response request_id mismatch: got %r want %r", msg.request_id, request.request_id
            )
            return False
        self._pending.pop(msg.reply_to, None)
        future.set_result(msg)
        return True

    # --- 关闭 -----------------------------------------------------------
    async def close(self, reason: str = "closed by local side") -> None:
        if self._closed:
            return
        self.close_reason = self.close_reason or reason
        task = self._recv_task
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
        else:
            await self._teardown(self.close_reason)

    async def _teardown(self, reason: str) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.close_reason:
            self.close_reason = reason
        self._fail_pending(ConnectionClosed(self.close_reason or "connection closed"))
        try:
            self._writer.close()
            await self._writer.wait_closed()
        except Exception:  # pragma: no cover - 已断开的 socket
            pass
        self._closed_event.set()
        if self._on_closed is not None:
            try:
                self._on_closed()
            except Exception:  # pragma: no cover
                self._log.exception("on_closed callback failed")

    def _fail_pending(self, exc: BaseException) -> None:
        pending = list(self._pending.values())
        self._pending.clear()
        for future, _request in pending:
            if not future.done():
                future.set_exception(exc)


__all__ = [
    "Connection",
    "ConnectionClosed",
    "DEFAULT_MAX_MESSAGE_BYTES",
    "FramingError",
    "HEADER_SIZE",
    "MAX_JSON_DEPTH",
    "RequestTimeout",
    "decode_payload",
    "encode_error_frame",
    "encode_header",
    "encode_message",
    "read_frame",
]
