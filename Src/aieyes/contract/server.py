"""interaction.v0.1 TCP 服务端：连接生命周期、请求分发、响应关联与事件。

职责边界：

* **连接方向**：交互侧监听 ``ContractConfig.host:port``，主控作为客户端
  连接（合同未固定方向，这是本适配器的选择，见 README）。
* **入站**：主控发来的请求（``capabilities.get`` / ``interaction.status.get`` /
  ``output.cancel``）交给 :class:`~aieyes.contract.handlers.ContractHandler`；
  ``guidance.event`` 交给状态机。非法消息尽量回合同错误码；无法识别为请求
  的消息只记诊断。方向错误（如主控发 ``frame.acquire``）回 ``rejected``。
* **出站**：``send_request`` 按 ``message_id``/``reply_to`` 关联并受
  ``deadline_mono_ms``（T15）约束；``send_event`` 只允许 ``playback.status``
  （``guidance.event`` 只能由主控发）。
* **重连/重启**：每建立一条新连接 ``epoch`` 递增，回调
  ``on_disconnect(old_epoch)`` / ``on_connect(new_epoch)``，供状态机撤销旧
  实时状态、重新握手；断线时所有 pending 请求立即失败（不伪造结果）。
* **来源身份**：默认只接受 ``producer.role == "main_control"``；``test_fixture``
  仅在显式测试入口（``allow_test_fixture=True``）被接受。TCP 本身无法认证
  对端，真实部署必须由本地适配器绑定通道身份（合同要求）。

服务端不实现设备能力，也不改状态机；它只保证“消息合同正确、超时/取消/
重连语义不被静默吞掉”。
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any, Awaitable, Callable, Mapping

from aieyes import clock
from aieyes.config import ContractConfig
from aieyes.contract.errors import (
    EVIDENCE_MODES,
    ROLES,
    ProtocolError,
    error_payload,
)
from aieyes.contract.handlers import (
    HANDLER_METHODS,
    INBOUND_REQUEST_TYPES,
    ContractHandler,
    HandlerFailure,
    HandlerReply,
    NullHandler,
    status_for_error_code,
)
from aieyes.contract.schema import (
    Message,
    REPLY_TYPE,
    REQUEST_TYPES,
    new_message_id,
    now_ms,
    response_for,
    validate_message,
)
from aieyes.contract.transport import (
    Connection,
    ConnectionClosed,
    DEFAULT_MAX_MESSAGE_BYTES,
    FramingError,
    RequestTimeout,
)

logger = logging.getLogger(__name__)

#: 无主控连接时调用出站 API 的错误
class NotConnected(ConnectionError):
    """main_control 尚未连接或连接已断开。"""


def read_host_boot_id(path: str = "/proc/sys/kernel/random/boot_id") -> str | None:
    """读取主机启动 ID；同一主机重启后变化，用于握手指纹。"""
    try:
        value = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


class ContractServer:
    """interaction.v0.1 TCP 服务端。

    参数
    ----
    config:
        ``ContractConfig``（监听地址、端口、实例 ID、T15、最大消息体）。
    handler:
        :class:`ContractHandler` 实现；默认 ``NullHandler``（诚实降级）。
    evidence_mode:
        本进程产生消息的证据模式；默认 ``synthetic`` 以避免误报 live，
        生产入口必须显式传 ``live``。
    clock_id:
        本地单调时钟域 ID；默认 ``aieyes.clock.CLOCK_ID``。
    allow_test_fixture:
        显式测试入口开关；打开后接受 ``test_fixture`` 角色（仍只允许
        replay/synthetic 证据模式）。
    on_connect / on_disconnect:
        连接建立/断开回调（可 async），参数为 epoch；状态机据此撤销旧状态。
    """

    def __init__(
        self,
        config: ContractConfig,
        *,
        handler: ContractHandler | None = None,
        evidence_mode: str = "synthetic",
        clock_id: str | None = clock.CLOCK_ID,
        allow_test_fixture: bool = False,
        on_connect: Callable[[int], Any] | None = None,
        on_disconnect: Callable[[int], Any] | None = None,
        max_message_bytes: int | None = None,
    ) -> None:
        if not isinstance(config, ContractConfig):
            raise TypeError("config must be aieyes.config.ContractConfig")
        if not isinstance(config.host, str) or not config.host:
            raise ValueError("contract.host must be a non-empty string")
        if not isinstance(config.port, int) or config.port < 0:
            raise ValueError("contract.port must be a non-negative integer")
        if config.role not in ROLES:
            raise ValueError(f"contract.role must be one of {ROLES}, got {config.role!r}")
        if evidence_mode not in EVIDENCE_MODES:
            raise ValueError(f"invalid evidence_mode: {evidence_mode!r}")
        self.config = config
        self.role = config.role
        self.instance_id = config.instance_id
        self.clock_id = clock_id
        self.evidence_mode = evidence_mode
        self.default_timeout_ms = int(config.response_timeout_ms)
        self.max_message_bytes = int(
            config.max_message_bytes if max_message_bytes is None else max_message_bytes
        )
        if self.max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive")
        self.allow_test_fixture = bool(allow_test_fixture)
        self.allowed_peer_roles = frozenset(
            {"main_control"} | ({"test_fixture"} if self.allow_test_fixture else set())
        )
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.host_boot_id = read_host_boot_id()
        self.handler: ContractHandler = handler if handler is not None else NullHandler(
            host_boot_id=self.host_boot_id
        )

        self._server: asyncio.AbstractServer | None = None
        self._conn: Connection | None = None
        self._inbound: asyncio.Queue[Message | None] = asyncio.Queue()
        self._worker: asyncio.Task[None] | None = None
        self._connected = asyncio.Event()
        self._started = False
        self._stopping = False
        self._epoch = 0
        self._started_mono_ms: int | None = None
        self.bound_port: int | None = None
        self.inbound_count = 0
        self.invalid_count = 0
        self.dispatched_requests = 0

    # --- 基本信息 -------------------------------------------------------
    @property
    def producer(self) -> dict[str, str]:
        return {"role": self.role, "instance_id": self.instance_id}

    @property
    def epoch(self) -> int:
        """连接代号：断线/重连递增，状态机用它丢弃旧 epoch 的迟到结果。"""
        return self._epoch

    @property
    def connected(self) -> bool:
        return self._conn is not None and not self._conn.closed

    @property
    def connection(self) -> Connection | None:
        return self._conn

    # --- 生命周期 -------------------------------------------------------
    async def start(self) -> None:
        if self._started:
            raise RuntimeError("ContractServer already started")
        self._started = True
        self._server = await asyncio.start_server(
            self._handle_client, self.config.host, self.config.port
        )
        sockets = self._server.sockets or ()
        self.bound_port = sockets[0].getsockname()[1] if sockets else self.config.port
        self._started_mono_ms = now_ms()
        self._worker = asyncio.get_running_loop().create_task(
            self._inbound_worker(), name="contract-inbound-worker"
        )
        logger.info(
            "contract server listening on %s:%s (role=%s instance=%s mode=%s)",
            self.config.host,
            self.bound_port,
            self.role,
            self.instance_id,
            self.evidence_mode,
        )

    async def stop(self) -> None:
        if not self._started:
            return
        self._stopping = True
        conn = self._conn
        if conn is not None and not conn.closed:
            await conn.close("contract server stopping")
        server = self._server
        self._server = None
        if server is not None:
            server.close()
            try:
                await server.wait_closed()
            except Exception:  # pragma: no cover
                pass
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except (asyncio.CancelledError, Exception):
                pass
            self._worker = None
        self._connected.clear()
        self._started = False
        self._stopping = False

    async def __aenter__(self) -> "ContractServer":
        await self.start()
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.stop()

    async def wait_connected(self, timeout_s: float | None = None) -> bool:
        """等待主控连上；超时返回 False（默认 ``accept_timeout_s``）。"""
        timeout = self.config.accept_timeout_s if timeout_s is None else timeout_s
        if self.connected:
            return True
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            return False

    async def wait_step(self, timeout_s: float = 1.0, quiet_s: float = 0.02) -> bool:
        """等待入站队列排空且网络侧不再有新消息（测试/收尾用）。

        仅 ``queue.join()`` 会在“帧还在 socket 里未解析”时提前返回；这里在队列
        排空后再观察一个静默窗口，确认 ``inbound_count`` 稳定才返回。
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        last = -1
        while True:
            try:
                await asyncio.wait_for(self._inbound.join(), timeout=max(0.0, deadline - loop.time()))
            except asyncio.TimeoutError:
                return False
            count = self.inbound_count
            if count == last:
                return True
            last = count
            await asyncio.sleep(quiet_s)
            if loop.time() > deadline:
                return False

    # --- 连接处理 -------------------------------------------------------
    async def _handle_client(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._stopping:
            writer.close()
            return
        old = self._conn
        if old is not None and not old.closed:
            logger.warning("new main_control connection replaces the existing one")
            await old.close("replaced by new connection")
        self._epoch += 1
        epoch = self._epoch
        conn = Connection(
            reader,
            writer,
            max_message_bytes=self.max_message_bytes,
            on_message=self._on_inbound_message,
            on_invalid=self._on_invalid_message,
            on_unmatched_response=self._on_unmatched_response,
            on_closed=lambda: self._spawn_closed_handler(),
            name=f"main_control@{epoch}",
        )
        self._conn = conn
        conn.start()
        self._connected.set()
        await self._call_callback(self.on_connect, epoch)
        logger.info("main_control connected (epoch=%d, peer=%s)", epoch, conn.peer)
        await conn.wait_closed()
        # 清理在 Connection._teardown 的 on_closed 回调里完成

    def _spawn_closed_handler(self) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:  # pragma: no cover - 事件循环关闭中
            return
        loop.create_task(self._handle_connection_closed(), name="contract-conn-closed")

    async def _handle_connection_closed(self) -> None:
        conn = self._conn
        if conn is not None and not conn.closed:
            return
        if conn is None:
            return
        # 只有“当前连接”断开才清状态；被替换的旧连接不重复触发
        if self._conn is conn:
            self._conn = None
            self._connected.clear()
            if self._started and not self._stopping:
                logger.warning(
                    "main_control disconnected (epoch=%d, reason=%s)", self._epoch, conn.close_reason
                )
                await self._call_callback(self.on_disconnect, self._epoch)

    async def _call_callback(self, callback: Callable[..., Any] | None, *args: Any) -> None:
        if callback is None:
            return
        try:
            result = callback(*args)
            if inspect.isawaitable(result):
                await result
        except Exception:  # pragma: no cover - 回调异常不应杀连接
            logger.exception("contract server callback failed")

    # --- 入站 -----------------------------------------------------------
    def _on_inbound_message(self, msg: Message) -> None:
        self.inbound_count += 1
        self._inbound.put_nowait(msg)

    async def _inbound_worker(self) -> None:
        while True:
            msg = await self._inbound.get()
            try:
                if msg is not None:
                    await self._dispatch_inbound(msg)
            except asyncio.CancelledError:
                raise
            except Exception:  # pragma: no cover - 防御兜底
                logger.exception("unexpected error handling inbound contract message")
            finally:
                self._inbound.task_done()

    async def _dispatch_inbound(self, msg: Message) -> None:
        if msg.is_event:
            await self._dispatch_event(msg)
            return
        if not msg.is_request:
            return
        if not self._peer_role_allowed(msg):
            logger.warning(
                "rejecting request %s from untrusted producer role %r", msg.type, msg.role
            )
            await self._safe_send(
                self._error_reply(
                    msg,
                    "INVALID_PAYLOAD",
                    f"producer role {msg.role!r} is not an accepted main_control identity",
                    status="rejected",
                )
            )
            return
        if msg.type not in INBOUND_REQUEST_TYPES:
            logger.warning("rejecting wrong-direction request %s from main_control", msg.type)
            await self._safe_send(
                self._error_reply(
                    msg,
                    "INVALID_PAYLOAD",
                    f"{msg.type} is an interaction→main_control request, not main_control→interaction",
                    status="rejected",
                )
            )
            return
        self.dispatched_requests += 1
        reply = await self._invoke_handler(msg)
        if reply is not None:
            await self._safe_send(reply)

    async def _dispatch_event(self, msg: Message) -> None:
        if msg.type != "guidance.event":
            logger.warning("ignoring wrong-direction event %s from main_control", msg.type)
            return
        if not self._peer_role_allowed(msg):
            logger.warning(
                "ignoring guidance.event from untrusted producer role %r", msg.role
            )
            return
        try:
            await self.handler.guidance_event(msg)
        except asyncio.CancelledError:
            raise
        except ProtocolError as exc:
            logger.warning("state machine rejected guidance.event: %s", exc)
        except Exception:  # pragma: no cover - 状态机异常不得杀连接
            logger.exception("handler.guidance_event failed")

    def _peer_role_allowed(self, msg: Message) -> bool:
        return msg.role in self.allowed_peer_roles

    async def _invoke_handler(self, request: Message) -> Message | None:
        method = getattr(self.handler, HANDLER_METHODS.get(request.type, ""), None)
        if method is None:
            return self._error_reply(
                request,
                "DEPENDENCY_UNAVAILABLE",
                f"handler does not implement {request.type}",
                status="unavailable",
            )
        remaining_ms = request.deadline_mono_ms - now_ms()
        if remaining_ms <= 0:
            return self._error_reply(
                request,
                "DEADLINE_EXCEEDED",
                "request deadline already exceeded; execution not started",
                status="unavailable",
            )
        try:
            result = await asyncio.wait_for(method(request), timeout=remaining_ms / 1000.0)
        except asyncio.TimeoutError:
            return self._error_reply(
                request,
                "DEADLINE_EXCEEDED",
                f"handler did not finish within deadline ({remaining_ms} ms)",
                status="unavailable",
            )
        except HandlerFailure as exc:
            return self._error_reply(
                request, exc.code, exc.message, status=exc.status, retryable=exc.retryable
            )
        except ProtocolError as exc:
            return self._error_reply(request, exc.code, exc.message)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 处理器 bug 不得杀连接
            logger.exception("handler for %s crashed", request.type)
            return self._error_reply(
                request,
                "DEPENDENCY_UNAVAILABLE",
                f"handler crashed: {exc!r}"[:300],
                status="error",
            )
        return self._reply_from_handler(request, result)

    def _reply_from_handler(self, request: Message, result: Any) -> Message:
        try:
            if isinstance(result, HandlerReply):
                return response_for(
                    request,
                    data=result.data,
                    producer=self.producer,
                    status=result.status,
                    error=result.error,
                    clock_id=request.clock_id or self.clock_id,
                )
            if isinstance(result, Mapping):
                return response_for(
                    request,
                    data=result,
                    producer=self.producer,
                    clock_id=request.clock_id or self.clock_id,
                )
            raise TypeError(f"handler returned {type(result).__name__}, expected mapping or HandlerReply")
        except (ProtocolError, TypeError, ValueError) as exc:
            logger.exception("handler for %s returned invalid data", request.type)
            return self._error_reply(
                request,
                "DEPENDENCY_UNAVAILABLE",
                f"handler produced invalid response data: {exc}"[:300],
                status="error",
            )

    def _error_reply(
        self,
        request: Message,
        code: str,
        message: str | None = None,
        *,
        status: str | None = None,
        retryable: bool | None = None,
    ) -> Message:
        error = error_payload(code, message or code, retryable)
        try:
            return response_for(
                request,
                data=None,
                producer=self.producer,
                status=status or status_for_error_code(code),
                error=error,
                clock_id=request.clock_id or self.clock_id,
            )
        except (ProtocolError, ValueError) as exc:  # pragma: no cover - 防御
            logger.warning("failed to build error response for %s: %s", request.type, exc)
            return Message(
                type=REPLY_TYPE[request.type],
                message_id=new_message_id(),
                producer=self.producer,
                data=None,
                reply_to=request.message_id,
                request_id=request.request_id,
                session_id=request.session_id,
                clock_id=request.clock_id or self.clock_id,
                timestamp_mono_ms=now_ms(),
                evidence_mode=self.evidence_mode,
                status="error",
                error=error,
            )

    async def _on_invalid_message(self, raw: dict[str, Any], exc: ProtocolError) -> None:
        """能识别为请求的非法消息回合同错误码；否则只诊断。"""
        self.invalid_count += 1
        msg_type = raw.get("type")
        message_id = raw.get("message_id")
        request_id = raw.get("request_id")
        if msg_type not in REQUEST_TYPES:
            logger.warning("invalid message (not a known request type): %s", exc)
            return
        if not isinstance(message_id, str) or not message_id:
            logger.warning("invalid request without usable message_id: %s", exc)
            return
        if not isinstance(request_id, str) or not request_id:
            logger.warning("invalid request without request_id: %s", exc)
            return
        producer = raw.get("producer")
        role = producer.get("role") if isinstance(producer, Mapping) else None
        if role not in self.allowed_peer_roles:
            logger.warning("invalid request from untrusted producer role %r: %s", role, exc)
            return
        code = exc.code if exc.code in ("UNSUPPORTED_VERSION", "CLOCK_UNVERIFIED", "INVALID_PAYLOAD") else "INVALID_PAYLOAD"
        reply = Message(
            type=REPLY_TYPE[msg_type],
            message_id=new_message_id(),
            producer=self.producer,
            data=None,
            reply_to=message_id,
            request_id=request_id,
            session_id=raw.get("session_id") if isinstance(raw.get("session_id"), str) else None,
            clock_id=raw.get("clock_id") if isinstance(raw.get("clock_id"), str) else self.clock_id,
            timestamp_mono_ms=now_ms(),
            evidence_mode=(
                raw.get("evidence_mode")
                if raw.get("evidence_mode") in EVIDENCE_MODES
                else self.evidence_mode
            ),
            status="rejected",
            error=error_payload(code, f"invalid request: {exc.message}"[:500]),
        )
        await self._safe_send(reply)

    async def _on_unmatched_response(self, msg: Message) -> None:
        # 合同：不匹配/迟到响应只能诊断，不得推进当前交互
        logger.warning(
            "discarding unmatched or late response type=%s reply_to=%r request_id=%r",
            msg.type,
            msg.reply_to,
            msg.request_id,
        )

    async def _safe_send(self, msg: Message) -> None:
        conn = self._conn
        if conn is None or conn.closed:
            logger.warning("dropping %s: no connected main_control", msg.type)
            return
        try:
            await conn.send(msg)
        except (ConnectionClosed, FramingError) as exc:
            logger.warning("failed to send %s: %s", msg.type, exc)

    # --- 出站 -----------------------------------------------------------
    async def send_request(self, msg: Message, *, timeout_ms: int | None = None) -> Message:
        """发送请求并等待关联响应。

        ``timeout_ms`` 缺省时使用消息 ``deadline_mono_ms`` 剩余时间；
        返回 ``*.result``（可能 ``status != ok``，调用方必须检查）。
        超时抛 :class:`~aieyes.contract.transport.RequestTimeout`；未连接抛
        :class:`NotConnected`。
        """
        if not msg.is_request:
            raise ValueError(f"{msg.type} is not a request message")
        validate_message(msg)
        conn = self._conn
        if conn is None or conn.closed:
            raise NotConnected("main_control is not connected")
        if timeout_ms is None:
            timeout_ms = max(0, msg.deadline_mono_ms - now_ms())
        return await conn.request(msg, timeout_ms=timeout_ms)

    async def send_event(self, msg: Message) -> None:
        """发送主动事件；交互侧只允许 ``playback.status``。"""
        if msg.type != "playback.status":
            raise ProtocolError(
                "INVALID_PAYLOAD",
                f"interaction may only send playback.status events, got {msg.type}",
            )
        validate_message(msg)
        conn = self._conn
        if conn is None or conn.closed:
            raise NotConnected("main_control is not connected")
        await conn.send(msg)

    async def publish_playback_status(self, **kwargs: Any) -> None:
        """便捷转发：见 ``handlers.InteractionContract.publish_playback_status``。"""
        from aieyes.contract.handlers import InteractionContract

        await InteractionContract(self).publish_playback_status(**kwargs)


__all__ = ["ContractServer", "NotConnected", "read_host_boot_id"]
