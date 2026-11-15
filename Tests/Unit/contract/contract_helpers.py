"""contract 单测共享工具。

目标板未安装 ``pytest-asyncio``（也不要为此新增依赖），因此异步用例统一用
``asyncio.run`` 在同步测试函数里执行。
"""

from __future__ import annotations

import asyncio
import json
import struct
from typing import Any, Callable, Mapping

from aieyes.config import ContractConfig
from aieyes.contract import ContractServer, FixtureClient, transport

INTERACTION = {"role": "interaction", "instance_id": "aieyes-contract-test"}
MAIN_CONTROL = {"role": "main_control", "instance_id": "mc-contract-test"}
TEST_FIXTURE = {"role": "test_fixture", "instance_id": "fixture-contract-test"}
CLOCK = "test-monotonic-boot"

#: 帧格式与 transport 保持一致：4 字节大端长度 + JSON 字节
MAX_MESSAGE_BYTES = 4 * 1024 * 1024


def run(coro):
    """在同步测试里执行协程。"""
    return asyncio.run(coro)


def make_config(**overrides: Any) -> ContractConfig:
    values: dict[str, Any] = {
        "host": "127.0.0.1",
        "port": 0,
        "role": "interaction",
        "instance_id": "aieyes-contract-test",
        "response_timeout_ms": 3000,
        "accept_timeout_s": 3.0,
    }
    values.update(overrides)
    return ContractConfig(**values)


class Rig:
    """启动一个临时端口 ContractServer，并按需接入一个主控夹具客户端。"""

    def __init__(
        self,
        *,
        handler: Any = None,
        allow_test_fixture: bool = False,
        evidence_mode: str = "synthetic",
        connect: bool = True,
        on_connect: Any = None,
        on_disconnect: Any = None,
        client_timeout_ms: int = 3000,
    ) -> None:
        self.handler = handler
        self.allow_test_fixture = allow_test_fixture
        self.evidence_mode = evidence_mode
        self.connect_client = connect
        self.on_connect = on_connect
        self.on_disconnect = on_disconnect
        self.client_timeout_ms = client_timeout_ms
        self.server: ContractServer | None = None
        self.client: FixtureClient | None = None

    async def __aenter__(self) -> "Rig":
        self.server = ContractServer(
            make_config(),
            handler=self.handler,
            evidence_mode=self.evidence_mode,
            allow_test_fixture=self.allow_test_fixture,
            on_connect=self.on_connect,
            on_disconnect=self.on_disconnect,
        )
        await self.server.start()
        if self.connect_client:
            self.client = FixtureClient(
                "127.0.0.1",
                self.server.bound_port,
                timeout_ms=self.client_timeout_ms,
            )
            await self.client.connect()
            await self.server.wait_connected(3.0)
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self.client is not None:
            await self.client.close()
        if self.server is not None:
            await self.server.stop()


async def raw_exchange(
    port: int,
    payload: Mapping[str, Any],
    *,
    timeout_s: float = 3.0,
) -> dict[str, Any] | None:
    """用裸 TCP 发一帧合同 JSON，读回一帧（没有响应/连接被关闭时返回 None）。"""
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        writer.write(struct.pack(">I", len(body)) + body)
        await writer.drain()
        try:
            frame = await asyncio.wait_for(
                transport.read_frame(reader, max_message_bytes=MAX_MESSAGE_BYTES),
                timeout_s,
            )
        except (asyncio.TimeoutError, asyncio.IncompleteReadError, transport.FramingError):
            return None
        if frame is None:
            return None
        return transport.decode_payload(frame)
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # pragma: no cover - 对端可能已关闭
            pass


async def wait_until(
    predicate: Callable[[], bool],
    *,
    timeout_s: float = 3.0,
    interval_s: float = 0.01,
) -> bool:
    """轮询等待某个本地状态变为真（用于 accept/断线等异步副作用）。"""
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout_s
    while loop.time() < deadline:
        if predicate():
            return True
        await asyncio.sleep(interval_s)
    return predicate()
