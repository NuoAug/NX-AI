"""transport：分帧、请求关联、超时、断线、非法消息诊断。"""

from __future__ import annotations

import asyncio

import pytest

from aieyes.contract import handlers, schema, transport

from contract_helpers import CLOCK, INTERACTION, MAIN_CONTROL, run

MAX = 4 * 1024 * 1024


async def _pair(on_message):
    """建一对已完成分帧握手的 Connection（服务端 + 客户端）。"""
    state: dict[str, transport.Connection] = {}
    accepted = asyncio.Event()

    async def handle(reader, writer):
        conn = transport.Connection(reader, writer, on_message=on_message, name="server")
        state["server"] = conn
        conn.start()
        accepted.set()
        await conn.wait_closed()

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    client = transport.Connection(reader, writer, name="client")
    client.start()
    await asyncio.wait_for(accepted.wait(), 3.0)
    return server, client, state["server"]


async def _close_all(server, *connections):
    for conn in connections:
        await conn.close("test done")
    server.close()
    await server.wait_closed()


def test_read_frame_roundtrip_and_clean_eof():
    async def main():
        reader = asyncio.StreamReader()
        reader.feed_data(transport.encode_header(7) + b'{"a":1}')
        reader.feed_eof()
        assert await transport.read_frame(reader, max_message_bytes=MAX) == b'{"a":1}'
        assert await transport.read_frame(reader, max_message_bytes=MAX) is None

    run(main())


def test_read_frame_rejects_bad_lengths_and_truncation():
    async def main():
        for header in (b"\x00\x00\x00\x00", b"\xff\xff\xff\xff"):
            reader = asyncio.StreamReader()
            reader.feed_data(header)
            reader.feed_eof()
            with pytest.raises(transport.FramingError):
                await transport.read_frame(reader, max_message_bytes=1024)

        truncated_header = asyncio.StreamReader()
        truncated_header.feed_data(b"\x00\x00")
        truncated_header.feed_eof()
        with pytest.raises(transport.FramingError):
            await transport.read_frame(truncated_header, max_message_bytes=1024)

        truncated_payload = asyncio.StreamReader()
        truncated_payload.feed_data(transport.encode_header(10) + b"{}")
        truncated_payload.feed_eof()
        with pytest.raises(transport.FramingError):
            await transport.read_frame(truncated_payload, max_message_bytes=1024)

    run(main())


def test_decode_payload_rejects_bad_json_and_non_object():
    with pytest.raises(transport.FramingError):
        transport.decode_payload(b"{not json")
    with pytest.raises(transport.FramingError):
        transport.decode_payload(b"[1, 2, 3]")
    assert transport.decode_payload(b'{"x": 1}') == {"x": 1}


def test_request_response_roundtrip_with_correlation():
    state: dict[str, transport.Connection] = {}

    async def on_message(msg):
        if not msg.is_request:
            return
        reply = schema.response_for(
            msg,
            data=handlers.capabilities_result_data(capabilities={"frame_provider": True}),
            producer=MAIN_CONTROL,
        )
        await state["server"].send(reply)

    async def main():
        server, client, server_conn = await _pair(on_message)
        state["server"] = server_conn
        try:
            request = schema.capabilities_get(producer=INTERACTION, clock_id=CLOCK)
            response = await client.request(request, timeout_ms=2000)
            assert response.type == "capabilities.result"
            assert response.reply_to == request.message_id
            assert response.request_id == request.request_id
            assert response.status == "ok"
            assert response.data["capabilities"]["frame_provider"] is True
            assert server_conn.received_count == 1
            assert client.sent_count == 1
            assert client.pending_count == 0
        finally:
            await _close_all(server, client, server_conn)

    run(main())


def test_request_timeout_leaves_no_pending_request():
    async def on_message(msg):
        # 故意不回响应，验证 T15 超时路径
        return None

    async def main():
        server, client, server_conn = await _pair(on_message)
        try:
            request = schema.capabilities_get(producer=INTERACTION, clock_id=CLOCK)
            with pytest.raises(transport.RequestTimeout):
                await client.request(request, timeout_ms=100)
            assert client.pending_count == 0
            # 超时后连接仍可用
            assert not client.closed
        finally:
            await _close_all(server, client, server_conn)

    run(main())


def test_peer_disconnect_fails_pending_request():
    state: dict[str, transport.Connection] = {}

    async def on_message(msg):
        # 收到请求不回复，由服务端连接主动断开
        if msg.is_request:
            asyncio.get_running_loop().create_task(
                state["server"].close("simulated peer disconnect")
            )

    async def main():
        server, client, server_conn = await _pair(on_message)
        state["server"] = server_conn
        try:
            request = schema.capabilities_get(producer=INTERACTION, clock_id=CLOCK)
            with pytest.raises(transport.ConnectionClosed):
                await client.request(request, timeout_ms=2000)
            await asyncio.wait_for(client.wait_closed(), 3.0)
            assert client.closed
        finally:
            await _close_all(server, client, server_conn)

    run(main())


def test_malformed_frame_closes_connection_and_calls_on_closed():
    reasons: list[str] = []

    async def main():
        peer_closed = asyncio.Event()

        async def handle(reader, writer):
            conn = transport.Connection(
                reader,
                writer,
                on_message=lambda msg: None,
                on_closed=peer_closed.set,
                name="server",
            )
            conn.start()
            await conn.wait_closed()
            reasons.append(conn.close_reason)

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        try:
            # 0 长度帧头：合同本层判定为非法帧，关闭连接
            writer.write(b"\x00\x00\x00\x00")
            await writer.drain()
            await asyncio.wait_for(peer_closed.wait(), 3.0)
            assert reasons and reasons[0].startswith("framing error")
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # pragma: no cover
                pass
            server.close()
            await server.wait_closed()

    run(main())


def test_unmatched_and_late_responses_only_logged():
    seen: list[str] = []

    async def on_unmatched(msg):
        seen.append(msg.reply_to or "")

    async def main():
        server_conn_holder: dict[str, transport.Connection] = {}

        async def handle(reader, writer):
            conn = transport.Connection(
                reader,
                writer,
                on_message=lambda msg: None,
                on_unmatched_response=on_unmatched,
                name="server",
            )
            server_conn_holder["conn"] = conn
            conn.start()
            await conn.wait_closed()

        server = await asyncio.start_server(handle, "127.0.0.1", 0)
        port = server.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        client = transport.Connection(reader, writer, name="client")
        client.start()
        try:
            await asyncio.sleep(0.05)
            orphan = schema.response_for(
                schema.capabilities_get(producer=INTERACTION, clock_id=CLOCK),
                data=handlers.capabilities_result_data(capabilities={"frame_provider": True}),
                producer=MAIN_CONTROL,
            )
            await client.send(orphan)
            for _ in range(50):
                if seen:
                    break
                await asyncio.sleep(0.02)
            assert seen == [orphan.reply_to]
            assert not client.closed
        finally:
            await client.close("done")
            await server_conn_holder["conn"].close("done")
            server.close()
            await server.wait_closed()

    run(main())


def test_encode_message_is_length_prefixed_json():
    request = schema.capabilities_get(producer=INTERACTION, clock_id=CLOCK)
    frame = transport.encode_message(request)
    size = int.from_bytes(frame[:4], "big")
    assert size == len(frame) - 4
    decoded = transport.decode_payload(frame[4:])
    assert decoded["type"] == "capabilities.get"
    assert decoded["message_id"] == request.message_id
    assert transport.encode_header(3) == b"\x00\x00\x00\x03"
    with pytest.raises(transport.FramingError):
        transport.encode_header(0)
