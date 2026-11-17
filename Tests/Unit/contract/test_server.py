"""server：TCP 服务端分发、错误码、取消、事件、重连与来源策略。"""

from __future__ import annotations

import asyncio

import pytest

from aieyes.contract import handlers, schema, transport
from aieyes.contract.handlers import FakeHandler
from aieyes.contract.server import NotConnected
from aieyes.contract.fixture_client import FixtureClient

from contract_helpers import (
    CLOCK,
    INTERACTION,
    MAIN_CONTROL,
    Rig,
    raw_exchange,
    run,
    wait_until,
)


def _cap_request(**kwargs):
    kwargs.setdefault("clock_id", CLOCK)
    return schema.capabilities_get(producer=MAIN_CONTROL, **kwargs)


def test_null_handler_reports_missing_capabilities_honestly():
    async def main():
        async with Rig() as rig:
            server, client = rig.server, rig.client
            assert server is not None and client is not None

            capabilities = await client.capabilities_get()
            assert capabilities.ok
            data = capabilities.data
            assert data["capabilities"]["frame_provider"] is False
            assert data["capabilities"]["scene_state"] is False
            assert "frame.acquire" in data["unimplemented"]

            # interaction→main_control 的请求不能由主控夹具发回（方向守卫另有测试），
            # 这里直接调用 NullHandler 确认其诚实报告能力缺失。
            frame_request = schema.frame_acquire(
                mode="latest", purpose="vqa", producer=INTERACTION, clock_id=CLOCK
            )
            with pytest.raises(handlers.HandlerFailure) as excinfo:
                await server.handler.frame_acquire(frame_request)
            assert excinfo.value.code == "DEPENDENCY_UNAVAILABLE"

            with pytest.raises(handlers.HandlerFailure):
                await server.handler.scene_query(
                    schema.scene_query(producer=INTERACTION, clock_id=CLOCK)
                )

            control_request = schema.control_intent(
                action="start_guidance",
                utterance_id="utt-1",
                final_transcript="开始",
                producer=INTERACTION,
                clock_id=CLOCK,
            )
            control_data = await server.handler.control_intent(control_request)
            assert control_data["accepted"] is False
            assert control_data["operation_state"] == "rejected"

            # 主控→交互的入站请求在真实线路上由 NullHandler 应答
            cancel = await client.output_cancel(
                scope="ordinary_turn", reason="user_barge_in", target_request_id="req-old"
            )
            assert cancel.ok
            assert cancel.data["device_stopped"] is False
            assert cancel.data["queue_cleared"] is False
            assert cancel.data["compute_ended"] is False

            status = await client.interaction_status_get()
            assert status.ok
            assert status.data["mode"] == "degraded"
            assert set(status.data["services"]) == set(schema.SERVICE_NAMES)

    run(main())


def test_direction_guard_rejects_request_sent_by_main_control():
    async def main():
        async with Rig() as rig:
            client = rig.client
            response = await client.frame_acquire()
            assert response.status == "rejected"
            assert response.error_code() == "INVALID_PAYLOAD"
            assert "interaction→main_control" in response.error["message"]

    run(main())


def test_invalid_request_payload_gets_contract_error_reply():
    async def main():
        # 裸连接会被后续连接顶掉，因此不建立常驻夹具客户端
        async with Rig(connect=False) as rig:
            server = rig.server
            assert server is not None

            request = schema.output_cancel(
                scope="ordinary_turn",
                reason="user_barge_in",
                producer=MAIN_CONTROL,
                target_request_id="req-old",
                clock_id=CLOCK,
            )
            payload = request.to_dict()
            payload["data"]["scope"] = "not_a_scope"
            response = await raw_exchange(server.bound_port, payload)
            assert response is not None
            assert response["type"] == "output.cancel.result"
            assert response["status"] == "rejected"
            assert response["reply_to"] == request.message_id
            assert response["error"]["code"] == "INVALID_PAYLOAD"
            assert server.invalid_count >= 1

            # 未知版本：能识别为请求就回 UNSUPPORTED_VERSION
            bad_version = request.to_dict()
            bad_version["schema_version"] = "interaction.v9.9"
            response = await raw_exchange(server.bound_port, bad_version)
            assert response is not None
            assert response["error"]["code"] == "UNSUPPORTED_VERSION"

            # 无法关联的非法消息只做诊断（不回响应、不崩溃）
            malformed = schema.capabilities_get(producer=MAIN_CONTROL, clock_id=CLOCK).to_dict()
            malformed["message_id"] = 12345  # 类型错误 → 无法关联，仅诊断
            assert await raw_exchange(server.bound_port, malformed, timeout_s=0.5) is None

            # 之后的新连接仍能拿到合法响应
            valid = schema.capabilities_get(producer=MAIN_CONTROL, clock_id=CLOCK).to_dict()
            response = await raw_exchange(server.bound_port, valid)
            assert response is not None
            assert response["status"] == "ok"

    run(main())


def test_expired_deadline_returns_deadline_exceeded():
    async def main():
        async with Rig(handler=FakeHandler()) as rig:
            fake = rig.handler
            assert isinstance(fake, FakeHandler)
            fake.script(
                "capabilities.get",
                handlers.capabilities_result_data(capabilities={"frame_provider": True}),
            )
            expired = _cap_request(deadline_mono_ms=schema.now_ms() - 1000)
            response = await rig.client.request(expired, timeout_ms=2000)
            assert response.error_code() == "DEADLINE_EXCEEDED"
            assert response.status in ("unavailable", "rejected")
            assert fake.counts.get("capabilities.get", 0) == 0

    run(main())


def test_slow_handler_times_out_client_but_connection_survives():
    async def main():
        async with Rig(handler=FakeHandler(), client_timeout_ms=3000) as rig:
            fake = rig.handler
            assert isinstance(fake, FakeHandler)
            fake.script(
                "capabilities.get",
                handlers.capabilities_result_data(capabilities={"frame_provider": True}),
            )
            fake.delay("capabilities.get", 0.4)
            # 服务端的 deadline 看门狗与客户端的 wait_for 都按同一个 100ms 计时：
            # 回包可能先到（DEADLINE_EXCEEDED），也可能本地先超时。两种都合法，
            # 本用例要保证的是——请求不会拿到成功结果，且连接在超时后仍可用。
            try:
                reply = await rig.client.capabilities_get(timeout_ms=100)
            except transport.RequestTimeout:
                pass
            else:
                assert reply.status == "unavailable", reply
                assert reply.error_code() == "DEADLINE_EXCEEDED", reply

            fake.delay("capabilities.get", 0)
            response = await rig.client.capabilities_get(timeout_ms=2500)
            assert response.ok

    run(main())


def test_handler_failures_map_to_contract_error_codes():
    async def main():
        async with Rig(handler=FakeHandler()) as rig:
            fake = rig.handler
            assert isinstance(fake, FakeHandler)

            fake.fail("capabilities.get", "BUSY", "model is loading")
            response = await rig.client.capabilities_get()
            assert response.status == "unavailable"
            assert response.error_code() == "BUSY"
            assert response.error["retryable"] is True

            fake.fail("interaction.status.get", "DEPENDENCY_UNAVAILABLE", "status tap failed")
            response = await rig.client.interaction_status_get()
            assert response.status == "unavailable"
            assert response.error_code() == "DEPENDENCY_UNAVAILABLE"

            def explode(request):
                raise RuntimeError("unexpected handler bug")

            fake.reset()
            fake.script("output.cancel", explode)
            response = await rig.client.output_cancel(
                scope="ordinary_session", reason="user_barge_in"
            )
            assert response.status == "error"
            assert response.error_code() == "DEPENDENCY_UNAVAILABLE"

    run(main())


def test_cancel_and_playback_events_roundtrip():
    async def main():
        async with Rig(handler=FakeHandler()) as rig:
            server, client = rig.server, rig.client
            fake = rig.handler
            assert isinstance(fake, FakeHandler)
            fake.script(
                "output.cancel",
                handlers.cancel_result_data(
                    device_stopped=True,
                    queue_cleared=True,
                    compute_ended=False,
                    basis="device_write",
                ),
            )

            cancel = await client.output_cancel(
                scope="ordinary_turn",
                reason="user_barge_in",
                target_request_id="req-old",
                old_generation=3,
                new_generation=4,
            )
            assert cancel.ok
            assert cancel.data["device_stopped"] is True
            assert cancel.data["queue_cleared"] is True
            assert cancel.data["basis"] == "device_write"
            assert fake.counts["output.cancel"] == 1

            await server.publish_playback_status(
                state="queued",
                source_request_id="req-old",
                utterance_id="utt-1",
                segment_index=0,
            )
            queued = await client.expect_playback_status("queued")
            assert queued.producer["role"] == "interaction"
            assert queued.clock_id == server.clock_id
            assert queued.evidence_mode == server.evidence_mode

            await server.publish_playback_status(
                state="started",
                source_request_id="req-old",
                utterance_id="utt-1",
                segment_index=0,
                start_basis="device_write",
            )
            started = await client.expect_playback_status("started")
            assert schema.playback_key(started) == ("", "req-old", "utt-1", 0)
            assert started.data["start_basis"] == "device_write"
            assert started.data["state"] == "started"

    run(main())


def test_reconnect_increments_epoch_and_disconnect_callback():
    async def main():
        connect_epochs: list[int] = []
        disconnect_epochs: list[int] = []

        async with Rig(
            on_connect=lambda epoch: connect_epochs.append(epoch),
            on_disconnect=lambda epoch: disconnect_epochs.append(epoch),
        ) as rig:
            server, first = rig.server, rig.client

            second = FixtureClient("127.0.0.1", server.bound_port, timeout_ms=2000)
            await second.connect()
            try:
                assert await wait_until(
                    lambda: server.epoch >= 2 and server.connected, timeout_s=3.0
                )
                # 旧连接被新连接替换后，旧客户端最终看到关闭
                assert await wait_until(lambda: first.closed, timeout_s=3.0)
                response = await second.capabilities_get()
                assert response.ok
            finally:
                await second.close()

            assert await wait_until(lambda: not server.connected, timeout_s=3.0)
            assert await wait_until(lambda: disconnect_epochs == [1, 2], timeout_s=3.0)
            assert connect_epochs[:2] == [1, 2]

    run(main())


def test_test_fixture_role_gated_unless_enabled():
    async def main():
        async with Rig(connect=False) as rig:
            server = rig.server
            fixture = FixtureClient(
                "127.0.0.1", server.bound_port, producer_role="test_fixture", timeout_ms=2000
            )
            await fixture.connect()
            try:
                await server.wait_connected(3.0)
                response = await fixture.capabilities_get()
                assert response.status == "rejected"
                assert response.error_code() == "INVALID_PAYLOAD"
            finally:
                await fixture.close()

        # 显式测试入口才允许 test_fixture 扮演主控
        async with Rig(allow_test_fixture=True, connect=False) as rig:
            server = rig.server
            fixture = FixtureClient(
                "127.0.0.1", server.bound_port, producer_role="test_fixture", timeout_ms=2000
            )
            await fixture.connect()
            try:
                await server.wait_connected(3.0)
                response = await fixture.capabilities_get()
                assert response.ok
            finally:
                await fixture.close()

    run(main())


def test_send_requires_connection_and_event_direction_guard():
    async def main():
        async with Rig(connect=False) as rig:
            server = rig.server
            with pytest.raises(NotConnected):
                await server.send_request(_cap_request(), timeout_ms=500)
            with pytest.raises(NotConnected):
                await server.publish_playback_status(state="queued", source_request_id="req-1")

            # 交互不得主动发 guidance.event（方向保护）
            event = schema.guidance_event(
                event_id="evt-1",
                sequence=1,
                priority="information",
                issued_mono_ms=schema.now_ms(),
                valid_until_mono_ms=schema.now_ms() + 1000,
                state_id="scene-1",
                text="x",
                interrupt_policy="queue",
                producer=INTERACTION,
                clock_id=CLOCK,
            )
            with pytest.raises(schema.ProtocolError) as excinfo:
                await server.send_event(event)
            assert excinfo.value.code == "INVALID_PAYLOAD"

    run(main())
