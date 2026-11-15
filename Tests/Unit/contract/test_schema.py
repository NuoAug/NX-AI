"""interaction.v0.1 schema：构造、编解码、校验与语义辅助函数。"""

from __future__ import annotations

import dataclasses

import pytest

from aieyes.contract import errors, handlers, schema
from aieyes.contract.errors import ERROR_CODES, ProtocolError

from contract_helpers import CLOCK, INTERACTION, MAIN_CONTROL, run


def _frame_request(**kwargs):
    return schema.frame_acquire(
        mode="latest",
        purpose="vqa",
        producer=INTERACTION,
        clock_id=CLOCK,
        evidence_mode="synthetic",
        **kwargs,
    )


def _scene_request():
    return schema.scene_query(producer=INTERACTION, clock_id=CLOCK)


def _frame_data(**overrides):
    data = handlers.frame_result_data(
        frame_id="frame-1",
        captured_mono_ms=schema.now_ms(),
        width_px=1280,
        height_px=720,
        sha256="a" * 64,
        asset_id="asset-1",
        path="/tmp/main-control/frame-1.jpg",
    )
    data.update(overrides)
    return data


def _scene_data(**overrides):
    now = schema.now_ms()
    data = handlers.scene_result_data(
        state_id="scene-1",
        sequence=1,
        observed_mono_ms=now,
        valid_until_mono_ms=now + 1000,
    )
    data.update(overrides)
    return data


# ---------------------------------------------------------------------------
# 常量与注册表
# ---------------------------------------------------------------------------
def test_contract_constants():
    assert schema.SCHEMA_VERSION == "interaction.v0.1"
    assert schema.ROLES == ("main_control", "interaction", "test_fixture")
    assert set(schema.EVIDENCE_MODES) == {"live", "replay", "synthetic"}
    assert set(schema.RESPONSE_STATUSES) == {"ok", "unavailable", "rejected", "error"}
    assert len(ERROR_CODES) == 13
    assert len(set(ERROR_CODES)) == 13


def test_message_type_registry_covers_nine_groups():
    expected = set(schema.REPLY_TYPE) | set(schema.REPLY_TYPE.values()) | {
        "guidance.event",
        "playback.status",
    }
    assert schema.ALL_MESSAGE_TYPES == expected
    assert len(schema.REQUEST_TYPES) == 7
    assert len(schema.RESPONSE_TYPES) == 7
    assert len(schema.EVENT_TYPES) == 2
    # 16 条消息都要有 data 校验器，由 validate_message 统一调用。
    assert set(schema._DATA_VALIDATORS) == set(schema.ALL_MESSAGE_TYPES)


# ---------------------------------------------------------------------------
# 信封编解码
# ---------------------------------------------------------------------------
def test_request_roundtrip_preserves_envelope():
    request = _frame_request()
    assert request.is_request
    assert request.reply_to is None
    assert request.status is None and request.error is None
    assert request.deadline_mono_ms is not None and request.deadline_mono_ms > 0

    raw = request.to_dict()
    assert list(raw)[:3] == ["schema_version", "type", "message_id"]
    assert raw["schema_version"] == schema.SCHEMA_VERSION
    assert raw["evidence_mode"] == "synthetic"

    restored = schema.parse_message(raw)
    assert restored.to_dict() == raw
    assert schema.validate_message(restored) is restored


def test_parse_tolerates_extension_fields_but_strict_rejects():
    raw = dict(_frame_request().to_dict())
    raw["x_future_field"] = 1
    assert schema.parse_message(raw) is not None
    with pytest.raises(ProtocolError) as excinfo:
        schema.parse_message(raw, strict=True)
    assert excinfo.value.code == "INVALID_PAYLOAD"


def test_unknown_type_and_version_are_rejected():
    raw = dict(_frame_request().to_dict())
    with pytest.raises(ProtocolError) as excinfo:
        schema.parse_message(dict(raw, type="bogus.type"))
    assert excinfo.value.code == "INVALID_PAYLOAD"

    with pytest.raises(ProtocolError) as excinfo:
        schema.parse_message(dict(raw, schema_version="interaction.v9.9"))
    assert excinfo.value.code == "UNSUPPORTED_VERSION"


def test_request_with_response_fields_is_invalid():
    request = _frame_request()
    bad = schema.Message(
        type=request.type,
        message_id=request.message_id,
        producer=dict(INTERACTION),
        data=request.data,
        request_id=request.request_id,
        clock_id=CLOCK,
        deadline_mono_ms=request.deadline_mono_ms,
        status="ok",
    )
    with pytest.raises(ProtocolError) as excinfo:
        schema.validate_message(bad)
    assert excinfo.value.code == "INVALID_PAYLOAD"


def test_live_mode_requires_verified_clock_except_capabilities():
    with pytest.raises(ProtocolError) as excinfo:
        schema.control_intent(
            action="start_guidance",
            utterance_id="utt-1",
            final_transcript="开始",
            producer=INTERACTION,
            evidence_mode="live",
        )
    assert excinfo.value.code == "CLOCK_UNVERIFIED"

    # 首次握手允许 clock_id=null
    handshake = schema.capabilities_get(producer=INTERACTION, evidence_mode="live")
    assert handshake.clock_id is None


# ---------------------------------------------------------------------------
# 响应关联与错误对象
# ---------------------------------------------------------------------------
def test_response_for_links_request_and_copies_session():
    request = schema.scene_query(
        producer=INTERACTION, clock_id=CLOCK, session_id="sess-1", request_id="req-42"
    )
    response = schema.response_for(
        request,
        data=_scene_data(),
        producer=MAIN_CONTROL,
    )
    assert response.type == "scene.result"
    assert response.reply_to == request.message_id
    assert response.request_id == "req-42"
    assert response.session_id == "sess-1"
    assert response.clock_id == request.clock_id
    assert response.evidence_mode == request.evidence_mode
    assert response.ok
    schema.validate_message(response)


def test_response_for_requires_error_on_failure_and_data_on_success():
    request = _frame_request()
    # 失败响应缺 error 时合同层补 DEPENDENCY_UNAVAILABLE（不崩溃）
    fallback = schema.response_for(request, producer=MAIN_CONTROL, data=None, status="error")
    assert fallback.status == "error"
    assert fallback.error_code() == "DEPENDENCY_UNAVAILABLE"
    with pytest.raises(ProtocolError):
        schema.response_for(request, producer=MAIN_CONTROL, data=None, status="ok")

    response = schema.error_response(
        request, "NO_FRAME", "camera not ready", producer=MAIN_CONTROL
    )
    assert response.status == "error"
    assert response.error_code() == "NO_FRAME"
    assert response.error["retryable"] is True
    assert schema.parse_message(response.to_dict()).error_code() == "NO_FRAME"


def test_all_thirteen_error_codes_constructable_and_retryable_defaults():
    request = _frame_request()
    for code in ERROR_CODES:
        message = schema.error_response(request, code, producer=MAIN_CONTROL)
        assert message.error_code() == code
        expected = errors.RETRYABLE_DEFAULT[code]
        assert message.error["retryable"] is expected

    assert errors.RETRYABLE_DEFAULT["DEPENDENCY_UNAVAILABLE"] is True
    assert errors.RETRYABLE_DEFAULT["CANCELLED"] is False
    assert errors.RETRYABLE_DEFAULT["OUTPUT_UNAVAILABLE"] is False
    with pytest.raises(ValueError):
        errors.error_payload("NOT_A_CONTRACT_CODE", "boom")
    with pytest.raises(ValueError):
        errors.ProtocolError("NOT_A_CONTRACT_CODE", "boom")


# ---------------------------------------------------------------------------
# 各消息组 data 校验
# ---------------------------------------------------------------------------
def test_frame_acquire_validation_rules():
    with pytest.raises(ProtocolError) as excinfo:
        schema.frame_acquire(
            mode="pinned", purpose="vqa", producer=INTERACTION, clock_id=CLOCK
        )
    assert excinfo.value.code == "INVALID_FRAME_REF"

    with pytest.raises(ProtocolError) as excinfo:
        schema.frame_acquire(
            mode="latest",
            purpose="vqa",
            producer=INTERACTION,
            clock_id=CLOCK,
            roi_xyxy_px=(0, 0, 10, 10),
        )
    assert excinfo.value.code == "INVALID_FRAME_REF"

    for bad in (
        dict(mode="bogus", purpose="vqa"),
        dict(mode="latest", purpose="bogus"),
        dict(mode="latest", purpose="vqa", max_age_ms=-1),
    ):
        with pytest.raises(ProtocolError) as excinfo:
            schema.frame_acquire(producer=INTERACTION, clock_id=CLOCK, **bad)
        assert excinfo.value.code == "INVALID_PAYLOAD"


def test_frame_result_asset_and_captured_rules():
    request = _frame_request()
    valid = schema.frame_result(request, producer=MAIN_CONTROL, data=_frame_data())
    assert valid.ok
    schema.validate_message(valid)

    for mutate in (
        lambda data: data["asset"].__setitem__("sha256", "abc"),
        lambda data: data["asset"].__setitem__("transport", "http"),
        lambda data: data.update(captured_mono_ms=None),
    ):
        data = _frame_data()
        mutate(data)
        with pytest.raises(ProtocolError) as excinfo:
            schema.frame_result(request, producer=MAIN_CONTROL, data=data)
        assert excinfo.value.code == "INVALID_PAYLOAD"


def test_derived_frame_keeps_parent_and_transform():
    request = schema.frame_acquire(
        mode="pinned",
        purpose="ocr",
        producer=INTERACTION,
        clock_id=CLOCK,
        frame_id="frame-1",
    )
    data = _frame_data(
        parent_frame_id="frame-0",
        transform_to_parent=[1, 0, 0, 0, 1, 0, 0, 0, 1],
    )
    result = schema.frame_result(request, producer=MAIN_CONTROL, data=data)
    assert result.data["parent_frame_id"] == "frame-0"
    assert len(result.data["transform_to_parent"]) == 9


def test_scene_result_invalid_measurement_must_have_null_distance():
    request = _scene_request()
    measurement = handlers.measurement_data(
        distance_m=3.0, valid=False, reason_code="DEPTH_INVALID"
    )
    data = _scene_data(
        objects=[
            handlers.scene_object_data(
                track_id="track-1", class_name="person", measurement=measurement
            )
        ]
    )
    with pytest.raises(ProtocolError) as excinfo:
        schema.scene_result(request, producer=MAIN_CONTROL, data=data)
    assert excinfo.value.code == "INVALID_PAYLOAD"

    ok_measurement = handlers.measurement_data(
        distance_m=None, valid=False, reason_code="DEPTH_INVALID"
    )
    ok_data = _scene_data(
        objects=[
            handlers.scene_object_data(
                track_id="track-1", class_name="person", measurement=ok_measurement
            )
        ]
    )
    result = schema.scene_result(request, producer=MAIN_CONTROL, data=ok_data)
    assert result.ok


def test_control_result_state_consistency():
    request = schema.control_intent(
        action="start_guidance",
        utterance_id="utt-1",
        final_transcript="开始",
        producer=INTERACTION,
        clock_id=CLOCK,
    )
    with pytest.raises(ProtocolError):
        schema.control_result(
            request,
            producer=MAIN_CONTROL,
            data=handlers.control_result_data(accepted=False, operation_state="completed"),
        )
    with pytest.raises(ProtocolError):
        schema.control_result(
            request,
            producer=MAIN_CONTROL,
            data=handlers.control_result_data(accepted=True, operation_state="rejected"),
        )

    good = schema.control_result(
        request,
        producer=MAIN_CONTROL,
        data=handlers.control_result_data(
            accepted=True, operation_state="completed", actual_mode="guidance"
        ),
    )
    assert schema.is_completed_ok(good, expected_mode="guidance")
    assert not schema.is_completed_ok(good, expected_mode="idle")
    pending = schema.control_result(
        request,
        producer=MAIN_CONTROL,
        data=handlers.control_result_data(
            accepted=True, operation_state="pending", actual_mode="guidance"
        ),
    )
    assert not schema.is_completed_ok(pending)


def test_output_cancel_scope_rules():
    valid_turn = schema.output_cancel(
        scope="ordinary_turn",
        reason="user_barge_in",
        producer=MAIN_CONTROL,
        target_request_id="req-1",
        old_generation=1,
        new_generation=2,
        clock_id=CLOCK,
    )
    assert valid_turn.is_request

    with pytest.raises(ProtocolError) as excinfo:
        schema.output_cancel(
            scope="ordinary_turn", reason="user_barge_in", producer=MAIN_CONTROL, clock_id=CLOCK
        )
    assert excinfo.value.code == "INVALID_PAYLOAD"

    with pytest.raises(ProtocolError):
        schema.output_cancel(
            scope="guidance_event",
            reason="user_barge_in",
            producer=MAIN_CONTROL,
            target_event_id="evt-1",
            target_request_id="req-1",
            clock_id=CLOCK,
        )

    with pytest.raises(ProtocolError):
        schema.output_cancel(
            scope="ordinary_session",
            reason="user_barge_in",
            producer=MAIN_CONTROL,
            old_generation=5,
            new_generation=4,
            clock_id=CLOCK,
        )

    valid_guidance = schema.output_cancel(
        scope="guidance_event",
        reason="user_barge_in",
        producer=MAIN_CONTROL,
        target_event_id="evt-1",
        clock_id=CLOCK,
    )
    assert valid_guidance.data["target_event_id"] == "evt-1"


def test_playback_status_and_guidance_event_rules():
    now = schema.now_ms()
    started = schema.playback_status(
        state="started",
        occurred_mono_ms=now,
        producer=INTERACTION,
        source_event_id="evt-1",
        priority="emergency",
        start_basis="device_write",
        clock_id=CLOCK,
    )
    assert schema.playback_key(started) == ("evt-1", "", "", 0)

    with pytest.raises(ProtocolError):
        schema.playback_status(
            state="started",
            occurred_mono_ms=now,
            producer=INTERACTION,
            source_event_id="evt-1",
            priority="emergency",
            clock_id=CLOCK,
        )

    with pytest.raises(ProtocolError):
        schema.playback_status(
            state="queued",
            occurred_mono_ms=now,
            producer=INTERACTION,
            source_event_id="evt-1",
            clock_id=CLOCK,
        )

    event = schema.guidance_event(
        event_id="evt-1",
        sequence=1,
        priority="emergency",
        issued_mono_ms=now,
        valid_until_mono_ms=now + 2000,
        state_id="scene-1",
        text="前方有障碍",
        interrupt_policy="immediate",
        producer=MAIN_CONTROL,
        clock_id=CLOCK,
    )
    assert schema.guidance_is_fresh(event)
    assert not schema.guidance_is_fresh(event, now_ms=now + 3000)


def test_interaction_status_minimum_fields():
    request = schema.interaction_status_get(producer=MAIN_CONTROL, clock_id=CLOCK)
    status = schema.interaction_status(
        request, producer=INTERACTION, data=handlers.interaction_status_data()
    )
    data = status.data
    for key in (
        "mode",
        "active_request_id",
        "audio_input_state",
        "audio_output_state",
        "services",
        "queues",
        "memory",
        "metrics_window",
        "last_error",
    ):
        assert key in data
    assert set(data["services"]) == set(schema.SERVICE_NAMES)
    assert data["mode"] == "degraded"
    schema.validate_message(status)


# ---------------------------------------------------------------------------
# 语义辅助：距离、序号、去重、握手指纹
# ---------------------------------------------------------------------------
def test_measurement_helpers_never_fabricate_distance():
    now = schema.now_ms()
    valid = handlers.measurement_data(
        distance_m=2.5, valid=True, valid_until_mono_ms=now + 1000
    )
    assert schema.measurement_distance_m(valid, now_ms=now) == 2.5
    assert schema.measurement_distance_m(valid, now_ms=now + 2000) is None
    assert schema.measurement_distance_m(handlers.measurement_data()) is None

    request = _scene_request()
    under_scene = _scene_data(
        objects=[
            handlers.scene_object_data(
                track_id="track-1",
                class_name="person",
                measurement=handlers.measurement_data(
                    distance_m=2.5, valid=True, valid_until_mono_ms=now + 1000
                ),
            )
        ]
    )
    result = schema.scene_result(request, producer=MAIN_CONTROL, data=under_scene)
    assert schema.scene_measurement(result, track_id="track-1", now_ms=now) == 2.5
    assert schema.scene_measurement(result, track_id="track-9", now_ms=now) is None
    # 场景本身过期也不得转述距离
    expired_scene = schema.scene_result(
        request,
        producer=MAIN_CONTROL,
        data=handlers.scene_result_data(
            state_id="scene-1",
            sequence=1,
            observed_mono_ms=now - 2000,
            valid_until_mono_ms=now - 1,
            objects=[
                handlers.scene_object_data(
                    track_id="track-1",
                    class_name="person",
                    measurement=handlers.measurement_data(
                        distance_m=2.5, valid=True, valid_until_mono_ms=now + 5000
                    ),
                )
            ],
        ),
    )
    assert schema.scene_measurement(expired_scene, now_ms=now) is None


def test_sequence_guard_detects_regression():
    guard = schema.SequenceGuard()
    assert guard.check("mc-1:scene", 5) is True
    assert guard.check("mc-1:scene", 5) is False
    with pytest.raises(ProtocolError) as excinfo:
        guard.check("mc-1:scene", 3)
    assert excinfo.value.code == "STALE_STATE"
    # 回退不更新 last（仍保留已接受的最大序号）
    assert guard.last("mc-1:scene") == 5
    guard.reset("mc-1:scene")
    assert guard.check("mc-1:scene", 1) is True
    with pytest.raises(ProtocolError):
        guard.check("mc-1:scene", -1)


def test_guidance_deduper_dedupes_expires_and_supersedes():
    now = schema.now_ms()

    def event(
        event_id: str,
        sequence: int,
        *,
        valid_until_ms: int = now + 5000,
        supersedes=None,
        issued_ms: int | None = None,
    ):
        return schema.guidance_event(
            event_id=event_id,
            sequence=sequence,
            priority="navigation",
            issued_mono_ms=now if issued_ms is None else issued_ms,
            valid_until_mono_ms=valid_until_ms,
            state_id="scene-1",
            text="前方左转",
            interrupt_policy="queue",
            producer=MAIN_CONTROL,
            supersedes_event_id=supersedes,
            clock_id=CLOCK,
        )

    deduper = schema.GuidanceDeduper()
    first = event("evt-1", 1)
    assert deduper.accept(first, now_ms=now) == "play"
    assert deduper.accept(first, now_ms=now) == "drop_duplicate"
    assert deduper.accept(event("evt-2", 2), now_ms=now) == "play"
    # 被指定替代的事件不再播放，即使此前没收到过
    assert deduper.accept(event("evt-3", 3, supersedes="evt-2"), now_ms=now) == "play"
    assert deduper.accept(event("evt-2", 2), now_ms=now) == "drop_superseded"
    # 过期事件不得补播（签发与有效期都在过去）
    assert (
        deduper.accept(
            event("evt-4", 4, issued_ms=now - 10_000, valid_until_ms=now - 1), now_ms=now
        )
        == "drop_stale"
    )


def test_handshake_fingerprint_changes_trigger_rehandshake():
    request = schema.capabilities_get(producer=INTERACTION, clock_id=CLOCK)
    response = schema.capabilities_result(
        request,
        producer=MAIN_CONTROL,
        capabilities={"frame_provider": True},
        host_boot_id="boot-1",
    )
    assert schema.handshake_fingerprint(response) == (
        MAIN_CONTROL["instance_id"],
        "boot-1",
        CLOCK,
    )
    restarted = schema.capabilities_result(
        request,
        producer=MAIN_CONTROL,
        capabilities={"frame_provider": True},
        host_boot_id="boot-2",
    )
    assert schema.handshake_fingerprint(restarted) != schema.handshake_fingerprint(response)

    with pytest.raises(ProtocolError):
        schema.handshake_fingerprint(_scene_request())
