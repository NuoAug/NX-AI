import time
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import Mock, AsyncMock

from aieyes.intent import IntentKind
import intent_adapter
from worker import RunSession, Worker
from server import BenchServer


def test_negated_mic_with_real_project_classifier():
    for text in ("别关麦克风", "不要关麦克风", "别开麦克风", "别打开麦克风"):
        assert intent_adapter.classify(text).kind == IntentKind.CHAT
        assert intent_adapter.control_feedback(text) == "好，我不更改麦克风设置。"
    assert intent_adapter.control_feedback("怎么关闭麦克风") is None


def test_run_stop_status_and_cancel_event(tmp_path):
    worker = Worker()
    session = RunSession(worker, "test-run", tmp_path)
    assert session._status() == "completed"
    session.request_stop("user")
    assert session._model_cancel.is_set()
    assert session._status() == "interrupted"


def test_finished_run_does_not_schedule_process_exit(monkeypatch):
    worker = Worker()
    worker.run = SimpleNamespace(run_id="previous", started_iso="test")
    worker.release_preview_ref = Mock()
    worker.emit_state = Mock()
    worker.log = Mock()
    worker.devices_patch = Mock(return_value={})
    worker._schedule_exit = Mock(side_effect=AssertionError("unexpected exit"))
    worker.on_run_finished(worker.run, "interrupted", [])
    assert worker.run is None
    time.sleep(0.25)
    worker._schedule_exit.assert_not_called()


def test_stop_during_assembly_skips_all_starts(tmp_path):
    session = RunSession(Worker(), "test-run", tmp_path)
    session._cleanup = AsyncMock()
    session.deps = SimpleNamespace(sink=Mock(), frontend=Mock(), contract=None)
    session.engine = SimpleNamespace(start=AsyncMock())

    def assemble_then_stop():
        session.request_stop("user")

    session._assemble = assemble_then_stop
    asyncio.run(session._main())
    session.deps.sink.start.assert_not_called()
    session.engine.start.assert_not_awaited()
    session._cleanup.assert_awaited_once()


def test_stop_before_loop_creation_skips_assembly(tmp_path):
    session = RunSession(Worker(), "test-run", tmp_path)
    session._cleanup = AsyncMock()
    session._assemble = Mock()
    session.request_stop("user")
    asyncio.run(session._main())
    session._assemble.assert_not_called()
    session._cleanup.assert_awaited_once()


def test_stop_during_sink_start_skips_engine(tmp_path):
    session = RunSession(Worker(), "test-run", tmp_path)
    session._cleanup = AsyncMock()
    session.deps = SimpleNamespace(sink=Mock(), frontend=Mock(), contract=None)
    session.engine = SimpleNamespace(start=AsyncMock())
    session._assemble = Mock()
    session.deps.sink.start.side_effect = lambda: session.request_stop("user")
    asyncio.run(session._main())
    session.engine.start.assert_not_awaited()
    session._cleanup.assert_awaited_once()


def test_idle_watchdog_cannot_stop_new_active_run():
    server = object.__new__(BenchServer)
    server.lock = threading.RLock()
    server._run_active = True
    server._debug_active = False
    server._preview_active = False
    server._worker = SimpleNamespace(alive=True, stop=Mock())
    server.stop_worker("idle_watchdog")
    server._worker.stop.assert_not_called()


def test_late_start_error_cannot_replace_finished_run():
    server = object.__new__(BenchServer)
    server.lock = threading.RLock()
    server._run_id = "run-1"
    server._run_active = False
    server.state = {"run": {"id": "run-1", "status": "interrupted"}}
    server.append_event = Mock()
    server.stop_worker = Mock()
    server.write_log = Mock()
    server._on_run_start_done("run-1", None, {"code": "worker_exited", "message": "late"})
    assert server.state["run"]["status"] == "interrupted"
    server.append_event.assert_not_called()
    server.stop_worker.assert_not_called()
