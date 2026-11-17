"""A 组定向用例：取消/停止/立即重试/worker 生命周期（清单 23/24/25/26/30/31/43）。

只用假 worker 与临时目录，不启动真实业务组件、不触碰设备或模型服务。
每个用例对应一条已复现缺陷或其复验契约；修复前后运行同一命令即可对比。
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import Mock

from .test_server import ServerTestBase, FakeWorker


class _RacingLock:
    """在第 N 次进入临界区时注入一次并发事件，用于稳定复现锁内外的竞态。"""

    def __init__(self, inner, trigger_at: int, hook) -> None:
        self._inner = inner
        self._trigger_at = trigger_at
        self._hook = hook
        self.count = 0

    def __enter__(self):
        self._inner.acquire()
        self.count += 1
        if self.count == self._trigger_at and self._hook is not None:
            hook, self._hook = self._hook, None
            hook()
        return self

    def __exit__(self, *exc_info):
        self._inner.release()
        return False


class LateCallbackContractTests(ServerTestBase):
    """清单 43：迟到回调不得覆盖本轮已结束状态。"""

    def test_late_start_error_cannot_overwrite_state_finalized_mid_callback(self) -> None:
        run = self.ctx.start_run("live")
        run_id = run["run_id"]
        self.ctx.append_event = Mock()
        self.ctx.stop_worker = Mock()
        self.ctx._watch_worker_exit = Mock()

        def finish_during_callback() -> None:
            # 模拟另一线程在本回调的两次加锁之间完成收尾（取消 → interrupted）。
            self.ctx._run_active = False
            self.ctx.state["run"] = {
                **(self.ctx.state.get("run") or {}),
                "id": run_id,
                "status": "interrupted",
            }

        self.ctx.lock = _RacingLock(threading.RLock(), 2, finish_during_callback)
        self.ctx._on_run_start_done(run_id, None, {"code": "worker_exited", "message": "迟到的启动错误"})
        self.assertEqual(self.ctx.state["run"]["status"], "interrupted")
        self.ctx.append_event.assert_not_called()
        self.ctx.stop_worker.assert_not_called()

    def test_late_debug_start_error_cannot_overwrite_finished_debug(self) -> None:
        debug_id = "dbg-late"
        with self.ctx.lock:
            self.ctx._debug_run_id = debug_id
            self.ctx._debug_active = False
            self.ctx.state["debug"] = {
                "run_id": debug_id,
                "module": "chat",
                "status": "interrupted",
                "started_at": "2026-11-15T00:00:00+08:00",
            }
        self.ctx.append_event = Mock()
        self.ctx.stop_worker = Mock()
        self.ctx._on_debug_start_done(debug_id, None, {"code": "worker_exited", "message": "迟到的启动错误"})
        self.assertEqual(self.ctx.state["debug"]["status"], "interrupted")
        self.ctx.stop_worker.assert_not_called()

    def test_stale_state_patch_from_previous_run_cannot_end_new_run(self) -> None:
        first = self.ctx.start_run("live")
        first_id = first["run_id"]
        self.ctx._watch_worker_exit = Mock()
        self.ctx._finalize_run("interrupted")
        self.assertEqual(self.ctx.state["run"]["status"], "interrupted")

        second = self.ctx.start_run("live")
        second_id = second["run_id"]
        self.ctx._handle_state_patch({"run": {"id": first_id, "status": "interrupted"}})

        self.assertTrue(self.ctx._run_active, "上一轮的迟到状态补丁结束掉了新一轮")
        self.assertEqual(self.ctx.state["run"]["id"], second_id)
        self.assertEqual(self.ctx.state["run"]["status"], "starting")

    def test_stale_state_patch_cannot_change_finished_run_status(self) -> None:
        run = self.ctx.start_run("live")
        run_id = run["run_id"]
        self.ctx._watch_worker_exit = Mock()
        self.ctx._finalize_run("interrupted")
        self.ctx._handle_state_patch({"run": {"id": run_id, "status": "completed"}})
        self.assertEqual(self.ctx.state["run"]["status"], "interrupted")


class RetryReclaimContractTests(ServerTestBase):
    """清单 25/26/31：立即重试与空闲回收的占用语义。"""

    def test_retry_start_after_failure_is_not_killed_by_delayed_reclaim(self) -> None:
        worker = FakeWorker(self.ctx)
        self.ctx._worker = worker
        self.ctx._run_id = "run-failed"
        self.ctx._run_active = False
        self.ctx.state["run"] = {"id": "run-failed", "status": "failed"}
        # 立即重试：新一轮已占用同一个存活的 worker。
        self.ctx.start_run("live")
        active_worker = self.ctx._worker
        self.ctx.stop_worker("run_start_failed")
        self.assertIs(self.ctx._worker, active_worker, "上一轮失败的延迟回收停掉了新一轮的 worker")
        self.assertTrue(active_worker.alive)

    def test_idle_watchdog_cannot_stop_worker_of_run_just_started(self) -> None:
        first = self.ctx.start_run("live")
        self.ctx._watch_worker_exit = Mock()
        self.ctx._finalize_run("interrupted")
        worker = self.ctx._worker
        # 立即重试：五秒空闲回收窗口内开始的新一轮必须保留 worker。
        second = self.ctx.start_run("live")
        self.assertNotEqual(first["run_id"], second["run_id"])
        self.ctx.stop_worker("idle_watchdog")
        self.assertIs(self.ctx._worker, worker)
        self.assertTrue(worker.alive)

    def test_stop_then_retry_is_reported_busy_until_run_finalizes(self) -> None:
        run = self.ctx.start_run("live")
        self.ctx.stop_run(run["run_id"])
        self.assertEqual(self.ctx.state["run"]["status"], "stopping")
        with self.assertRaises(Exception) as caught:
            self.ctx.start_run("live")
        self.assertEqual(getattr(caught.exception, "status", None), 409)
        self.ctx._handle_state_patch({"run": {"id": run["run_id"], "status": "interrupted"}})
        self.assertFalse(self.ctx._run_active)
        retry = self.ctx.start_run("live")
        self.assertEqual(retry["status"], "starting")


class CancelStatusContractTests(ServerTestBase):
    """清单 24：取消如实记 interrupted。"""

    def test_cancelled_run_is_recorded_interrupted_not_completed(self) -> None:
        run = self.ctx.start_run("live")
        self.ctx._watch_worker_exit = Mock()
        self.ctx.stop_run(run["run_id"])
        self.ctx._handle_state_patch({"run": {"id": run["run_id"], "status": "interrupted"}})
        self.assertEqual(self.ctx.state["run"]["status"], "interrupted")
        self.assertFalse(self.ctx._run_active)
        self.assertEqual(self.ctx.state["services"]["interaction"], "idle")

    def test_worker_exit_after_finalize_keeps_cancel_status(self) -> None:
        run = self.ctx.start_run("live")
        self.ctx._watch_worker_exit = Mock()
        self.ctx._finalize_run("interrupted")
        self.ctx._on_worker_exit("exit")
        self.assertEqual(self.ctx.state["run"]["status"], "interrupted")


class CancelEventWiringTests(ServerTestBase):
    """清单 23：取消事件必须传到本轮/本次调试的模型请求。"""

    def test_debug_vlm_client_receives_the_debug_cancel_event(self) -> None:
        import vlm_adapter
        import worker as worker_mod

        seen: list[object] = []

        class _Recorder:
            def __init__(self, *args, cancel_event=None, **kwargs) -> None:
                seen.append(cancel_event)

        original = vlm_adapter.VlmClient
        vlm_adapter.VlmClient = _Recorder
        try:
            worker = worker_mod.Worker.__new__(worker_mod.Worker)
            worker.lock = threading.RLock()
            worker.cfg = SimpleNamespace(llm=SimpleNamespace(), vlm=SimpleNamespace())
            worker.bench = {}
            ctx = worker_mod.DebugContext(worker, "dbg-wire", self.root / "dbg-wire", "chat", {}, {})
            client_event = ctx.stop_event
            ctx.vlm_client()
            self.assertIs(seen[-1], client_event)
            debug_run = SimpleNamespace(ctx=ctx, runner=Mock())
            worker_mod.DebugRun.stop(debug_run)
            self.assertTrue(client_event.is_set(), "调试停止没有把取消事件交给模型请求")
        finally:
            vlm_adapter.VlmClient = original


class RunEntryCancelEventTests(ServerTestBase):
    """清单 30/23：综合运行入口把停止事件接在同一取消事件上。"""

    def test_run_session_model_cancel_is_set_before_loop_exists(self) -> None:
        from worker import RunSession, Worker

        worker = Worker.__new__(Worker)
        worker.lock = threading.RLock()
        worker.cfg = SimpleNamespace()
        worker.bench = {}
        session = RunSession(worker, "run-wire", self.root / "run-wire")
        self.assertIsNone(session._loop)
        self.assertIsNone(session._stop_event)
        session.request_stop("user")
        self.assertTrue(session._model_cancel.is_set())
        self.assertEqual(session._status(), "interrupted")


class AssemblyCancelTests(ServerTestBase):
    """清单 30：装配在 engine.start 之后被取消时不得继续启动 contract/frontend。"""

    def _session(self):
        import asyncio
        from unittest.mock import AsyncMock

        from worker import RunSession, Worker

        worker = Worker()
        worker.cfg = SimpleNamespace(audio=SimpleNamespace(sample_rate=16000))
        session = RunSession(worker, "run-asm", self.root / "run-asm")
        session._cleanup = AsyncMock()
        session.deps = SimpleNamespace(sink=Mock(), frontend=Mock(), contract=Mock())
        session.engine = SimpleNamespace(start=AsyncMock())
        session._assemble = Mock()
        return session, asyncio

    def test_cancel_during_engine_start_skips_contract_and_frontend(self) -> None:
        session, asyncio = self._session()

        async def stop_mid_engine():
            session.request_stop("user")

        session.engine.start.side_effect = stop_mid_engine
        asyncio.run(session._main())
        session.engine.start.assert_awaited_once()
        session.deps.contract.start.assert_not_called()
        session.deps.frontend.start.assert_not_called()
        self.assertFalse(session._contract_started)
        self.assertFalse(session._frontend_started)
        session._cleanup.assert_awaited_once()

    def test_cancel_during_contract_start_skips_frontend(self) -> None:
        session, asyncio = self._session()

        def stop_mid_contract():
            session.request_stop("user")

        session.deps.contract.start.side_effect = stop_mid_contract
        asyncio.run(session._main())
        session.deps.contract.start.assert_called_once()
        session.deps.frontend.start.assert_not_called()
        # contract.start 已执行完，但取消后再也不得启动 frontend
        self.assertFalse(session._frontend_started)


class AbortIdempotenceTests(ServerTestBase):
    """cancel_http：重复 abort/close 不抛异常，句柄被真正作废（可移植性修复）。"""

    def test_repeated_abort_and_close_are_idempotent(self) -> None:
        import socket as socketmod

        from cancel_http import LocalModelSession

        listener = socketmod.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        client = socketmod.socket()
        client.connect(listener.getsockname())
        peer, _ = listener.accept()
        try:
            session = LocalModelSession(threading.Event())
            session._remember(client)
            session._abort()
            session._abort()
            self.assertEqual(len(session._sockets), 0)
            self.assertEqual(client.fileno(), -1, "句柄未被真正关闭，阻塞读无法被唤醒")
            session.close()
            session.close()
            self.assertFalse(session._watcher.is_alive())
        finally:
            for sock in (client, peer, listener):
                try:
                    sock.close()
                except OSError:
                    pass


# ---------------------------------------------------------------- 线程级状态链
class ThreadBackedWorker:
    """真实线程模拟 worker 进程：状态补丁由独立的"运行线程"回推。

    目的不是复刻 supervisor 的管道，而是让取消→收尾→重试发生在真实线程交错上，
    以便用 is_alive()/proc.poll() 观察进程/线程是否真的退出。
    """

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.pid = 910000 + (id(self) % 9000)
        self.alive = True
        self.calls: list[tuple[str, dict]] = []
        self.run_id = ""
        self._run_stop = threading.Event()
        self._exited = threading.Event()
        self._finished = threading.Event()
        self._thread: threading.Thread | None = None
        self.proc = SimpleNamespace(poll=lambda: 0 if self._exited.is_set() else None)

    # supervisor.WorkerProcess 兼容面
    def start(self) -> None:
        self.ctx._worker_ready.set()

    def call_async(self, cmd, args=None, timeout=180.0, on_done=None):
        self.calls.append((cmd, dict(args or {})))
        if cmd == "run.start":
            self.run_id = str((args or {}).get("run_id") or "")
            self._run_stop.clear()
            self._finished.clear()
            self._thread = threading.Thread(
                target=self._serve_run, name=f"thread-worker-{self.run_id}", daemon=True
            )
            self._thread.start()
            if on_done is not None:
                on_done({"run_id": self.run_id}, None)
        return f"req-{len(self.calls)}"

    def call(self, cmd, args=None, timeout=60.0):
        self.calls.append((cmd, dict(args or {})))
        return {}

    def wait(self, request_id: str, timeout: float = 0.0) -> dict:
        return {}

    def send(self, cmd: str, args: dict | None = None) -> str:
        self.calls.append((cmd, dict(args or {})))
        if cmd == "run.stop":
            self._run_stop.set()
        elif cmd == "shutdown":
            self._exited.set()
        return f"req-{len(self.calls)}"

    def stop(self, **kwargs) -> str:
        self._exited.set()
        self._run_stop.set()
        if self._thread is not None:
            self._thread.join(1.0)
        self.alive = False
        return "graceful"

    def _serve_run(self) -> None:
        self.ctx._on_worker_message(
            {"t": "state", "patch": {"run": {"id": self.run_id, "status": "running"}}}
        )
        self._run_stop.wait(5.0)
        if self._exited.is_set():
            return
        self.ctx._on_worker_message(
            {
                "t": "state",
                "patch": {
                    "run": {
                        "id": self.run_id,
                        "status": "interrupted",
                        "ended_at": "2026-11-16T00:00:00+08:00",
                    }
                },
            }
        )
        self._finished.set()


class ZeroDelayRetryChainTests(ServerTestBase):
    def setUp(self) -> None:
        super().setUp()
        self.spawned: list[ThreadBackedWorker] = []
        self.ctx._spawn_worker = self._thread_spawn  # type: ignore[method-assign]

    def _thread_spawn(self):
        worker = ThreadBackedWorker(self.ctx)
        self.ctx._worker = worker
        self.ctx._worker_ready.clear()
        self.ctx._worker_hello = {}
        worker.start()
        self.spawned.append(worker)
        return worker

    def test_cancel_finishes_in_thread_then_zero_delay_retry_keeps_runs_isolated(self):
        old = self.ctx.start_run("live")
        worker = self.ctx._worker
        old_id = old["run_id"]
        self.assertEqual(self.ctx.state["run"]["id"], old_id)
        self.assertIsNotNone(worker._thread)

        self.ctx.stop_run()
        self.assertTrue(worker._finished.wait(2.0), "取消后 worker 运行线程未收尾")
        self.assertFalse(worker._thread.is_alive())
        self.assertFalse(self.ctx._run_active)

        # 取消后 0ms 立刻重试：watchdog 的 5s 回收线程还在睡眠，worker 必须被复用而不是被杀。
        new = self.ctx.start_run("live")
        new_id = new["run_id"]
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(self.ctx.state["run"]["id"], new_id)
        self.assertEqual(str(self.ctx._run_id), new_id)
        self.assertEqual(self.ctx._event_run_id, new_id)
        self.assertIs(self.ctx._worker, worker)
        self.assertTrue(worker.alive)

        # 旧轮的迟到回调/状态补丁/事件一律丢弃。
        self.ctx._on_run_start_done(old_id, None, {"code": "worker_exited", "message": "late"})
        self.ctx._handle_state_patch({"run": {"id": old_id, "status": "failed", "error": "late"}})
        self.ctx._on_worker_message(
            {"t": "state", "patch": {"run": {"id": old_id, "status": "completed"}}}
        )
        self.ctx._on_worker_message(
            {
                "t": "event",
                "event": {
                    "type": "asr.final",
                    "run_id": old_id,
                    "turn_id": "turn-old",
                    "payload": {"text": "上一轮残留识别"},
                },
            }
        )
        self.ctx._on_worker_message(
            {
                "t": "event",
                "event": {
                    "type": "assistant.delta",
                    "run_id": old_id,
                    "turn_id": "turn-old",
                    "payload": {"text": "上一轮残留回答"},
                },
            }
        )
        state = self.ctx.public_state()
        self.assertEqual(state["run"]["id"], new_id)
        self.assertTrue(self.ctx._run_active)
        self.assertEqual(state["counts"]["turns"], 0)
        self.assertIsNone(state["last_error"])
        self.assertEqual(self.ctx._sessions, [])

        # 新一轮自己的事件仍必须入账（过滤不能过宽）。
        self.ctx._on_worker_message(
            {
                "t": "event",
                "event": {
                    "type": "asr.final",
                    "run_id": new_id,
                    "turn_id": "turn-new",
                    "payload": {"text": "新一轮识别"},
                },
            }
        )
        self.assertEqual([t.get("userText") for t in self.ctx._sessions], ["新一轮识别"])
        worker.stop()

    def test_cancel_then_watchdog_reclaims_worker_and_retry_spawns_new_process(self):
        import time as _time

        from .test_server import server_mod

        # watchdog 线程在 _finalize_run 里就启动并立即 time.sleep(5.0)，
        # 所以缩短 sleep 必须发生在取消之前，否则这条用例会随机等满 5 秒。
        real_sleep = server_mod.time.sleep
        server_mod.time.sleep = lambda s: real_sleep(min(float(s), 0.05))
        try:
            old = self.ctx.start_run("live")
            worker = self.ctx._worker
            self.ctx.stop_run()
            self.assertTrue(worker._finished.wait(2.0))
            self.assertFalse(self.ctx._run_active)
            deadline = _time.time() + 5.0
            while _time.time() < deadline and self.ctx._worker is not None:
                real_sleep(0.01)
        finally:
            server_mod.time.sleep = real_sleep

        self.assertIsNone(self.ctx._worker, "空闲 worker 未被回收")
        self.assertFalse(worker.alive)
        self.assertEqual(worker.proc.poll(), 0)
        self.assertFalse(worker._thread.is_alive())

        new = self.ctx.start_run("live")
        self.assertNotEqual(new["run_id"], old["run_id"])
        self.assertIsNot(self.ctx._worker, worker)
        self.assertIs(self.ctx._worker, self.spawned[-1])
        self.assertEqual(self.ctx.state["run"]["id"], new["run_id"])


class StaleWorkerEventIsolationTests(ServerTestBase):
    def test_event_without_run_tag_still_accepted_while_run_active(self):
        run = self.ctx.start_run("live")
        self.ctx._on_worker_message(
            {
                "t": "event",
                "event": {"type": "asr.final", "turn_id": "turn-1", "payload": {"text": "无标签"}},
            }
        )
        self.assertEqual([t.get("userText") for t in self.ctx._sessions], ["无标签"])
        self.assertEqual(self.ctx._sessions[0]["turnId"], "turn-1")
        self.assertEqual(self.ctx._event_run_id, run["run_id"])

    def test_event_tagged_with_older_run_is_dropped(self):
        self.ctx.start_run("live")
        self.ctx._finalize_run("interrupted")
        run2 = self.ctx.start_run("live")
        self.ctx._on_worker_message(
            {
                "t": "event",
                "event": {
                    "type": "asr.final",
                    "run_id": "run-previous-000000",
                    "turn_id": "turn-old",
                    "payload": {"text": "上一轮残留"},
                },
            }
        )
        self.assertEqual(self.ctx._sessions, [])
        self.assertEqual(self.ctx._event_run_id, run2["run_id"])


class WorkerEventTagContractTests(ServerTestBase):
    """bridge 契约：worker 负责在事件封套里补 run_id（bridge.py 文件头 EmitFn 注释）。"""

    def test_run_session_bridge_events_carry_the_session_run_id(self):
        from worker import RunSession

        seen: list[tuple] = []

        class _WorkerStub:
            def emit_event(self, etype, turn_id, payload, *, run_id=None):
                seen.append((etype, turn_id, dict(payload or {}), run_id))

            def emit_state(self, patch):
                pass

        session = RunSession.__new__(RunSession)
        session.worker = _WorkerStub()
        session.run_id = "run-session-1"
        session._emit_bridge_event("asr.final", "turn-9", {"text": "x"})
        session._emit_bridge_event("frame", None, {"frame_id": "f1"})
        self.assertEqual([item[3] for item in seen], ["run-session-1", "run-session-1"])

    def test_worker_emit_event_writes_run_id_into_envelope(self):
        import worker as worker_mod
        from worker import Worker

        messages: list[dict] = []
        worker = Worker.__new__(Worker)
        worker.timings = {}
        worker.run = SimpleNamespace(run_id="run-active-7")
        worker.debug = None
        saved = worker_mod.emit_message
        worker_mod.emit_message = messages.append  # type: ignore[assignment]
        try:
            worker.emit_event("asr.partial", "turn-1", {"text": "半句"})
        finally:
            worker_mod.emit_message = saved  # type: ignore[assignment]
        envelope = messages[-1]["event"]
        self.assertEqual(envelope["run_id"], "run-active-7")
        self.assertEqual(envelope["type"], "asr.partial")
        self.assertEqual(envelope["turn_id"], "turn-1")

    def test_emit_event_outside_any_session_carries_empty_run_id(self):
        import worker as worker_mod
        from worker import Worker

        messages: list[dict] = []
        worker = Worker.__new__(Worker)
        worker.timings = {}
        worker.run = None
        worker.debug = None
        saved = worker_mod.emit_message
        worker_mod.emit_message = messages.append  # type: ignore[assignment]
        try:
            worker.emit_event("error", None, {"code": "idle"})
        finally:
            worker_mod.emit_message = saved  # type: ignore[assignment]
        self.assertEqual(messages[-1]["event"]["run_id"], "")

    def test_run_finished_does_not_schedule_worker_exit(self):
        import worker as worker_mod
        from worker import Worker

        worker = Worker.__new__(Worker)
        worker.lock = threading.RLock()
        worker.state = {"run": {}}
        session = SimpleNamespace(run_id="run-x", started_iso="2026-11-16T00:00:00+08:00")
        worker.run = session
        worker.debug = None
        worker.devices_patch = lambda: {}  # type: ignore[method-assign]
        worker.release_preview_ref = lambda *a, **k: None  # type: ignore[method-assign]
        worker._schedule_exit = Mock()  # type: ignore[method-assign]
        messages: list[dict] = []
        saved = worker_mod.emit_message
        worker_mod.emit_message = messages.append  # type: ignore[assignment]
        try:
            worker.on_run_finished(session, "interrupted", [])
        finally:
            worker_mod.emit_message = saved  # type: ignore[assignment]
        worker._schedule_exit.assert_not_called()
        self.assertIsNone(worker.run)
        self.assertFalse([m for m in messages if m.get("t") == "exit"])
        patches = [m["patch"] for m in messages if m.get("t") == "state"]
        self.assertEqual(patches[-1]["run"]["status"], "interrupted")
        self.assertEqual(patches[-1]["run"]["id"], "run-x")


    def test_terminal_state_patch_reports_devices_already_closed(self):
        """不变量：终态补丁自带"设备已关闭"，于是"看见 run 结束"就等价于"音频已放手"。

        lead 裁定不加 stop 完成闸门（双 worker 并存 31–47ms 不修）的前提就是这条：
        interrupted/completed/failed 的补丁里 audio_in/audio_out 必须是 unknown、camera 必须是
        closed —— 新一轮能起，说明上一轮已经在补丁里放掉了所有设备。
        """

        import worker as worker_mod
        from worker import Worker

        worker = Worker.__new__(Worker)
        worker.lock = threading.RLock()
        worker.state = {"run": {}}
        worker.timings = {}
        worker.hub = None
        worker.audio_in = {"status": "unknown", "detail": ""}
        worker.audio_out = {"status": "unknown", "detail": ""}
        session = SimpleNamespace(run_id="run-gone", started_iso="2026-11-16T00:00:00+08:00")
        worker.run = session
        worker.debug = None
        worker.release_preview_ref = lambda *a, **k: None  # type: ignore[method-assign]
        messages: list[dict] = []
        saved = worker_mod.emit_message
        worker_mod.emit_message = messages.append  # type: ignore[assignment]
        try:
            worker.on_run_finished(session, "interrupted", [])
        finally:
            worker_mod.emit_message = saved  # type: ignore[assignment]
        patch = [m["patch"] for m in messages if m.get("t") == "state"][-1]
        self.assertEqual(patch["run"]["status"], "interrupted")
        self.assertEqual(patch["devices"]["audio_in"]["status"], "unknown")
        self.assertEqual(patch["devices"]["audio_out"]["status"], "unknown")
        self.assertEqual(patch["devices"]["camera"]["status"], "closed")

class StaleWorkerCallbackIdentityTests(ServerTestBase):
    """清单 25/30：被替换掉的旧 worker 的回调不得影响新一轮。

    真机复现（logs/M1_probe_s2.jsonl）：空闲回收把 `_worker` 置 None 之后旧进程还在
    stop() 里退出，紧接着到达的 run.start 会立刻拉起新 worker —— 探针 5ms 采样看到
    两个 worker 进程并存 31ms（max_live_pids=2）。并存本身不致命，致命的是**旧进程的
    退出/事件回调**落到新一轮上：那会把刚起的 run 判成 failed，或把旧事件记进新 run。
    这里锁住 `_spawn_worker()` 里按身份过滤的那两行 lambda。
    """

    def _process_class(self, created: list) -> type:
        class _Process:
            def __init__(self, **kwargs) -> None:
                self.kwargs = kwargs
                self.pid = 9000 + len(created)
                self.alive = True
                self.calls: list[tuple[str, dict]] = []
                created.append(self)

            def start(self) -> None:
                return None

            def send(self, cmd, args=None) -> None:
                self.calls.append((cmd, dict(args or {})))

            def call(self, cmd, args=None, timeout=180.0):
                self.calls.append((cmd, dict(args or {})))
                return {"ok": True}

            def call_async(self, cmd, args=None, timeout=180.0, on_done=None) -> None:
                self.calls.append((cmd, dict(args or {})))
                if on_done is not None:
                    on_done({"run_id": (args or {}).get("run_id"), "status": "running"}, None)

            def stop(self, **kwargs) -> str:
                self.alive = False
                return "graceful"

        return _Process

    def test_callbacks_of_a_replaced_worker_do_not_touch_the_new_run(self) -> None:
        import supervisor
        import server as server_mod

        created: list = []
        original = supervisor.WorkerProcess
        # ServerTestBase 把 _spawn_worker 换成了 FakeWorker 工厂；这里换回产品实现，
        # 因为本次要验的正是 `_spawn_worker()` 里按 worker 身份过滤回调的那两行 lambda。
        self.ctx._spawn_worker = server_mod.BenchServer._spawn_worker.__get__(self.ctx)
        supervisor.WorkerProcess = self._process_class(created)
        try:
            old_worker = self.ctx._spawn_worker()
            run = self.ctx.start_run("live")
            events_before = len(self.ctx._events)

            # 旧 worker 还没退出，回收逻辑已经清掉 _worker 并起了新的
            new_worker = self.ctx._spawn_worker()
            self.assertIs(self.ctx._worker, new_worker)
            self.assertTrue(new_worker.alive)

            # 旧进程迟到的事件：不得入新一轮
            old_worker.kwargs["on_message"](
                {
                    "t": "event",
                    "event": {
                        "type": "asr.partial",
                        "run_id": run["run_id"],
                        "turn_id": "t-old",
                        "payload": {"text": "旧进程残留"},
                    },
                }
            )
            self.assertEqual(len(self.ctx._events), events_before)

            # 旧进程迟到退出：不得把新一轮判成 failed
            old_worker.kwargs["on_exit"]("crash")
            self.assertTrue(self.ctx._run_active)
            self.assertEqual(self.ctx.state["run"]["status"], "starting")

            # 当前 worker 真退出仍必须生效（证明过滤不是死代码）
            new_worker.kwargs["on_exit"]("crash")
            self.assertFalse(self.ctx._run_active)
            self.assertEqual(self.ctx.state["run"]["status"], "failed")
            self.assertIn("worker 退出：crash", str(self.ctx.state["run"].get("error")))
        finally:
            supervisor.WorkerProcess = original
