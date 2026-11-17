"""A3：debug 连续两轮 + 空闲窗口的旧事件隔离。

三条独立洞（修复前各自可复现，见 logs/A3_repro_prefix.log）：

A. worker 侧标签不固定：`DebugContext.emit`（worker.py:576-577）与它构造的
   `Bridge`（worker.py:569-573，`Bridge._emit` 只传 3 个参数）都不带 run_id，
   于是 `Worker.emit_event` 落到 `run_id is None` 分支去问 `_active_run_id()`
   （worker.py:788-793，"debug 优先"）——第一轮 debug 的残留线程在第二轮起来之后，
   事件被贴成**新** debug 的 run_id。
B. server 侧过滤是死代码：`_handle_worker_event`（server.py:606-619）把 run_id 过滤
   写在 `elif` 后面，而 debug 轮先命中 `if self._event_kind == "debug"` → 永不执行。
C. 空闲窗口：`_close_run_files`（server.py:758-766）把 `_event_run_id` 置 None 之后，
   `and self._event_run_id` 短路为假 → 带非空 run_id 的旧事件照旧入账
   （仍会 `_apply_event_locked`：frames / timings / last_error / events 计数）。
"""
from __future__ import annotations

import copy
import threading
import unittest

from .test_server import ServerTestBase, server_mod


class _Recorder:
    """记录 worker.emit_event 的实参（含 run_id 关键字）。"""

    def __init__(self) -> None:
        self.events: list[tuple[str, object, dict, object]] = []
        self.cfg = None
        self.bench = {}
        self.run = None
        self.debug = None

    def emit_event(self, etype, turn_id, payload, *, run_id=None):
        self.events.append((etype, turn_id, dict(payload or {}), run_id))

    def log(self, *args, **kwargs):
        pass


class DebugRunIdParityTests(ServerTestBase):
    """最高风险点对账：server 侧的 debug_id 与 worker 侧 DebugRun.run_id 是同一个字符串。"""

    def test_server_debug_id_reaches_worker_debug_start_verbatim(self):
        started = self.ctx.start_debug("audio", {})
        sent = dict(self.ctx._worker.calls[-1][1])
        print(
            "[parity] server.start_debug.run_id=%r server._debug_run_id=%r "
            "server._event_run_id=%r debug.start.args.run_id=%r"
            % (started["run_id"], self.ctx._debug_run_id, self.ctx._event_run_id, sent.get("run_id"))
        )
        self.assertEqual(sent.get("run_id"), started["run_id"])
        self.assertEqual(self.ctx._debug_run_id, started["run_id"])
        self.assertEqual(self.ctx._event_run_id, started["run_id"])

    def test_worker_debug_run_keeps_the_id_verbatim(self):
        from worker import DebugRun

        worker = _Recorder()
        worker.lock = threading.RLock()
        debug_run = DebugRun(worker, "dbg-1", self.root / "dbg-1", "audio", {}, {})
        print(
            "[parity] DebugRun.run_id=%r ctx.run_id=%r bridge.run_id=%r"
            % (debug_run.run_id, debug_run.ctx.run_id, debug_run.ctx.bridge.run_id)
        )
        self.assertEqual(debug_run.run_id, "dbg-1")
        self.assertEqual(debug_run.ctx.run_id, "dbg-1")
        self.assertEqual(debug_run.ctx.bridge.run_id, "dbg-1")


class DebugEventTaggingTests(ServerTestBase):
    """worker 侧：事件标签必须在会话对象构造时固定，不能靠"当前是谁"推断。"""

    def _worker_shell(self, debug_id: str):
        """真 Worker 的方法体 + 手工属性（不跑 __init__ 的配置加载）。"""
        import worker as worker_mod

        worker = worker_mod.Worker.__new__(worker_mod.Worker)
        worker.lock = threading.RLock()
        worker.timings = {}
        worker.cfg = None
        worker.bench = {}
        worker.run = None
        worker.debug = type("NewDebug", (), {"run_id": debug_id})()  # 第二轮 debug 已经在跑
        return worker, worker_mod

    def _capture(self, worker, worker_mod, action):
        messages: list[dict] = []
        saved = worker_mod.emit_message
        worker_mod.emit_message = messages.append  # type: ignore[assignment]
        try:
            action()
        finally:
            worker_mod.emit_message = saved  # type: ignore[assignment]
        return messages

    def test_debug_context_emit_is_tagged_with_its_own_run_id(self):
        from worker import DebugContext

        worker, worker_mod = self._worker_shell("dbg-new")
        # 第一轮里建好 DebugContext（构造时绑定 run_id），此时 worker.debug 已指向第二轮。
        old = DebugContext(worker, "dbg-old", self.root / "dbg-old", "audio", {}, {})

        def act():
            old.emit("asr.final", {"text": "上一轮残留识别"})

        messages = self._capture(worker, worker_mod, act)
        envelope = messages[-1]["event"]
        self.assertEqual(envelope["run_id"], "dbg-old")
        self.assertEqual(envelope["type"], "asr.final")

    def test_debug_bridge_emit_is_tagged_with_its_own_run_id(self):
        from worker import DebugContext

        worker, worker_mod = self._worker_shell("dbg-new")
        old = DebugContext(worker, "dbg-old", self.root / "dbg-old", "audio", {}, {})
        messages = self._capture(worker, worker_mod, lambda: old.bridge.error("old_blew", "上一轮错误"))
        envelope = messages[-1]["event"]
        self.assertEqual(envelope["run_id"], "dbg-old")
        self.assertEqual(envelope["type"], "error")

    def test_debug_context_emit_without_any_debug_session_keeps_empty_tag(self):
        from worker import DebugContext

        worker, worker_mod = self._worker_shell("dbg-unused")
        worker.debug = None
        ctx = DebugContext(worker, "dbg-old", self.root / "dbg-old", "audio", {}, {})
        messages = self._capture(worker, worker_mod, lambda: ctx.emit("log", {"level": "INFO", "message": "x"}))
        self.assertEqual(messages[-1]["event"]["run_id"], "dbg-old")


def _stale_events(debug_id: str) -> list[dict]:
    """上一轮 debug 残留线程仍在飞的事件（asr.final / assistant.delta / frame / timing / error）。"""
    return [
        {"type": "asr.final", "run_id": debug_id, "turn_id": "turn-old", "payload": {"text": "上一轮残留识别"}},
        {"type": "assistant.delta", "run_id": debug_id, "turn_id": "turn-old", "payload": {"text": "上一轮残留回答"}},
        {
            "type": "frame",
            "run_id": debug_id,
            "turn_id": None,
            "payload": {"frame_id": "frame-old", "slot": "model", "file": "old.jpg", "width": 8, "height": 8},
        },
        {"type": "timing", "run_id": debug_id, "turn_id": None, "payload": {"asr_ms": 123}},
        {"type": "error", "run_id": debug_id, "turn_id": None, "payload": {"code": "old_blew", "message": "上一轮错误"}},
    ]


class DebugRoundIsolationTests(ServerTestBase):
    """server 侧：旧 debug 轮的事件不得写进新 debug 轮 / 空闲窗口。"""

    def setUp(self) -> None:
        super().setUp()
        self._real_sleep = server_mod.time.sleep
        server_mod.time.sleep = lambda seconds: self._real_sleep(min(float(seconds), 0.02))

    def tearDown(self) -> None:
        server_mod.time.sleep = self._real_sleep
        super().tearDown()

    def _finish_debug(self, debug_id: str, status: str = "completed") -> None:
        self.ctx._on_worker_message(
            {"t": "state", "patch": {"debug": {"run_id": debug_id, "status": status}}}
        )

    def _inject(self, events: list[dict]) -> None:
        for event in events:
            self.ctx._on_worker_message({"t": "event", "event": dict(event)})

    def test_old_debug_events_do_not_pollute_the_next_debug_round(self):
        first = self.ctx.start_debug("audio", {})
        old_id = first["run_id"]
        self._finish_debug(old_id)
        self.assertFalse(self.ctx._debug_active)

        second = self.ctx.start_debug("audio", {})
        new_id = second["run_id"]
        self.assertNotEqual(new_id, old_id)
        self.assertEqual(self.ctx._event_run_id, new_id)

        counts_before = dict(self.ctx.state["counts"])
        timings_before = dict(self.ctx.state["timings"])
        self._inject(_stale_events(old_id))

        self.assertEqual(self.ctx.state["counts"], counts_before, "上一轮事件改动了计数")
        self.assertEqual(self.ctx.state["timings"], timings_before, "上一轮 timing 写进了新轮")
        self.assertIsNone(self.ctx.state["frames"]["model"], "上一轮帧写进了新轮帧面板")
        self.assertNotIn("frame-old", self.ctx._frames, "上一轮帧进了帧索引")
        self.assertIsNone(self.ctx.state.get("last_error"), "上一轮错误写进了新轮")

    def test_old_debug_events_in_idle_window_are_dropped(self):
        first = self.ctx.start_debug("audio", {})
        old_id = first["run_id"]
        self._finish_debug(old_id)
        self.assertIsNone(self.ctx._event_run_id, "收尾后应处于无活动轮次的空闲窗口")

        counts_before = dict(self.ctx.state["counts"])
        frames_before = copy.deepcopy(self.ctx.state["frames"])
        self._inject(_stale_events(old_id))

        self.assertEqual(self.ctx.state["counts"], counts_before, "空闲窗口仍记了上一轮事件")
        self.assertEqual(self.ctx.state["frames"], frames_before, "空闲窗口仍记了上一轮帧")
        self.assertNotIn("frame-old", self.ctx._frames, "空闲窗口仍把上一轮帧进了索引")
        self.assertIsNone(self.ctx.state.get("last_error"), "空闲窗口仍写了上一轮错误")
        self.assertIsNone(self.ctx.state["timings"]["asr_ms"], "空闲窗口仍写了上一轮 timing")

    def test_event_without_run_tag_is_still_accepted(self):
        """判据 1：worker 无任何会话时 run_id 为空 → 保持既有入账行为，不能被过滤误杀。"""
        self.assertIsNone(self.ctx._event_run_id)
        events_before = int(self.ctx.state["counts"]["events"])
        self.ctx._on_worker_message(
            {
                "t": "event",
                "event": {
                    "type": "log",
                    "run_id": "",
                    "turn_id": None,
                    "payload": {"level": "INFO", "module": "worker", "message": "无会话事件"},
                },
            }
        )
        self.assertEqual(int(self.ctx.state["counts"]["events"]), events_before + 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
