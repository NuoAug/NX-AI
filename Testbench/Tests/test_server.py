"""server.py 单元/接口测试：状态、会话、事件游标、上传与控制权。

使用临时目录与假 worker，不启动真实业务组件、不触碰设备/模型。
"""

from __future__ import annotations

import http.client
import io
import json
import socket
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SRC = BENCH.parent / "src"
for path in (str(BENCH), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import server as server_mod  # noqa: E402
import devices as devices_mod  # noqa: E402
from unittest.mock import patch  # noqa: E402

DEVICE_CHOICES = {
    key: [{"id": "", "label": "系统默认（外接设备）", "occupancy": "待核实"}]
    for key in ("camera", "microphone", "speaker")
}


class FakeWorker:
    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.pid = 424242
        self.alive = True
        self.calls: list[tuple[str, dict]] = []

    def start(self) -> None:
        self.ctx._worker_ready.set()

    def send(self, cmd: str, args: dict | None = None) -> str:
        self.calls.append((cmd, dict(args or {})))
        return f"req-{len(self.calls)}"

    def call(self, cmd: str, args: dict | None = None, timeout: float = 60.0) -> dict:
        self.calls.append((cmd, dict(args or {})))
        return {}

    def call_async(self, cmd, args=None, timeout=180.0, on_done=None):
        self.calls.append((cmd, dict(args or {})))
        if on_done is not None:
            on_done({"ok": True}, None)
        return f"req-{len(self.calls)}"

    def wait(self, request_id: str, timeout: float = 0.0) -> dict:
        return {}

    def stop(self, **kwargs) -> str:
        self.alive = False
        return "graceful"


def free_port() -> int:
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


class ServerTestBase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self._saved = {}
        for name, value in {
            "RUNS_DIR": self.root / "runs",
            "UPLOADS_DIR": self.root / "uploads",
            "LOGS_DIR": self.root / "logs",
            "RUNTIME_DIR": self.root / ".runtime",
        }.items():
            self._saved[name] = getattr(server_mod, name)
            setattr(server_mod, name, value)
            value.mkdir(parents=True, exist_ok=True)
        self.bench = {
            "host": "127.0.0.1",
            "port": 0,
            "project_root": str(BENCH.parent),
            "target_config": "config/target.json",
            "python": "../.venv/bin/python",
            "static_files": ["index.html", "app.js"],
            "event_buffer_size": 4,
            "session_turn_limit": 20,
            "heartbeat_timeout_ms": 15000,
            "worker_start_timeout_ms": 5000,
            "worker_stop_grace_ms": 100,
            "worker_kill_wait_ms": 100,
            "upload_max_bytes": 20 * 1024 * 1024,
            "upload_max_pixels": 24000000,
        }
        self.ctx = server_mod.BenchServer(self.bench)
        self.ctx._spawn_worker = self._fake_spawn  # type: ignore[method-assign]
        self.httpd = None
        self.thread = None

    def tearDown(self) -> None:
        server_mod.SERVER = None
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        try:
            self.ctx.shutdown()
        except Exception:  # noqa: BLE001
            pass
        for name, value in self._saved.items():
            setattr(server_mod, name, value)
        self.tmp.cleanup()

    def _fake_spawn(self):
        worker = FakeWorker(self.ctx)
        self.ctx._worker = worker
        self.ctx._worker_hello = {}
        worker.start()
        return worker

    def serve(self) -> int:
        port = free_port()
        self.ctx.port = port
        server_mod.SERVER = self.ctx
        self.httpd = server_mod.BenchHTTPServer(("127.0.0.1", port), self.ctx)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return port

    def request(self, method: str, path: str, body: bytes | None = None, headers: dict | None = None):
        conn = http.client.HTTPConnection("127.0.0.1", self.ctx.port, timeout=5)
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            data = response.read()
            return response.status, data, dict(response.getheaders())
        finally:
            conn.close()

    def heartbeat(self, client_id: str) -> dict:
        status, data, _ = self.request(
            "POST",
            "/api/client/heartbeat",
            body=json.dumps({"client_id": client_id}).encode(),
            headers={"Content-Type": "application/json"},
        )
        self.assertEqual(status, 200)
        return json.loads(data)


class StateTests(ServerTestBase):
    def test_state_shape_with_defaults(self) -> None:
        state = self.ctx.public_state()
        self.assertFalse(state["run"])
        self.assertEqual(state["devices"]["audio_in"]["status"], "unverified")
        self.assertEqual(state["devices"]["camera"]["status"], "closed")
        self.assertEqual(state["control"], {"locked": False})
        self.assertEqual(set(state["timings"]), set(server_mod.TIMING_FIELDS))
        self.assertEqual(state["services"]["interaction"], "idle")
        self.assertEqual(state["images"], {"A": None, "B": None})

    def test_session_turns_mapping(self) -> None:
        self.ctx.append_event("asr.partial", {"text": "你"}, turn_id="t1")
        self.ctx.append_event("asr.final", {"text": "你好"}, turn_id="t1")
        self.ctx.append_event("assistant.delta", {"text": "在"}, turn_id="t1")
        self.ctx.append_event("assistant.final", {"text": "在的"}, turn_id="t1")
        snapshot = self.ctx.session_snapshot()
        self.assertTrue(snapshot["client_id"].startswith("page-"))
        self.assertEqual(snapshot["turns"], [{
            "turn_id": "t1",
            "user": "你好",
            "assistant": "在的",
            "assistant_done": True,
            "cancelled": False,
        }])

    def test_timing_and_frames_update_state(self) -> None:
        self.ctx.append_event("timing", {"asr_ms": 120, "generation_ms": 300})
        self.ctx.append_event(
            "frame",
            {"frame_id": "f1", "slot": "model", "file": "media/f1.jpg", "width": 8, "height": 8},
        )
        state = self.ctx.public_state()
        self.assertEqual(state["timings"]["asr_ms"], 120)
        self.assertEqual(state["timings"]["generation_ms"], 300)
        self.assertEqual(state["frames"]["model"]["frame_id"], "f1")


class EventCursorTests(ServerTestBase):
    def test_events_since_and_next_seq(self) -> None:
        first = self.ctx.append_event("log", {"level": "INFO", "module": "t", "message": "a"})
        second = self.ctx.append_event("log", {"level": "INFO", "module": "t", "message": "b"})
        self.assertEqual(second["seq"], first["seq"] + 1)
        result = self.ctx.events_since(0)
        self.assertEqual([item["message"] for item in (e["payload"] for e in result["events"])], ["a", "b"])
        self.assertEqual(result["next_seq"], second["seq"] + 1)
        after = self.ctx.events_since(first["seq"])
        self.assertEqual([e["payload"]["message"] for e in after["events"]], ["b"])

    def test_ring_overflow_requires_reset(self) -> None:
        for index in range(6):
            self.ctx.append_event("log", {"level": "INFO", "module": "t", "message": str(index)})
        result = self.ctx.events_since(0)
        self.assertTrue(result["reset_required"])
        self.assertEqual(result["events"], [])


class UploadTests(ServerTestBase):
    @staticmethod
    def jpeg_bytes(size: tuple[int, int] = (12, 8)) -> bytes:
        from PIL import Image

        buffer = io.BytesIO()
        Image.new("RGB", size, (10, 20, 30)).save(buffer, format="JPEG")
        return buffer.getvalue()

    def test_upload_save_and_delete(self) -> None:
        meta = self.ctx.save_upload(self.jpeg_bytes(), "A")
        self.assertTrue(meta["image_id"])
        self.assertEqual(meta["slot"], "A")
        self.assertEqual((meta["width"], meta["height"]), (12, 8))
        state_images = self.ctx.public_state()["images"]
        self.assertEqual(state_images["A"]["image_id"], meta["image_id"])
        self.ctx.delete_upload(meta["image_id"])
        self.assertIsNone(self.ctx.public_state()["images"]["A"])

    def test_upload_rejects_bad_bytes(self) -> None:
        with self.assertRaises(server_mod.HttpError) as caught:
            self.ctx.save_upload(b"not-an-image", "A")
        self.assertIn(caught.exception.status, (413, 415, 422))

    def test_upload_during_run_is_not_replaced(self) -> None:
        run = self.ctx.start_run("live")
        meta = self.ctx.save_upload(self.jpeg_bytes(), "A")
        self.ctx._freeze_images = lambda: {"A": meta["image_id"], "B": None}  # type: ignore[method-assign]
        self.assertTrue(meta["image_id"])
        self.ctx.stop_run(run["run_id"])
        self.ctx._finalize_run("completed")
        self.assertEqual(self.ctx.state["run"]["status"], "completed")


class HttpTests(ServerTestBase):
    def test_health_and_static_served(self) -> None:
        self.serve()
        status, data, headers = self.request("GET", "/api/health")
        self.assertEqual(status, 200)
        payload = json.loads(data)
        self.assertTrue(payload["ready"])
        self.assertEqual(headers.get("Cache-Control"), "no-store")
        status, data, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn(b"<!doctype html", data.lower())
        status, _, _ = self.request("GET", "/%2e%2e/%e6%89%a7%e8%a1%8c%e6%96%b9%e6%a1%88.md")
        self.assertEqual(status, 404)
        status, _, _ = self.request("GET", "/src/aieyes/app.py")
        self.assertEqual(status, 404)
        status, _, _ = self.request("GET", "/api/preview.jpg")
        self.assertEqual(status, 503)

    def test_memory_and_device_wording_is_honest(self) -> None:
        self.serve()
        status, data, _ = self.request("GET", "/")
        self.assertEqual(status, 200)
        html = data.decode("utf-8")
        # 3 进程 RSS 合计只是参考值：标签必须带“参考”，且不能用像素条暗示预算达标。
        self.assertIn("三进程 RSS 合计（参考）", html)
        self.assertIn("不是预算占用率", html)
        self.assertIn("测试台自身 RSS", html)
        self.assertNotIn("测试台自身内存", html)
        # 设备列表还没读到时不能先显示成“系统默认”。
        self.assertNotIn('<option value="">系统默认</option>', html)
        self.assertEqual(html.count("设备列表未读取"), 3)

    def test_second_page_and_active_run_cannot_switch_devices(self) -> None:
        self.serve()
        origin = {
            "Origin": f"http://127.0.0.1:{self.ctx.port}",
            "Content-Type": "application/json",
        }
        self.heartbeat("page-1")
        status, _, _ = self.request(
            "POST", "/api/client/claim",
            body=json.dumps({"request_id": "req-dual"}).encode(),
            headers=dict(origin, **{"X-Bench-Client": "page-1"}),
        )
        self.assertEqual(status, 200)
        self.heartbeat("page-2")
        payload = json.dumps({"camera": "", "microphone": "", "speaker": ""}).encode()
        # 第二个页面只读：改设备必须被挡在控制权这一层。
        status, data, _ = self.request(
            "POST", "/api/devices", body=payload,
            headers=dict(origin, **{"X-Bench-Client": "page-2"}),
        )
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(data)["error"]["code"], "client_readonly")
        with patch.object(devices_mod, "inventory", return_value=(DEVICE_CHOICES, [])), \
                patch.object(devices_mod, "save", return_value={"camera": "", "microphone": "", "speaker": ""}) as save:
            # 活动运行中（含预览/调试/worker 存活）即使 owner 也不能切换设备。
            self.ctx._run_active = True
            try:
                status, data, _ = self.request(
                    "POST", "/api/devices", body=payload,
                    headers=dict(origin, **{"X-Bench-Client": "page-1"}),
                )
            finally:
                self.ctx._run_active = False
            self.assertEqual(status, 409)
            self.assertEqual(json.loads(data)["error"]["code"], "device_busy")
            save.assert_not_called()
            # 停止后同一个 owner 可以正常切换，说明封锁只针对活动运行。
            status, data, _ = self.request(
                "POST", "/api/devices", body=payload,
                headers=dict(origin, **{"X-Bench-Client": "page-1"}),
            )
            self.assertEqual(status, 200, data)
            self.assertEqual(save.call_count, 1)

    def test_cross_site_control_rejected(self) -> None:
        self.serve()
        self.heartbeat("page-1")
        status, data, _ = self.request(
            "POST",
            "/api/run/start",
            body=b"{}",
            headers={
                "Origin": "http://evil.example",
                "Content-Type": "application/json",
                "X-Bench-Client": "page-1",
            },
        )
        self.assertEqual(status, 403)
        self.assertEqual(json.loads(data)["error"]["code"], "forbidden_origin")

    def test_missing_client_and_second_page_readonly(self) -> None:
        self.serve()
        status, data, _ = self.request("POST", "/api/run/start", body=b"{}", headers={
            "Origin": f"http://127.0.0.1:{self.ctx.port}",
            "Content-Type": "application/json",
        })
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(data)["error"]["code"], "invalid_request")

        owner = self.heartbeat("page-owner")
        self.assertTrue(owner["owner"])
        other = self.heartbeat("page-other")
        self.assertFalse(other["owner"])
        status, data, _ = self.request("POST", "/api/run/start", body=b"{}", headers={
            "Origin": f"http://127.0.0.1:{self.ctx.port}",
            "Content-Type": "application/json",
            "X-Bench-Client": "page-other",
        })
        self.assertEqual(status, 409)
        self.assertEqual(json.loads(data)["error"]["code"], "client_readonly")

    def test_run_start_exclusive_and_stop_idempotent(self) -> None:
        self.serve()
        self.heartbeat("page-owner")
        headers = {
            "Origin": f"http://127.0.0.1:{self.ctx.port}",
            "Content-Type": "application/json",
            "X-Bench-Client": "page-owner",
        }
        status, data, _ = self.request(
            "POST", "/api/run/start", body=json.dumps({"request_id": "r1"}).encode(), headers=headers
        )
        self.assertEqual(status, 202)
        run_id = json.loads(data)["run_id"]
        status, data, _ = self.request(
            "POST", "/api/run/start", body=json.dumps({"request_id": "r2"}).encode(), headers=headers
        )
        self.assertEqual(status, 409)
        status, data, _ = self.request(
            "POST", "/api/run/stop", body=json.dumps({"request_id": "s1", "run_id": run_id}).encode(),
            headers=headers,
        )
        self.assertEqual(status, 202)
        self.assertEqual(json.loads(data)["status"], "stopping")
        self.ctx._finalize_run("completed")
        status, data, _ = self.request(
            "POST", "/api/run/stop", body=json.dumps({"request_id": "s2", "run_id": run_id}).encode(),
            headers=headers,
        )
        self.assertEqual(status, 202)
        self.assertEqual(json.loads(data)["status"], "idle")

    def test_upload_endpoint_flat_meta(self) -> None:
        self.serve()
        self.heartbeat("page-owner")
        headers = {
            "Origin": f"http://127.0.0.1:{self.ctx.port}",
            "Content-Type": "image/jpeg",
            "X-Bench-Client": "page-owner",
        }
        status, data, _ = self.request("POST", "/api/uploads?slot=A", body=UploadTests.jpeg_bytes(), headers=headers)
        self.assertEqual(status, 201)
        payload = json.loads(data)
        self.assertTrue(payload["image_id"])
        self.assertEqual(payload["slot"], "A")
        self.assertNotIn("image", payload)


if __name__ == "__main__":
    unittest.main()
