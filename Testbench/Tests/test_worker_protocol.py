"""worker.py 行协议冒烟测试：hello / status / shutdown。

只启动 worker 进程本身（不启动综合测试、预览或调试），因此不加载模型、不开音频/相机。
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SRC = BENCH.parent / "src"
WORKER = BENCH / "worker.py"


class WorkerProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        self.proc = subprocess.Popen(
            [sys.executable, "-B", str(WORKER)],
            cwd=str(BENCH),
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        self.lines: queue.Queue = queue.Queue()
        self._reader = threading.Thread(target=self._pump, daemon=True)
        self._reader.start()

    def _pump(self) -> None:
        assert self.proc.stdout is not None
        for line in self.proc.stdout:
            self.lines.put(line)
        self.lines.put(None)

    def tearDown(self) -> None:
        if self.proc.poll() is None:
            self.proc.kill()
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass

    def read_message(self, timeout: float = 30.0) -> dict:
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.fail("worker 超过 %.1fs 未输出协议消息" % timeout)
        self.assertIsNotNone(line, "worker stdout 已关闭")
        return json.loads(line)

    def send(self, message: dict) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(message, ensure_ascii=False) + "\n")
        self.proc.stdin.flush()

    def wait_for(self, predicate, limit: int = 80) -> dict:
        for _ in range(limit):
            message = self.read_message()
            if predicate(message):
                return message
        self.fail("未等到期望的协议消息")

    def test_hello_status_shutdown(self) -> None:
        hello = self.wait_for(lambda item: item.get("t") == "hello")
        self.assertIn("pid", hello)
        self.assertIn("capabilities", hello)

        self.send({"id": "req-status", "cmd": "status"})
        ack = self.wait_for(lambda item: item.get("t") == "ack" and item.get("id") == "req-status")
        self.assertIn("capabilities", ack.get("result") or {})
        self.assertIn("run", ack.get("result") or {})

        self.send({"id": "req-shutdown", "cmd": "shutdown"})
        ack = self.wait_for(lambda item: item.get("t") == "ack" and item.get("id") == "req-shutdown")
        self.assertTrue((ack.get("result") or {}).get("shutdown", True))
        self.assertEqual(self.proc.wait(timeout=20), 0)


if __name__ == "__main__":
    unittest.main()
