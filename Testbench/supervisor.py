"""单工作进程监管：按需启动、按行 JSON 命令往返、进程归属校验与停止。

设计约束（见 Docs/测试台/执行方案.md §4/§6）：

* server 同一时刻最多持有一个 worker；worker 同时最多一个业务任务。
* 启动前记录 pid / 进程组 / /proc 启动时间；停止前核对三项归属。
* 只结束测试台自己创建的 worker 进程组，不使用 ``pkill -f``。
* stdout 专供按行 JSON 协议；stderr 写入测试台日志。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable

MAX_LINE_BYTES = 8 * 1024 * 1024


class WorkerRequestError(RuntimeError):
    """worker 返回 error 应答。"""

    def __init__(self, error: dict) -> None:
        self.error = error or {}
        super().__init__(str(self.error.get("message") or self.error.get("code") or "worker error"))


class WorkerProcess:
    """一个测试台自有 worker 子进程。"""

    def __init__(
        self,
        *,
        bench_dir: Path,
        project_root: Path,
        python_path: str,
        log_path: Path,
        on_message: Callable[[dict], None],
        on_exit: Callable[[str], None] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> None:
        self.bench_dir = Path(bench_dir)
        self.project_root = Path(project_root)
        self.python_path = python_path
        self.log_path = Path(log_path)
        self.on_message = on_message
        self.on_exit = on_exit
        self.extra_env = dict(extra_env or {})

        self._proc: subprocess.Popen | None = None
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._pending: dict[str, dict] = {}
        self._seq = 0
        self._pid: int | None = None
        self._pgid: int | None = None
        self._start_ticks: int | None = None
        self._alive = False
        self._exit_reason = ""
        self._exit_reported = False
        self._threads: list[threading.Thread] = []
        self._log_handle = None

    # ------------------------------------------------------------------ 属性
    @property
    def pid(self) -> int | None:
        return self._pid

    @property
    def pgid(self) -> int | None:
        return self._pgid

    @property
    def alive(self) -> bool:
        return self._alive

    @property
    def start_ticks(self) -> int | None:
        return self._start_ticks

    # ------------------------------------------------------------------ 启动
    def start(self) -> None:
        with self._lock:
            if self._alive:
                return
            self.bench_dir.mkdir(parents=True, exist_ok=True)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            env = os.environ.copy()
            env.update(
                {
                    "PYTHONPATH": str(self.project_root / "Src"),
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONUNBUFFERED": "1",
                    "XDG_CACHE_HOME": str(self.bench_dir / ".runtime" / "cache"),
                    "TMPDIR": str(self.bench_dir / ".runtime" / "tmp"),
                    "TMP": str(self.bench_dir / ".runtime" / "tmp"),
                    "TEMP": str(self.bench_dir / ".runtime" / "tmp"),
                    "CUDA_CACHE_PATH": str(self.bench_dir / ".runtime" / "cache" / "cuda"),
                    "MPLCONFIGDIR": str(self.bench_dir / ".runtime" / "cache" / "mpl"),
                }
            )
            env.update(self.extra_env)
            for key in ("XDG_CACHE_HOME", "TMPDIR", "TMP", "CUDA_CACHE_PATH", "MPLCONFIGDIR"):
                Path(env[key]).mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self.log_path, "a", encoding="utf-8", buffering=1)
            cmd = [self.python_path, "-B", str(self.bench_dir / "worker.py")]
            kwargs: dict[str, Any] = {}
            if os.name == "posix":
                kwargs["start_new_session"] = True
            self._proc = subprocess.Popen(
                cmd,
                cwd=str(self.bench_dir),
                env=env,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                **kwargs,
            )
            self._pid = self._proc.pid
            self._pgid = self._pid
            self._start_ticks = _read_start_ticks(self._pid)
            self._alive = True
            self._exit_reason = ""
            self._exit_reported = False
            self._write_worker_record()
            self._threads = [
                threading.Thread(target=self._read_stdout, name="worker-stdout", daemon=True),
                threading.Thread(target=self._read_stderr, name="worker-stderr", daemon=True),
            ]
            for thread in self._threads:
                thread.start()

    # ------------------------------------------------------------------ 命令
    def send(self, cmd: str, args: dict | None = None) -> str:
        with self._lock:
            if not self._alive or self._proc is None or self._proc.stdin is None:
                raise WorkerRequestError(
                    {"code": "worker_not_running", "message": "工作进程未运行", "retryable": True}
                )
            self._seq += 1
            request_id = f"req{self._seq}"
            line = json.dumps(
                {"id": request_id, "cmd": cmd, "args": args or {}},
                ensure_ascii=False,
            )
            with self._write_lock:
                self._proc.stdin.write(line + "\n")
                self._proc.stdin.flush()
            return request_id

    def call(self, cmd: str, args: dict | None = None, timeout: float = 60.0) -> dict:
        request_id = self.send(cmd, args)
        return self.wait(request_id, timeout=timeout)

    def call_async(
        self,
        cmd: str,
        args: dict | None,
        *,
        timeout: float = 180.0,
        on_done: Callable[[dict | None, dict | None], None],
    ) -> str:
        """后台等待应答；回调参数为 (result, error)。异常在线程内捕获。"""

        request_id = self.send(cmd, args)

        def _wait() -> None:
            result: dict | None = None
            error: dict | None = None
            try:
                result = self.wait(request_id, timeout=timeout)
            except WorkerRequestError as exc:
                error = exc.error
            except Exception as exc:  # noqa: BLE001 - 超时/进程退出
                error = {"code": "worker_timeout", "message": str(exc), "retryable": True}
            try:
                on_done(result, error)
            except Exception:  # noqa: BLE001 - 回调异常不能杀线程
                pass

        threading.Thread(target=_wait, name="worker-call", daemon=True).start()
        return request_id

    def wait(self, request_id: str, timeout: float = 60.0) -> dict:
        with self._lock:
            slot = self._pending.get(request_id)
            if slot is None:
                slot = {"event": threading.Event(), "result": None, "error": None}
                self._pending[request_id] = slot
        if not slot["event"].wait(timeout):
            with self._lock:
                self._pending.pop(request_id, None)
            raise WorkerRequestError(
                {"code": "worker_timeout", "message": "工作进程应答超时", "retryable": True}
            )
        with self._lock:
            self._pending.pop(request_id, None)
            error = slot["error"]
            result = slot["result"]
        if error:
            raise WorkerRequestError(error)
        return result or {}

    # ------------------------------------------------------------------ 读取
    def _read_stdout(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        try:
            for line in proc.stdout:
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    message = json.loads(line)
                except json.JSONDecodeError:
                    self._write_stderr_line(f"[non-json stdout] {line[:500]}")
                    continue
                self._dispatch(message)
        except Exception as exc:  # noqa: BLE001
            self._write_stderr_line(f"[stdout reader] {exc}")
        finally:
            self._handle_exit()

    def _read_stderr(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        try:
            for line in proc.stderr:
                self._write_stderr_line(line.rstrip("\n"))
        except Exception:  # noqa: BLE001
            pass

    def _write_stderr_line(self, line: str) -> None:
        handle = self._log_handle
        if handle is None:
            return
        try:
            handle.write(f"{time.strftime('%H:%M:%S')} {line}\n")
        except Exception:  # noqa: BLE001
            pass

    def _dispatch(self, message: dict) -> None:
        kind = message.get("t")
        if kind == "ack":
            request_id = str(message.get("id") or "")
            with self._lock:
                slot = self._pending.get(request_id)
            if slot is not None:
                slot["result"] = message.get("result") or {}
                slot["event"].set()
            return
        if kind == "error":
            request_id = str(message.get("id") or "")
            with self._lock:
                slot = self._pending.get(request_id)
            if slot is not None:
                slot["error"] = message.get("error") or {
                    "code": "worker_error",
                    "message": "worker 处理失败",
                    "retryable": False,
                }
                slot["event"].set()
                return
        try:
            self.on_message(message)
        except Exception:  # noqa: BLE001
            pass

    def _handle_exit(self) -> None:
        proc = self._proc
        code = None
        try:
            code = proc.wait(timeout=1.0) if proc is not None else None
        except Exception:  # noqa: BLE001
            pass
        with self._lock:
            if not self._alive and self._exit_reported:
                return
            self._alive = False
            self._exit_reason = f"exit_code={code}"
            reason = self._exit_reason
            if self._exit_reported:
                return
            self._exit_reported = True
            pending = list(self._pending.values())
            self._pending.clear()
        for slot in pending:
            slot["error"] = {
                "code": "worker_exited",
                "message": "工作进程已退出",
                "retryable": True,
            }
            slot["event"].set()
        self._clear_worker_record()
        if self.on_exit is not None:
            try:
                self.on_exit(reason)
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ 停止
    def stop(self, *, grace_ms: int = 10000, kill_wait_ms: int = 3000) -> str:
        """请求 worker 退出；10s 未退出则 SIGTERM 自有进程组，再 3s 才 SIGKILL。"""
        proc = self._proc
        if proc is None or proc.poll() is not None or not self._alive:
            return "exited"
        try:
            self.send("shutdown", {})
        except Exception:  # noqa: BLE001
            pass
        if self._wait_exit(grace_ms / 1000.0):
            return "graceful"
        if not self._verify_owned():
            return "lost_ownership"
        self._terminate_group(signal.SIGTERM, "SIGTERM")
        if self._wait_exit(kill_wait_ms / 1000.0):
            return "terminated"
        if self._verify_owned():
            self._terminate_group(signal.SIGKILL, "SIGKILL")
        self._wait_exit(1.0)
        return "killed"

    def kill_now(self) -> None:
        if self._verify_owned():
            self._terminate_group(signal.SIGKILL, "SIGKILL")
        self._wait_exit(1.0)

    def _wait_exit(self, seconds: float) -> bool:
        deadline = time.monotonic() + max(0.0, seconds)
        proc = self._proc
        while time.monotonic() < deadline:
            if proc is not None and proc.poll() is not None:
                return True
            time.sleep(0.05)
        return proc is not None and proc.poll() is not None

    def _terminate_group(self, sig: int, label: str) -> None:
        pgid = self._pgid
        if not pgid:
            return
        self._write_stderr_line(f"[supervisor] send {label} pgid={pgid}")
        try:
            if os.name == "posix":
                os.killpg(pgid, sig)
            else:
                self._proc.kill() if self._proc else None
        except ProcessLookupError:
            pass
        except Exception:  # noqa: BLE001
            try:
                if self._proc is not None:
                    self._proc.kill()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ 归属
    def _verify_owned(self) -> bool:
        """核对 pid / 启动时间 / 命令行三项归属，避免误杀其他进程。"""
        pid = self._pid
        if not pid or not _pid_exists(pid):
            return False
        ticks = _read_start_ticks(pid)
        if self._start_ticks is not None and ticks is not None and ticks != self._start_ticks:
            self._write_stderr_line("[supervisor] start_ticks mismatch, refuse kill")
            return False
        cmdline = _read_cmdline(pid)
        if cmdline and "worker.py" not in cmdline:
            self._write_stderr_line("[supervisor] cmdline mismatch, refuse kill")
            return False
        return True

    def _write_worker_record(self) -> None:
        path = self.bench_dir / ".runtime" / "worker.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(
                    {
                        "pid": self._pid,
                        "pgid": self._pgid,
                        "start_ticks": self._start_ticks,
                        "argv": [self.python_path, "worker.py"],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except Exception:  # noqa: BLE001
            pass

    def _clear_worker_record(self) -> None:
        path = self.bench_dir / ".runtime" / "worker.json"
        try:
            if path.exists():
                path.unlink()
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# /proc 辅助（Linux）；不可用时返回 None，由调用方保守处理
# ---------------------------------------------------------------------------
def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except Exception:  # noqa: BLE001
        return False


def _read_start_ticks(pid: int | None) -> int | None:
    if not pid:
        return None
    path = Path(f"/proc/{pid}/stat")
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    # comm 可能含空格/括号，取最后一个 ')' 之后的字段；starttime 为字段 22。
    idx = raw.rfind(")")
    if idx < 0:
        return None
    fields = raw[idx + 2 :].split()
    if len(fields) < 20:
        return None
    try:
        return int(fields[19])
    except ValueError:
        return None


def _read_cmdline(pid: int | None) -> str:
    if not pid:
        return ""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return ""
    return raw.replace(b"\x00", b" ").decode("utf-8", errors="replace")


def worker_python(python_path: str) -> str:
    """解析 bench.json 中相对项目根的 python 路径。"""
    path = Path(python_path)
    if path.is_absolute():
        return str(path)
    # 用 abspath 只做路径规范化，不解析 .venv/bin/python 这类符号链接。
    return os.path.abspath(os.path.join(os.getcwd(), python_path))


def default_env(project_root: Path, bench_dir: Path) -> dict[str, str]:
    runtime = bench_dir / ".runtime"
    return {
        "PYTHONPATH": str(project_root / "Src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONIOENCODING": "utf-8",
        "XDG_CACHE_HOME": str(runtime / "cache"),
        "TMPDIR": str(runtime / "tmp"),
        "TMP": str(runtime / "tmp"),
        "TEMP": str(runtime / "tmp"),
        "CUDA_CACHE_PATH": str(runtime / "cuda"),
    }


if __name__ == "__main__":  # pragma: no cover - 手工排查用
    print(json.dumps({"ok": True, "platform": sys.platform}))
