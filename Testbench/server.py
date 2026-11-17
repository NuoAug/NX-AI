"""测试台 HTTP 服务：静态页面、状态/事件 API、按需管理工作进程。

约束（见 ``testbench/执行方案.md`` §5/§6/§9）：

* 只监听 ``127.0.0.1``，同源访问；POST/DELETE 校验 Origin，缺失时要求内存令牌。
* 真正的业务在 worker 子进程内执行；HTTP 线程不直接操作音频/相机/状态机。
* 一个 worker 同时最多一个业务任务；预览是独立引用。
* 事件按全局单调 ``seq`` 入环形缓冲，同一份事件写入本轮 ``events.jsonl``。
* 浏览器断开超时后停止活动任务并记录 ``client_disconnected``。
"""

from __future__ import annotations

import collections
import json
import mimetypes
import os
import re
import secrets
import signal
import socket
import sys
import threading
import time
import traceback
import urllib.request
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

BENCH_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BENCH_DIR.parent
RUNS_DIR = BENCH_DIR / "runs"
UPLOADS_DIR = BENCH_DIR / "uploads"
LOGS_DIR = BENCH_DIR / "logs"
RUNTIME_DIR = BENCH_DIR / ".runtime"

STATIC_FILES = {"index.html", "app.js"}
EVENT_TYPES = frozenset(
    {
        "state",
        "asr.partial",
        "asr.final",
        "assistant.delta",
        "assistant.final",
        "playback",
        "frame",
        "timing",
        "log",
        "error",
    }
)
TIMING_FIELDS = (
    "asr_ms",
    "vision_ms",
    "generation_ms",
    "tts_ms",
    "playback_wait_ms",
    "acoustic_first_audio_ms",
)

_UPLOAD_ROUTE = re.compile(r"^/api/uploads/([0-9a-fA-F]+)$")
_FRAME_ROUTE = re.compile(r"^/api/frames/([^/]+)$")


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def load_bench() -> dict:
    with open(BENCH_DIR / "config" / "bench.json", "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("bench.json 顶层必须是对象")
    return data


def wall_ms() -> int:
    return int(time.time() * 1000)


def wall_iso(ms: int | None = None) -> str:
    moment = datetime.fromtimestamp((ms if ms is not None else wall_ms()) / 1000.0)
    return moment.astimezone().isoformat(timespec="seconds")


def _rotate(path: Path, max_bytes: int, count: int) -> None:
    try:
        if not path.is_file() or path.stat().st_size < max_bytes:
            return
        for index in range(count - 1, 0, -1):
            src = path.with_name(f"{path.name}.{index}")
            dst = path.with_name(f"{path.name}.{index + 1}")
            if src.exists():
                if dst.exists():
                    dst.unlink()
                src.rename(dst)
        path.rename(path.with_name(f"{path.name}.1"))
    except OSError:
        pass


class HttpError(Exception):
    def __init__(self, status: int, code: str, message: str, *, retryable: bool = False) -> None:
        self.status = int(status)
        self.code = str(code)
        self.message = str(message)
        self.retryable = bool(retryable)
        super().__init__(message)


# ---------------------------------------------------------------------------
# 服务器
# ---------------------------------------------------------------------------
class BenchServer:
    def __init__(self, bench: dict) -> None:
        self.bench = bench
        self.host = str(bench.get("host") or "127.0.0.1")
        self.port = int(bench.get("port") or 18780)
        self.project_root = Path(bench.get("project_root") or PROJECT_ROOT)
        self.python_path = str(bench.get("python") or "../.venv/bin/python")
        self.poll_ms = int(bench.get("poll_ms", 500))
        self.event_buffer_size = int(bench.get("event_buffer_size", 4000))
        self.session_turn_limit = int(bench.get("session_turn_limit", 200))
        self.heartbeat_timeout_ms = int(bench.get("heartbeat_timeout_ms", 15000))
        self.worker_start_timeout_ms = int(bench.get("worker_start_timeout_ms", 180000))
        self.worker_stop_grace_ms = int(bench.get("worker_stop_grace_ms", 10000))
        self.worker_kill_wait_ms = int(bench.get("worker_kill_wait_ms", 3000))
        self.upload_max_bytes = int(bench.get("upload_max_bytes", 20 * 1024 * 1024))
        self.upload_max_pixels = int(bench.get("upload_max_pixels", 24_000_000))
        self.json_max_bytes = int(bench.get("json_max_bytes", 1_048_576))
        self.http_read_timeout_s = float(bench.get("http_read_timeout_s", 10))
        self.log_rotate_bytes = int(bench.get("log_rotate_bytes", 10 * 1024 * 1024))
        self.log_rotate_count = int(bench.get("log_rotate_count", 5))
        self.model_health_url = str(bench.get("model_health_url") or "http://127.0.0.1:8080/health")
        self.model_health_timeout_s = float(bench.get("model_health_timeout_ms", 2000)) / 1000.0
        self.model_health_cache_ms = int(bench.get("model_health_cache_ms", 3000))

        self.lock = threading.RLock()
        self.token = secrets.token_urlsafe(24)
        self.started_ms = wall_ms()
        self.shutdown_event = threading.Event()

        self._seq = 0
        self._events: collections.deque = collections.deque(maxlen=self.event_buffer_size)
        self._sessions: list[dict] = []
        self._frames: dict[str, dict] = {}
        self._request_cache: dict[str, dict] = {}
        self._request_order: collections.deque = collections.deque(maxlen=100)
        self._events_fp = None
        self._metrics_fp = None
        self._event_run_id: str | None = None
        self._event_kind: str | None = None
        self._run_active = False
        self._debug_active = False
        self._run_id = ""
        self._debug_run_id = ""
        self._preview_active = False
        self._worker: Any = None
        self._spawning = False
        self._worker_ready = threading.Event()
        self._worker_hello: dict = {}
        self._model_state = "unknown"
        self._model_detail = ""
        self._model_checked_ms = 0
        self._last_preview: tuple[bytes, str, int] | None = None
        self._preview_error = ""

        self.state: dict = {
            "run": None,
            "debug": None,
            "preview": {"active": False, "last_frame_ts": None, "error": None, "frames": 0},
            "services": {"model": "unknown", "interaction": "idle", "worker": "none"},
            "devices": {
                "audio_in": {"status": "unverified", "detail": ""},
                "audio_out": {"status": "unverified", "detail": ""},
                "camera": {"status": "closed", "detail": ""},
            },
            "metrics": {
                "sampled_at": None,
                "worker_rss_bytes": None,
                "model_rss_bytes": None,
                "bench_rss_bytes": None,
                "cpu_percent": None,
                "gpu_percent": None,
                "temperature_c": None,
            },
            "timings": {field: None for field in TIMING_FIELDS},
            "frames": {"model": None, "A": None, "B": None},
            "capabilities": {"contract": False, "speaker": False},
            "control": {"locked": False},
            "counts": {"events": 0, "turns": 0},
        }
        self.heartbeats: dict[str, dict] = {}
        self.controller: str | None = None
        self.active_upload_ids: set[str] = set()

        self._load_initial_capabilities()
        self._load_uploads()
        self._load_metrics()

    # ------------------------------------------------------------------ 初始化
    def _load_initial_capabilities(self) -> None:
        """只读 JSON，不初始化任何业务组件。"""
        try:
            target = self.project_root / str(self.bench.get("target_config") or "config/target.json")
            data = json.loads(target.read_text(encoding="utf-8"))
            contract = data.get("contract") or {}
            speaker = data.get("speaker") or {}
            self.state["capabilities"] = {
                "contract": bool(contract.get("enabled", False)),
                "speaker": bool(speaker.get("enabled", False)),
            }
        except Exception:  # noqa: BLE001
            pass

    def _load_uploads(self) -> None:
        from media import UploadStore

        self.uploads = UploadStore(
            UPLOADS_DIR,
            max_bytes=self.upload_max_bytes,
            max_pixels=self.upload_max_pixels,
        )
        self._sync_images_state()

    def _load_metrics(self) -> None:
        from metrics import MetricsSampler

        self.metrics = MetricsSampler(
            proc_root="/proc",
            sys_root="/sys",
            worker_pid_provider=lambda: self._worker.pid if self._worker is not None else None,
            bench_pid=os.getpid(),
        )

    def _sync_images_state(self) -> None:
        slots = self.uploads.slots()
        with self.lock:
            self.state["images"] = {
                name: (self._public_image(meta) if meta else None) for name, meta in slots.items()
            }

    @staticmethod
    def _public_image(meta: dict) -> dict:
        return {
            "image_id": meta.get("image_id"),
            "slot": meta.get("slot"),
            "width": meta.get("width"),
            "height": meta.get("height"),
            "format": meta.get("format"),
            "url": f"/api/uploads/{meta.get('image_id')}",
        }

    # ------------------------------------------------------------------ 日志
    def write_log(self, level: str, module: str, message: str) -> None:
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {level} {module}: {message}"
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            path = LOGS_DIR / "server.log"
            _rotate(path, self.log_rotate_bytes, self.log_rotate_count)
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except OSError:
            pass
        try:
            # 前台运行时同步打到终端；后台由 start.sh 重定向到服务器日志，
            # 此时不再重复输出，避免日志文件出现双份。
            if sys.stderr.isatty():
                print(line, file=sys.stderr)
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------ 事件
    def append_event(
        self,
        etype: str,
        payload: dict | None = None,
        *,
        turn_id: str | None = None,
    ) -> dict:
        if etype not in EVENT_TYPES:
            etype = "log"
            payload = {"level": "WARN", "module": "server", "message": f"未知事件类型已丢弃"}
        with self.lock:
            self._seq += 1
            event = {
                "seq": self._seq,
                "run_id": self._event_run_id,
                "turn_id": turn_id,
                "type": etype,
                "ts_mono_ms": int(time.monotonic() * 1000),
                "payload": dict(payload or {}),
            }
            self._events.append(event)
            self.state["counts"]["events"] = len(self._events)
            self._apply_event_locked(event)
            fp = self._events_fp
            if fp is not None:
                try:
                    fp.write(json.dumps(event, ensure_ascii=False) + "\n")
                    fp.flush()
                except OSError:
                    pass
            return event

    def _apply_event_locked(self, event: dict) -> None:
        etype = event["type"]
        payload = event["payload"]
        turn_id = event.get("turn_id")
        if self._event_kind == "debug" and etype in (
            "asr.partial",
            "asr.final",
            "assistant.delta",
            "assistant.final",
        ):
            # 调试输出走独立面板，不写入聊天会话快照
            self.state["counts"]["turns"] = len(self._sessions)
            return
        if etype in ("asr.partial", "asr.final"):
            turn = self._ensure_turn(turn_id)
            text = str(payload.get("text") or "")
            if text:
                turn["userText"] = text
            if etype == "asr.final":
                turn["final"] = True
                turn["utteranceId"] = payload.get("utterance_id")
        elif etype == "assistant.delta":
            turn = self._ensure_turn(turn_id)
            turn["assistantText"] = str(turn.get("assistantText") or "") + str(
                payload.get("text") or ""
            )
            turn["generating"] = True
        elif etype == "assistant.final":
            turn = self._ensure_turn(turn_id)
            turn["assistantText"] = str(payload.get("text") or "")
            turn["generating"] = False
            turn["cancelled"] = bool(payload.get("cancelled"))
        elif etype == "timing":
            with self.lock:
                for key, value in payload.items():
                    if key in TIMING_FIELDS and isinstance(value, (int, float)):
                        self.state["timings"][key] = value
        elif etype == "frame":
            self._note_frame_locked(payload)
        elif etype == "error":
            self.state["last_error"] = {
                "code": payload.get("code"),
                "message": payload.get("message"),
                "ts": event["ts_mono_ms"],
            }
        self.state["counts"]["turns"] = len(self._sessions)

    def _ensure_turn(self, turn_id: str | None) -> dict:
        key = str(turn_id or "")
        if key:
            for turn in reversed(self._sessions):
                if str(turn.get("turnId") or "") == key:
                    return turn
        turn = {
            "turnId": key or f"turn-{len(self._sessions) + 1}",
            "userText": "",
            "assistantText": "",
            "final": False,
            "generating": False,
            "cancelled": False,
            "utteranceId": None,
        }
        self._sessions.append(turn)
        if len(self._sessions) > self.session_turn_limit:
            del self._sessions[: len(self._sessions) - self.session_turn_limit]
        return turn

    def _note_frame_locked(self, payload: dict) -> None:
        slot = str(payload.get("slot") or "model")
        frame_id = str(payload.get("frame_id") or "")
        file = str(payload.get("file") or "")
        if not frame_id:
            return
        url = f"/api/frames/{frame_id}"
        info = {
            "frame_id": frame_id,
            "url": url,
            "file": file,
            "width": payload.get("width"),
            "height": payload.get("height"),
            "t_capture_ms": payload.get("t_capture_ms"),
            "purpose": payload.get("purpose"),
            "run_id": self._event_run_id,
        }
        self.state["frames"][slot if slot in ("model", "A", "B") else "model"] = info
        self._frames[frame_id] = info
        if len(self._frames) > 400:
            for key in list(self._frames.keys())[:100]:
                self._frames.pop(key, None)

    def events_since(self, after: int) -> dict:
        with self.lock:
            events = list(self._events)
            if not events:
                return {"events": [], "next_seq": self._seq + 1, "last_seq": self._seq}
            oldest = events[0]["seq"]
            if after < oldest - 1:
                return {
                    "events": [],
                    "reset_required": True,
                    "next_seq": self._seq + 1,
                    "last_seq": self._seq,
                }
            selected = [event for event in events if event["seq"] > after][:200]
            return {
                "events": selected,
                "next_seq": (selected[-1]["seq"] + 1) if selected else self._seq + 1,
                "last_seq": self._seq,
            }

    # ------------------------------------------------------------------ 状态输出
    def public_state(self) -> dict:
        with self.lock:
            run = dict(self.state["run"]) if self.state["run"] else None
            if run and run.get("started_at"):
                run["elapsed_ms"] = max(0, wall_ms() - self._iso_to_ms(run["started_at"]))
            data = json.loads(json.dumps(self.state, ensure_ascii=False))
            data["run"] = run
            data["server_time_ms"] = wall_ms()
            data["last_seq"] = self._seq
            data["uptime_ms"] = max(0, wall_ms() - self.started_ms)
            data["poll_ms"] = self.poll_ms
            data["worker"] = {
                "alive": bool(self._worker is not None and self._worker.alive),
                "pid": getattr(self._worker, "pid", None),
            }
            data["services"]["worker"] = "alive" if data["worker"]["alive"] else "none"
            if not data["worker"]["alive"]:
                data["worker"]["pid"] = None
            data["token_required"] = False
            return data

    @staticmethod
    def _iso_to_ms(value: str) -> int:
        try:
            return int(datetime.fromisoformat(value).timestamp() * 1000)
        except Exception:  # noqa: BLE001
            return wall_ms()

    # ------------------------------------------------------------------ worker 生命周期
    def _spawn_worker(self) -> Any:
        from supervisor import WorkerProcess

        worker = WorkerProcess(
            bench_dir=BENCH_DIR,
            project_root=self.project_root,
            python_path=os.path.abspath(BENCH_DIR / self.python_path),
            log_path=LOGS_DIR / "worker.log",
            on_message=lambda message: self._on_worker_message(message) if self._worker is worker else None,
            on_exit=lambda reason: self._on_worker_exit(reason) if self._worker is worker else None,
        )
        self._worker = worker
        self._worker_ready.clear()
        self._worker_hello = {}
        worker.start()
        self.write_log("INFO", "supervisor", f"worker spawned pid={worker.pid}")
        return worker

    def ensure_worker(self) -> Any:
        return self._ensure_worker_locked()


    def _ensure_worker_locked(self) -> Any:
        """并发安全版本：同一时刻只允许一次 spawn，其余调用等待 hello。"""
        worker: Any = None
        spawned = False
        with self.lock:
            worker = self._worker
            if worker is not None and worker.alive:
                return worker
            if self._spawning:
                worker = self._worker
            else:
                self._spawning = True
                worker = self._spawn_worker()
                spawned = True
        if spawned:
            try:
                if not self._worker_ready.wait(15.0):
                    self.write_log("WARN", "supervisor", "worker hello 超时")
            finally:
                with self.lock:
                    self._spawning = False
        else:
            self._worker_ready.wait(15.0)
        return worker

    def _worker_alive(self) -> bool:
        worker = self._worker
        return bool(worker is not None and worker.alive)

    def stop_worker(self, reason: str = "recycle") -> None:
        with self.lock:
            worker = self._worker
            if reason in ("idle_recycle", "idle_watchdog", "debug_start_failed", "run_start_failed") and (self._run_active or self._debug_active or self._preview_active):
                return
            if worker is None or not worker.alive:
                return
            self._worker = None
            self.state["services"]["worker"] = "none"
            self._worker_ready.clear()
        status = worker.stop(
            grace_ms=self.worker_stop_grace_ms,
            kill_wait_ms=self.worker_kill_wait_ms,
        )
        self.write_log("INFO", "supervisor", f"worker stop: {status} ({reason})")

    def _on_worker_exit(self, reason: str) -> None:
        self.write_log("INFO", "supervisor", f"worker exited: {reason}")
        with self.lock:
            self._worker_ready.clear()
            self.state["services"]["worker"] = "none"
            active_run = self._run_active
            run_id = self._run_id
            debug_active = self._debug_active
            debug_id = self._debug_run_id
        if active_run and run_id:
            with self.lock:
                run = self.state.get("run") or {}
                status = str(run.get("status") or "")
                if status in ("starting", "running", "stopping"):
                    self.state["run"] = {
                        **run,
                        "status": "failed",
                        "ended_at": wall_iso(),
                        "error": f"worker 退出：{reason}",
                    }
                    self._run_active = False
                    self._close_run_files()
                    self._release_active_uploads()
            self.append_event(
                "error",
                {"code": "worker_exited", "message": f"工作进程退出：{reason}", "retryable": True},
            )
        if debug_active and debug_id:
            with self.lock:
                debug = self.state.get("debug") or {}
                if str(debug.get("status") or "") in ("starting", "running", "stopping"):
                    self.state["debug"] = {**debug, "status": "failed", "error": f"worker 退出：{reason}"}
                    self._debug_active = False
                    self._close_run_files()
                    self._release_active_uploads()
        self._sync_services()

    def _sync_services(self) -> None:
        with self.lock:
            run = self.state.get("run") or {}
            if self._run_active:
                status = str(run.get("status") or "running")
                self.state["services"]["interaction"] = "stopping" if status == "stopping" else "running"
            elif self._debug_active:
                self.state["services"]["interaction"] = "debug"
            elif self._preview_active:
                self.state["services"]["interaction"] = "preview"
            else:
                self.state["services"]["interaction"] = "idle"
            self.state["services"]["worker"] = "alive" if self._worker_alive() else "none"
            self.state["services"]["model"] = self._model_state
            self.state["control"] = {"locked": False}

    # ------------------------------------------------------------------ worker 消息
    def _on_worker_message(self, message: dict) -> None:
        kind = message.get("t")
        try:
            if kind == "hello":
                self._handle_hello(message)
            elif kind == "event":
                self._handle_worker_event(message.get("event") or {})
            elif kind == "state":
                self._handle_state_patch(message.get("patch") or {})
            elif kind == "preview":
                self._handle_preview(message)
            elif kind == "log":
                self.append_event(
                    "log",
                    {
                        "level": str(message.get("level") or "INFO").upper(),
                        "module": str(message.get("module") or "worker"),
                        "message": str(message.get("message") or ""),
                    },
                )
            elif kind == "exit":
                self.write_log("INFO", "supervisor", "worker 主动退出")
        except Exception:  # noqa: BLE001
            self.write_log("WARN", "server", f"worker message 处理失败: {traceback.format_exc(limit=3)}")

    def _handle_hello(self, message: dict) -> None:
        with self.lock:
            self._worker_hello = dict(message)
            capabilities = message.get("capabilities") or {}
            if isinstance(capabilities, dict):
                for key in ("contract", "speaker"):
                    if key in capabilities:
                        self.state["capabilities"][key] = bool(capabilities[key])
            self.state["services"]["worker"] = "alive"
            self._worker_ready.set()
        self.write_log("INFO", "supervisor", f"worker hello pid={message.get('pid')}")

    def _handle_worker_event(self, event: dict) -> None:
        etype = str(event.get("type") or "")
        payload = event.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        source_run_id = str(event.get("run_id") or "")
        if self._event_kind == "debug":
            for key in ("image_id", "image_a", "image_b"):
                if key in payload:
                    payload.pop(key, None)
        if source_run_id:
            current_run_id = self._event_run_id
            # 上一轮会话的残留线程（旧音频帧/识别分段/回答增量）仍在飞时，事件不得写入新一轮；
            # 收尾后 `_event_run_id` 为 None（空闲窗口）同样不是"当前轮"，也必须丢弃。
            if not current_run_id or source_run_id != str(current_run_id):
                return
        self.append_event(etype, payload, turn_id=event.get("turn_id"))

    def _handle_state_patch(self, patch: dict) -> None:
        if not isinstance(patch, dict):
            return
        run_status = None
        with self.lock:
            for key in ("run", "debug", "preview", "devices", "services", "timings"):
                if key not in patch:
                    continue
                value = patch[key]
                if key == "run":
                    # 本轮已结束或属于上一轮的迟到补丁不得改写当前状态（清单 24/43）。
                    if not self._run_active:
                        continue
                    if isinstance(value, dict) and value.get("id") and str(value["id"]) != str(self._run_id):
                        continue
                if key == "debug":
                    if not self._debug_active:
                        continue
                    if (
                        isinstance(value, dict)
                        and value.get("run_id")
                        and str(value["run_id"]) != str(self._debug_run_id)
                    ):
                        continue
                if isinstance(value, dict):
                    current = self.state.get(key)
                    if isinstance(current, dict):
                        merged = dict(current)
                        merged.update(value)
                        self.state[key] = merged
                    else:
                        self.state[key] = dict(value)
                    if key == "run":
                        run_status = value.get("status")
                        if value.get("id"):
                            self.state.setdefault("run", {})
                    if key == "debug":
                        self.state.setdefault("debug", {})
                else:
                    self.state[key] = value
        self._sync_services()
        if run_status in ("completed", "failed", "interrupted"):
            self._finalize_run(str(run_status))
        debug_status = None
        with self.lock:
            debug = self.state.get("debug")
            if isinstance(debug, dict):
                debug_status = debug.get("status")
        if debug_status in ("completed", "failed", "interrupted"):
            self._finalize_debug(str(debug_status))

    def _handle_preview(self, message: dict) -> None:
        import base64

        raw = message.get("jpeg_b64") or ""
        try:
            data = base64.b64decode(raw)
        except Exception:  # noqa: BLE001
            return
        frame_id = str(message.get("frame_id") or "")
        with self.lock:
            self._last_preview = (data, frame_id, int(message.get("ts_mono_ms") or 0))
            self.state["preview"]["last_frame_ts"] = int(message.get("ts_mono_ms") or 0)
            self.state["preview"]["active"] = True
            self.state["preview"]["frames"] = int(self.state["preview"].get("frames") or 0) + 1
            self._preview_active = True

    # ------------------------------------------------------------------ run
    def start_run(self, mode: str = "live") -> dict:
        with self.lock:
            if self._run_active:
                raise HttpError(409, "busy", "已有综合测试运行中，不能重复开始", retryable=True)
            if self._debug_active:
                raise HttpError(409, "busy", "功能调试运行中，不能开始综合测试", retryable=True)
            run_id = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
            run_dir = RUNS_DIR / run_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "media").mkdir(parents=True, exist_ok=True)
            (run_dir / "logs").mkdir(parents=True, exist_ok=True)
            self._run_active = True
            self._run_id = run_id
            self._event_kind = "run"
            self._event_run_id = run_id
            self._reset_run_state()
            self.state["run"] = {
                "id": run_id,
                "mode": "live" if mode != "replay" else "replay",
                "status": "starting",
                "started_at": wall_iso(),
            }
            self._open_run_files(run_dir)
        self.write_log("INFO", "run", f"run start {run_id}")
        try:
            worker = self.ensure_worker()
            worker.call_async(
                "run.start",
                {"run_id": run_id, "run_dir": str(run_dir)},
                timeout=self.worker_start_timeout_ms / 1000.0,
                on_done=lambda result, error: self._on_run_start_done(run_id, result, error),
            )
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self._run_active = False
                self.state["run"] = {
                    "id": run_id,
                    "mode": "live",
                    "status": "failed",
                    "started_at": wall_iso(),
                    "ended_at": wall_iso(),
                    "error": f"worker 启动失败：{exc}",
                }
                self._close_run_files()
            raise HttpError(503, "worker_unavailable", f"工作进程不可用：{exc}", retryable=True) from exc
        return {"run_id": run_id, "status": "starting"}

    def _reset_run_state(self) -> None:
        with self.lock:
            self._sessions = []
            self._frames = {}
            self.state["frames"] = {"model": None, "A": None, "B": None}
            self.state["timings"] = {field: None for field in TIMING_FIELDS}
            self.state["last_error"] = None
            self.state["counts"] = {"events": 0, "turns": 0}
            self._last_preview = None

    def _open_run_files(self, run_dir: Path) -> None:
        try:
            self._events_fp = open(run_dir / "events.jsonl", "a", encoding="utf-8", buffering=1)
        except OSError:
            self._events_fp = None
        try:
            self._metrics_fp = open(run_dir / "metrics.jsonl", "a", encoding="utf-8", buffering=1)
        except OSError:
            self._metrics_fp = None

    def _close_run_files(self) -> None:
        for name in ("_events_fp", "_metrics_fp"):
            handle = getattr(self, name, None)
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
                setattr(self, name, None)
        self._event_run_id = None
        self._event_kind = None

    def _on_run_start_done(self, run_id: str, result: dict | None, error: dict | None) -> None:
        with self.lock:
            if self._run_id != run_id or not self._run_active:
                return
        if error:
            message = str(error.get("message") or "启动失败")
            with self.lock:
                # 复查放在与写入相同的锁内：两次加锁之间该轮可能已按取消收尾，
                # 迟到错误不得把 interrupted 覆盖成 failed（清单 43）。
                if self._run_id != run_id or not self._run_active:
                    return
                self._run_active = False
                run = dict(self.state.get("run") or {})
                run.update(
                    {
                        "id": run_id,
                        "status": "failed",
                        "ended_at": wall_iso(),
                        "error": message,
                    }
                )
                self.state["run"] = run
                self._close_run_files()
            self.append_event(
                "error",
                {"code": str(error.get("code") or "engine_failed"), "message": message, "retryable": True},
            )
            self.write_log("WARN", "run", f"run start failed: {message}")
            threading.Thread(target=self.stop_worker, args=("run_start_failed",), daemon=True).start()
        else:
            self.write_log("INFO", "run", f"run {run_id} ack: {result}")

    def stop_run(self, run_id: str | None = None) -> dict:
        with self.lock:
            if not self._run_active:
                return {"status": "idle"}
            active_id = self._run_id
            if run_id and run_id != active_id:
                raise HttpError(409, "run_mismatch", "run_id 与当前任务不一致")
            self.state["run"] = {**(self.state.get("run") or {}), "status": "stopping"}
            worker = self._worker
        self.write_log("INFO", "run", f"run stop {active_id}")
        if worker is not None and worker.alive:
            try:
                worker.send("run.stop", {"reason": "user"})
            except Exception as exc:  # noqa: BLE001
                self.write_log("WARN", "run", f"run.stop 发送失败: {exc}")
        return {"run_id": active_id, "status": "stopping"}

    def _finalize_run(self, status: str) -> None:
        with self.lock:
            if not self._run_active:
                return
            self._run_active = False
            run = dict(self.state.get("run") or {})
            run["status"] = status
            run.setdefault("ended_at", wall_iso())
            self.state["run"] = run
            self._close_run_files()
            self._release_active_uploads()
            self.state["services"]["interaction"] = "idle"
        self.append_event(
            "log",
            {"level": "INFO", "module": "run", "message": f"运行结束：{status}"},
        )
        self.write_log("INFO", "run", f"run finished: {status}")
        self._watch_worker_exit()

    def _watch_worker_exit(self) -> None:
        """回收运行结束后仍空闲的 worker；新运行中的 worker 保留。"""

        def _check() -> None:
            time.sleep(5.0)
            with self.lock:
                busy = self._run_active or self._debug_active or self._preview_active
                worker = self._worker
            if not busy and worker is not None and worker.alive:
                self.stop_worker("idle_watchdog")

        threading.Thread(target=_check, name="worker-watchdog", daemon=True).start()

    # ------------------------------------------------------------------ debug
    def start_debug(self, module: str, input_data: dict) -> dict:
        module = str(module or "").strip()
        if not module:
            raise HttpError(400, "invalid_request", "缺少 module")
        specs = self.debug_modules()
        allowed = {item["id"] for item in specs}
        if module not in allowed:
            raise HttpError(400, "invalid_module", f"未知调试模块：{module}")
        with self.lock:
            if self._run_active:
                raise HttpError(409, "busy", "综合测试运行中，不能开始功能调试", retryable=True)
            if self._debug_active:
                raise HttpError(409, "busy", "已有功能调试运行中", retryable=True)
            debug_id = f"dbg-{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(2)}"
            run_dir = RUNS_DIR / debug_id
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "media").mkdir(parents=True, exist_ok=True)
            images = self._freeze_images()
            self._debug_active = True
            self._debug_run_id = debug_id
            self._event_kind = "debug"
            self._event_run_id = debug_id
            self.state["debug"] = {
                "run_id": debug_id,
                "module": module,
                "status": "starting",
                "started_at": wall_iso(),
            }
            self.state["timings"] = {field: None for field in TIMING_FIELDS}
            self.state["frames"] = {"model": None, "A": None, "B": None}
            self._open_run_files(run_dir)
        self.write_log("INFO", "debug", f"debug start {debug_id} module={module}")
        try:
            worker = self.ensure_worker()
            worker.call_async(
                "debug.start",
                {
                    "run_id": debug_id,
                    "run_dir": str(run_dir),
                    "module": module,
                    "input": input_data if isinstance(input_data, dict) else {},
                    "images": images,
                },
                timeout=self.worker_start_timeout_ms / 1000.0,
                on_done=lambda result, error: self._on_debug_start_done(debug_id, result, error),
            )
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self._debug_active = False
                self.state["debug"] = {
                    "run_id": debug_id,
                    "module": module,
                    "status": "failed",
                    "started_at": wall_iso(),
                    "ended_at": wall_iso(),
                    "error": f"worker 启动失败：{exc}",
                }
                self._close_run_files()
                self._release_active_uploads()
            raise HttpError(503, "worker_unavailable", f"工作进程不可用：{exc}", retryable=True) from exc
        return {"run_id": debug_id, "module": module, "status": "starting"}

    def _on_debug_start_done(self, debug_id: str, result: dict | None, error: dict | None) -> None:
        if error:
            message = str(error.get("message") or "调试启动失败")
            with self.lock:
                # 与运行入口同一条规则：该次调试已结束（例如取消）时丢弃迟到错误。
                if self._debug_run_id != debug_id or not self._debug_active:
                    return
                self._debug_active = False
                debug = dict(self.state.get("debug") or {})
                debug.update({"status": "failed", "ended_at": wall_iso(), "error": message})
                self.state["debug"] = debug
                self._close_run_files()
                self._release_active_uploads()
            self.append_event(
                "error",
                {"code": str(error.get("code") or "debug_failed"), "message": message, "retryable": True},
            )
            threading.Thread(target=self.stop_worker, args=("debug_start_failed",), daemon=True).start()

    def stop_debug(self) -> dict:
        with self.lock:
            if not self._debug_active:
                return {"status": "idle"}
            debug_id = self._debug_run_id
            self.state["debug"] = {**(self.state.get("debug") or {}), "status": "stopping"}
            worker = self._worker
        if worker is not None and worker.alive:
            try:
                worker.send("debug.stop", {})
            except Exception as exc:  # noqa: BLE001
                self.write_log("WARN", "debug", f"debug.stop 发送失败: {exc}")
        return {"run_id": debug_id, "status": "stopping"}

    def debug_action(self, action: str, input_data: dict) -> dict:
        with self.lock:
            if not self._debug_active:
                raise HttpError(409, "no_debug", "没有运行中的调试任务", retryable=True)
            worker = self._worker
        if worker is None or not worker.alive:
            raise HttpError(503, "worker_unavailable", "工作进程不可用", retryable=True)
        try:
            result = worker.call(
                "debug.action",
                {"action": action, "input": input_data if isinstance(input_data, dict) else {}},
                timeout=30.0,
            )
        except Exception as exc:  # noqa: BLE001
            error = getattr(exc, "error", {}) or {}
            raise HttpError(
                400 if error.get("code") in ("invalid_request", "missing_image") else 503,
                str(error.get("code") or "debug_action_failed"),
                str(error.get("message") or f"调试动作失败：{exc}"),
                retryable=True,
            ) from exc
        return result

    def _finalize_debug(self, status: str) -> None:
        with self.lock:
            if not self._debug_active:
                return
            self._debug_active = False
            debug = dict(self.state.get("debug") or {})
            debug["status"] = status
            debug.setdefault("ended_at", wall_iso())
            self.state["debug"] = debug
            self._close_run_files()
            self._release_active_uploads()
            self.state["services"]["interaction"] = "idle"
        self.write_log("INFO", "debug", f"debug finished: {status}")
        self._watch_worker_exit()

    def _freeze_images(self) -> dict:
        """冻结当前 A/B 上传图（若存在），标记为被活动任务引用。"""
        images: dict[str, dict] = {}
        with self.lock:
            slots = self.uploads.slots()
            for slot, meta in slots.items():
                if not meta:
                    continue
                image_id = str(meta.get("image_id") or "")
                try:
                    path, _mime = self.uploads.get(image_id)
                except KeyError:
                    continue
                images[slot] = {"image_id": image_id, "path": str(path)}
                self.uploads.set_in_use(image_id, True)
                self.active_upload_ids.add(image_id)
        return images

    def _release_active_uploads(self) -> None:
        with self.lock:
            ids = list(self.active_upload_ids)
            self.active_upload_ids.clear()
        for image_id in ids:
            try:
                self.uploads.set_in_use(image_id, False)
            except Exception:  # noqa: BLE001
                pass
        self._sync_images_state()

    # ------------------------------------------------------------------ preview
    def start_preview(self) -> dict:
        worker = self.ensure_worker()
        with self.lock:
            self.state["preview"]["error"] = None
            self.state["preview"]["active"] = True
            self._preview_active = True
            self._last_preview = None
        try:
            worker.call("preview.start", {}, timeout=30.0)
        except Exception as exc:  # noqa: BLE001
            with self.lock:
                self.state["preview"]["active"] = False
                self._preview_active = False
                self.state["preview"]["error"] = f"预览启动失败：{exc}"
            raise HttpError(503, "camera_unavailable", f"相机预览启动失败：{exc}", retryable=True) from exc
        self.write_log("INFO", "preview", "preview start")
        return {"active": True}

    def stop_preview(self) -> dict:
        with self.lock:
            worker = self._worker
            self.state["preview"]["active"] = False
            self._preview_active = False
            self._last_preview = None
        if worker is not None and worker.alive:
            try:
                worker.call("preview.stop", {}, timeout=10.0)
            except Exception:  # noqa: BLE001
                pass
        self.write_log("INFO", "preview", "preview stop")
        self._maybe_recycle_worker()
        return {"active": False}

    def _maybe_recycle_worker(self) -> None:
        with self.lock:
            busy = self._run_active or self._debug_active or self._preview_active
            worker = self._worker
        if not busy and worker is not None and worker.alive:
            threading.Thread(target=self.stop_worker, args=("idle_recycle",), daemon=True).start()

    def preview_bytes(self) -> tuple[bytes, str] | None:
        with self.lock:
            if not self._preview_active:
                return None
            return self._last_preview[0:2] if self._last_preview else None

    # ------------------------------------------------------------------ 上传
    def save_upload(self, raw: bytes, slot: str) -> dict:
        from media import UploadError

        try:
            meta = self.uploads.save(raw, slot)
        except UploadError as exc:
            status = 413 if exc.code == 413 else (422 if exc.code == 422 else 415)
            raise HttpError(status, "invalid_image", exc.message) from exc
        except ValueError as exc:
            raise HttpError(400, "invalid_request", str(exc)) from exc
        self._sync_images_state()
        self.write_log("INFO", "upload", f"upload {meta.get('image_id')} slot={slot}")
        return self._public_image(self.uploads.slots().get(slot) or meta)

    def delete_upload(self, image_id: str) -> dict:
        ok = self.uploads.delete(image_id)
        self._sync_images_state()
        if not ok:
            raise HttpError(404, "not_found", "图片不存在")
        return {"deleted": True}

    def upload_file(self, image_id: str) -> tuple[Path, str] | None:
        try:
            return self.uploads.get(image_id)
        except KeyError:
            return None

    def frame_file(self, frame_id: str) -> Path | None:
        with self.lock:
            info = self._frames.get(frame_id)
            run_id = str(info.get("run_id") or "") if info else ""
            file = str(info.get("file") or "") if info else ""
        if not info or not file:
            return None
        base = (RUNS_DIR / run_id).resolve() if run_id else None
        candidate = (base / file).resolve() if base else None
        if candidate is None or not candidate.is_file():
            return None
        try:
            candidate.relative_to(RUNS_DIR.resolve())
        except ValueError:
            return None
        return candidate

    # ------------------------------------------------------------------ debug 元数据
    def debug_modules(self) -> list[dict]:
        try:
            import debug as debug_module
        except Exception as exc:  # noqa: BLE001
            self.write_log("WARN", "debug", f"debug.py 不可用: {exc}")
            return []
        specs = getattr(debug_module, "MODULE_SPECS", None)
        if not isinstance(specs, list):
            return []
        return [dict(item) for item in specs if isinstance(item, dict)]

    def session_snapshot(self) -> dict:
        with self.lock:
            turns = [
                {
                    "turn_id": turn.get("turnId"),
                    "user": turn.get("userText", ""),
                    "assistant": turn.get("assistantText", ""),
                    "assistant_done": not bool(turn.get("generating")),
                    "cancelled": bool(turn.get("cancelled")),
                }
                for turn in self._sessions
            ]
            return {
                "client_id": f"page-{secrets.token_hex(8)}",
                "token": self.token,
                "last_seq": self._seq,
                "turns": turns,
                "run": dict(self.state["run"]) if self.state["run"] else None,
                "debug": dict(self.state["debug"]) if self.state["debug"] else None,
                "server_time_ms": wall_ms(),
            }

    # ------------------------------------------------------------------ 控制者/心跳
    def heartbeat(self, client_id: str) -> dict:
        client_id = str(client_id or "").strip()
        if not client_id or len(client_id) > 80:
            raise HttpError(400, "invalid_request", "缺少 client_id")
        now = time.monotonic()
        with self.lock:
            entry = self.heartbeats.get(client_id)
            if entry is None:
                entry = {"first_seen": now, "last_seen": now}
                self.heartbeats[client_id] = entry
            if self.controller is None:
                self.controller = client_id
                self.write_log("INFO", "client", f"controller 接管: {client_id}")
            entry["last_seen"] = now
            controller = self.controller == client_id
        return {
            "controller": controller,
            "owner": controller,
            "client_id": client_id,
            "server_time_ms": wall_ms(),
            "poll_ms": self.poll_ms,
            "active": {
                "run": self._run_id if self._run_active else None,
                "debug": self._debug_run_id if self._debug_active else None,
                "preview": bool(self._preview_active),
            },
        }

    def claim_control(self, client_id: str) -> dict:
        with self.lock:
            if self.controller != client_id and (
                self._run_active or self._debug_active or self._preview_active
            ):
                raise HttpError(409, "control_busy", "其他页面正在测试或预览，请先在原页面停止")
            self.controller = client_id
            self.heartbeats[client_id] = {"first_seen": time.monotonic(), "last_seen": time.monotonic()}
            self.write_log("INFO", "client", f"用户主动取得控制: {client_id}")
        return self.heartbeat(client_id)

    def require_controller(self, client_id: str) -> None:
        with self.lock:
            controller = self.controller
        if controller is not None and client_id and client_id != controller:
            raise HttpError(409, "client_readonly", "当前页面只读：已有其他页面控制测试台")

    def monitor_heartbeats(self) -> None:
        timeout_s = self.heartbeat_timeout_ms / 1000.0
        while not self.shutdown_event.wait(1.0):
            now = time.monotonic()
            lost: str | None = None
            with self.lock:
                controller = self.controller
                if controller is not None:
                    entry = self.heartbeats.get(controller)
                    last = entry["last_seen"] if entry else 0.0
                    if now - last > timeout_s:
                        lost = controller
                        self.controller = None
                        self.heartbeats.pop(controller, None)
            if lost:
                self.write_log("WARN", "client", f"控制器断开: {lost}")
                self.append_event(
                    "log",
                    {
                        "level": "WARN",
                        "module": "client",
                        "message": "控制页面断开超过 15 秒，停止活动任务",
                    },
                )
                self._stop_active_for_disconnect()
                self.append_event(
                    "log",
                    {
                        "level": "WARN",
                        "module": "client",
                        "message": "client_disconnected",
                    },
                )

    def _stop_active_for_disconnect(self) -> None:
        with self.lock:
            run_active = self._run_active
            run_id = self._run_id
            debug_active = self._debug_active
            worker = self._worker
            self._preview_active = False
            self.state["preview"]["active"] = False
        if debug_active:
            try:
                self.stop_debug()
            except Exception:  # noqa: BLE001
                pass
        if run_active and worker is not None and worker.alive:
            with self.lock:
                self.state["run"] = {
                    **(self.state.get("run") or {}),
                    "status": "stopping",
                    "stop_reason": "client_disconnected",
                }
            try:
                worker.send("run.stop", {"reason": "client_disconnected"})
            except Exception:  # noqa: BLE001
                pass
        if worker is not None and worker.alive:
            try:
                worker.send("preview.stop", {})
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------------ 后台采样
    def metrics_loop(self) -> None:
        interval = max(0.5, float(self.bench.get("metrics_interval_ms", 2000)) / 1000.0)
        while not self.shutdown_event.wait(interval):
            try:
                sample = self.metrics.sample()
            except Exception:  # noqa: BLE001
                continue
            metrics = {key: sample.get(key) for key in self.state["metrics"]}
            metrics["detail"] = sample.get("detail")
            with self.lock:
                self.state["metrics"] = metrics
                fp = self._metrics_fp
                if fp is not None:
                    try:
                        fp.write(json.dumps(sample, ensure_ascii=False) + "\n")
                        fp.flush()
                    except OSError:
                        pass

    def model_health_loop(self) -> None:
        while not self.shutdown_event.wait(1.0):
            now = wall_ms()
            with self.lock:
                if now - self._model_checked_ms < self.model_health_cache_ms:
                    continue
                self._model_checked_ms = now
            ok, detail = self._probe_model()
            with self.lock:
                self._model_state = "ready" if ok else "unavailable"
                self._model_detail = detail
                self.state["services"]["model"] = self._model_state
                self.state["services"]["model_detail"] = detail

    def _probe_model(self) -> tuple[bool, str]:
        try:
            with urllib.request.urlopen(self.model_health_url, timeout=self.model_health_timeout_s) as response:
                body = response.read(2048)
                status = getattr(response, "status", 200)
            if status != 200:
                return False, f"HTTP {status}"
            data = json.loads(body.decode("utf-8", errors="replace"))
            if isinstance(data, dict) and str(data.get("status", "")).lower() == "ok":
                return True, "ok"
            return True, str(data)[:200]
        except Exception as exc:  # noqa: BLE001
            return False, f"{exc}"

    # ------------------------------------------------------------------ 关闭
    def shutdown(self) -> None:
        self.shutdown_event.set()
        self.write_log("INFO", "server", "server shutting down")
        with self.lock:
            run_active = self._run_active
            debug_active = self._debug_active
            worker = self._worker
            self._preview_active = False
        if run_active and worker is not None and worker.alive:
            try:
                worker.send("run.stop", {"reason": "shutdown"})
            except Exception:  # noqa: BLE001
                pass
            time.sleep(0.5)
        if debug_active and worker is not None and worker.alive:
            try:
                worker.send("debug.stop", {})
            except Exception:  # noqa: BLE001
                pass
        self.stop_worker("server_shutdown")
        with self.lock:
            self._close_run_files()
            self._release_active_uploads()


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
SERVER: BenchServer | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "TestbenchHTTP/1.0"
    protocol_version = "HTTP/1.1"

    # --- 基础 -----------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        return

    def setup(self) -> None:
        super().setup()
        try:
            self.connection.settimeout(self.server.http_read_timeout_s if hasattr(self.server, "http_read_timeout_s") else 10.0)  # type: ignore[attr-defined]
        except OSError:
            pass

    def _ctx(self) -> BenchServer:
        if SERVER is None:  # pragma: no cover - 仅防御
            raise HttpError(503, "not_ready", "服务器未就绪")
        return SERVER

    def _allowed_hosts(self) -> set[str]:
        ctx = self._ctx()
        return {f"127.0.0.1:{ctx.port}", f"localhost:{ctx.port}", f"[::1]:{ctx.port}"}

    def _allowed_origins(self) -> set[str]:
        ctx = self._ctx()
        return {
            f"http://127.0.0.1:{ctx.port}",
            f"http://localhost:{ctx.port}",
        }

    def _check_host(self) -> None:
        host = str(self.headers.get("Host") or "")
        if host not in self._allowed_hosts():
            raise HttpError(403, "forbidden_host", "只允许本机回环访问")

    def _check_control(self, *, require_owner: bool = True) -> str:
        self._check_host()
        ctx = self._ctx()
        origin = self.headers.get("Origin")
        if origin:
            if origin not in self._allowed_origins():
                raise HttpError(403, "forbidden_origin", "跨站请求被拒绝")
        else:
            token = str(self.headers.get("X-Bench-Token") or "")
            if not secrets.compare_digest(token, ctx.token):
                raise HttpError(403, "forbidden_origin", "缺少同源凭据")
        client_id = str(self.headers.get("X-Bench-Client") or "").strip()
        if not client_id:
            raise HttpError(400, "invalid_request", "缺少 X-Bench-Client")
        if require_owner:
            ctx.require_controller(client_id)
        return client_id

    def _read_body(self, limit: int) -> bytes:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise HttpError(400, "invalid_request", "Content-Length 非法") from exc
        if length < 0:
            raise HttpError(400, "invalid_request", "Content-Length 非法")
        if length > limit:
            raise HttpError(413, "too_large", f"请求体超过上限 {limit} 字节")
        chunks: list[bytes] = []
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                raise HttpError(400, "invalid_request", "请求体不完整")
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _read_json(self) -> dict:
        raw = self._read_body(self._ctx().json_max_bytes)
        if not raw:
            return {}
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise HttpError(400, "invalid_json", "JSON 解析失败") from exc
        if not isinstance(data, dict):
            raise HttpError(400, "invalid_json", "JSON 顶层必须是对象")
        return data

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_bytes(self, status: int, data: bytes, content_type: str, *, cache: bool = False) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "public, max-age=0" if cache else "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        try:
            self.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_error(self, error: HttpError) -> None:
        self._send_json(
            error.status,
            {
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "retryable": error.retryable,
                }
            },
        )

    def _dispatch(self, method: str) -> None:
        try:
            parsed = urlsplit(self.path)
            path = parsed.path
            query = dict(
                item.split("=", 1) if "=" in item else (item, "")
                for item in (parsed.query or "").split("&")
                if item
            )
            if method == "GET":
                self._handle_get(path, query)
            elif method == "POST":
                self._handle_post(path, query)
            elif method == "DELETE":
                self._handle_delete(path)
            else:
                self._send_error(HttpError(405, "method_not_allowed", "不支持的方法"))
        except HttpError as error:
            self._send_error(error)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as exc:  # noqa: BLE001
            self._ctx().write_log("WARN", "http", f"{method} {self.path} 失败: {exc}")
            self._send_error(HttpError(500, "internal_error", "服务器内部错误"))

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    # --- GET -----------------------------------------------------------
    def _handle_get(self, path: str, query: dict) -> None:
        self._check_host()
        ctx = self._ctx()
        if path in ("/", "/index.html"):
            return self._serve_static("index.html")
        if path == "/app.js":
            return self._serve_static("app.js")
        if path == "/favicon.ico":
            return self._send_bytes(204, b"", "image/x-icon")
        if path == "/api/devices":
            from devices import inventory, selection
            choices, errors = inventory()
            return self._send_json(200, {"choices": choices, "selected": selection(), "errors": errors})
        if path == "/api/health":
            with ctx.lock:
                worker_alive = bool(ctx._worker is not None and ctx._worker.alive)
                model = ctx._model_state
                uptime = max(0, wall_ms() - ctx.started_ms)
            return self._send_json(
                200,
                {
                    "status": "ok",
                    "ready": True,
                    "time": wall_iso(),
                    "server_time_ms": wall_ms(),
                    "uptime_ms": uptime,
                    "pid": os.getpid(),
                    "host": ctx.host,
                    "port": ctx.port,
                    "worker": {"alive": worker_alive},
                    "services": {"model": model},
                },
            )
        if path == "/api/state":
            return self._send_json(200, ctx.public_state())
        if path == "/api/events":
            try:
                after = int(query.get("after") or 0)
            except ValueError as exc:
                raise HttpError(400, "invalid_request", "after 必须是整数") from exc
            return self._send_json(200, ctx.events_since(max(0, after)))
        if path == "/api/session":
            return self._send_json(200, ctx.session_snapshot())
        if path == "/api/debug/modules":
            return self._send_json(200, {"modules": ctx.debug_modules()})
        if path == "/api/preview.jpg":
            with ctx.lock:
                active = ctx._preview_active
            if not active:
                return self._send_bytes(503, b"", "image/jpeg")
            data = ctx.preview_bytes()
            if data is None:
                return self._send_bytes(204, b"", "image/jpeg")
            return self._send_bytes(200, data[0], "image/jpeg")
        if path == "/api/state-snapshot":
            return self._send_json(200, ctx.session_snapshot())
        match = _FRAME_ROUTE.match(path)
        if match:
            frame_file = ctx.frame_file(match.group(1))
            if frame_file is None:
                raise HttpError(404, "not_found", "图像不存在")
            return self._send_bytes(200, frame_file.read_bytes(), "image/jpeg")
        match = _UPLOAD_ROUTE.match(path)
        if match:
            found = ctx.upload_file(match.group(1))
            if found is None:
                raise HttpError(404, "not_found", "图片不存在")
            return self._send_bytes(200, found[0].read_bytes(), found[1])
        raise HttpError(404, "not_found", "资源不存在")

    def _serve_static(self, name: str) -> None:
        ctx = self._ctx()
        allowed = set(ctx.bench.get("static_files") or [])
        if name not in STATIC_FILES or (allowed and name not in allowed):
            raise HttpError(404, "not_found", "资源不存在")
        path = (BENCH_DIR / name).resolve()
        if not path.is_file():
            raise HttpError(404, "not_found", f"{name} 不存在")
        content_type, _ = mimetypes.guess_type(name)
        if content_type is None:
            content_type = "application/octet-stream"
        if name.endswith(".js"):
            content_type = "application/javascript; charset=utf-8"
        if name.endswith(".html"):
            content_type = "text/html; charset=utf-8"
        return self._send_bytes(200, path.read_bytes(), content_type)

    # --- POST ----------------------------------------------------------
    def _handle_post(self, path: str, query: dict) -> None:
        ctx = self._ctx()
        if path == "/api/client/heartbeat":
            self._check_host()
            data = self._read_json()
            return self._send_json(200, ctx.heartbeat(str(data.get("client_id") or "")))
        if path == "/api/client/claim":
            client_id = self._check_control(require_owner=False)
            self._read_json()
            return self._send_json(200, ctx.claim_control(client_id))
        self._check_control()
        if path == "/api/devices":
            from devices import inventory, save
            data = self._read_json()
            with ctx.lock:
                if ctx._run_active or ctx._debug_active or ctx._preview_active or ctx._spawning or ctx._worker_alive():
                    raise HttpError(409, "device_busy", "请先停止测试、调试和预览后再切换设备")
                choices, errors = inventory()
                try:
                    selected = save(data, choices)
                except ValueError as exc:
                    raise HttpError(422, "device_unavailable", str(exc))
            return self._send_json(200, {"selected": selected, "errors": errors})
        if path == "/api/run/start":
            data = self._read_json()
            mode = str(data.get("mode") or "live")
            cached = self._cached_response("run.start", data)
            if cached:
                return cached
            try:
                result = ctx.start_run(mode)
            except HttpError as error:
                self._store_response("run.start", data, error)
                raise
            body = {
                "run_id": result["run_id"],
                "status": result["status"],
                "server_time_ms": wall_ms(),
            }
            self._store_body("run.start", data, 202, body)
            return self._send_json(202, body)
        if path == "/api/run/stop":
            data = self._read_json()
            cached = self._cached_response("run.stop", data)
            if cached:
                return cached
            result = ctx.stop_run(str(data.get("run_id") or "") or None)
            body = {"status": result.get("status"), "server_time_ms": wall_ms()}
            if result.get("run_id"):
                body["run_id"] = result["run_id"]
            self._store_body("run.stop", data, 202, body)
            return self._send_json(202, body)
        if path == "/api/preview/start":
            data = self._read_json()
            cached = self._cached_response("preview.start", data)
            if cached:
                return cached
            result = ctx.start_preview()
            body = {"active": True, "server_time_ms": wall_ms()}
            self._store_body("preview.start", data, 202, body)
            return self._send_json(202, body)
        if path == "/api/preview/stop":
            data = self._read_json()
            cached = self._cached_response("preview.stop", data)
            if cached:
                return cached
            ctx.stop_preview()
            body = {"active": False, "server_time_ms": wall_ms()}
            self._store_body("preview.stop", data, 200, body)
            return self._send_json(200, body)
        if path == "/api/uploads":
            raw = self._read_body(ctx.upload_max_bytes)
            slot = str((query.get("slot") or "A")).upper()
            meta = ctx.save_upload(raw, slot)
            body = dict(meta)
            body["server_time_ms"] = wall_ms()
            return self._send_json(201, body)
        if path == "/api/debug/start":
            data = self._read_json()
            cached = self._cached_response("debug.start", data)
            if cached:
                return cached
            try:
                result = ctx.start_debug(
                    str(data.get("module") or ""),
                    data.get("input") if isinstance(data.get("input"), dict) else {},
                )
            except HttpError as error:
                self._store_response("debug.start", data, error)
                raise
            body = {
                "run_id": result["run_id"],
                "module": result["module"],
                "status": result["status"],
                "server_time_ms": wall_ms(),
            }
            self._store_body("debug.start", data, 202, body)
            return self._send_json(202, body)
        if path == "/api/debug/stop":
            data = self._read_json()
            result = ctx.stop_debug()
            return self._send_json(202, {"status": result.get("status")})
        if path == "/api/debug/action":
            data = self._read_json()
            result = ctx.debug_action(
                str(data.get("action") or ""),
                data.get("input") if isinstance(data.get("input"), dict) else {},
            )
            return self._send_json(200, {"result": result, "server_time_ms": wall_ms()})
        raise HttpError(404, "not_found", "资源不存在")

    # --- DELETE --------------------------------------------------------
    def _handle_delete(self, path: str) -> None:
        ctx = self._ctx()
        self._check_control()
        match = _UPLOAD_ROUTE.match(path)
        if match:
            result = ctx.delete_upload(match.group(1))
            return self._send_json(200, result)
        raise HttpError(404, "not_found", "资源不存在")

    # --- request_id 幂等 ----------------------------------------------
    def _request_key(self, action: str, data: dict) -> str | None:
        request_id = str(data.get("request_id") or "").strip()
        return f"{action}:{request_id}" if request_id else None

    def _cached_response(self, action: str, data: dict) -> Any:
        key = self._request_key(action, data)
        if not key:
            return None
        ctx = self._ctx()
        with ctx.lock:
            cached = ctx._request_cache.get(key)
        if not cached:
            return None
        if "error" in cached:
            raise HttpError(**cached["error"])
        return self._send_json(cached["status"], cached["body"])

    def _store_body(self, action: str, data: dict, status: int, body: dict) -> None:
        key = self._request_key(action, data)
        if not key:
            return
        ctx = self._ctx()
        with ctx.lock:
            ctx._request_cache[key] = {"status": status, "body": body}
            if key not in ctx._request_order:
                ctx._request_order.append(key)
                if len(ctx._request_order) > 100:
                    old = ctx._request_order.popleft()
                    ctx._request_cache.pop(old, None)

    def _store_response(self, action: str, data: dict, error: HttpError) -> None:
        key = self._request_key(action, data)
        if not key:
            return
        ctx = self._ctx()
        with ctx.lock:
            ctx._request_cache[key] = {
                "error": {
                    "status": error.status,
                    "code": error.code,
                    "message": error.message,
                    "retryable": error.retryable,
                }
            }


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
class BenchHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, address: tuple[str, int], ctx: BenchServer) -> None:
        self.ctx = ctx
        super().__init__(address, Handler)


def acquire_server_lock(path: Path):
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+", encoding="utf-8")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    handle.seek(0)
    handle.truncate()
    handle.write(str(os.getpid()))
    handle.flush()
    return handle


def main(argv: list[str] | None = None) -> int:
    instance_lock = acquire_server_lock(RUNTIME_DIR / "server.lock")
    if instance_lock is None:
        print("测试台已运行，拒绝启动第二个服务实例", file=sys.stderr)
        return 2
    try:
        return _serve(argv)
    finally:
        instance_lock.close()


def _serve(argv: list[str] | None = None) -> int:
    global SERVER

    for directory in (RUNS_DIR, UPLOADS_DIR, LOGS_DIR, RUNTIME_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    bench = load_bench()
    ctx = BenchServer(bench)
    SERVER = ctx
    try:
        httpd = BenchHTTPServer((ctx.host, ctx.port), ctx)
    except OSError as exc:
        ctx.write_log("ERROR", "server", f"监听 {ctx.host}:{ctx.port} 失败: {exc}")
        return 2
    httpd.http_read_timeout_s = ctx.http_read_timeout_s  # type: ignore[attr-defined]
    ctx.write_log(
        "INFO",
        "server",
        f"listening http://{ctx.host}:{ctx.port}/ pid={os.getpid()}",
    )
    print(f"http://{ctx.host}:{ctx.port}/", flush=True)

    def _signal_handler(signum: int, _frame: Any) -> None:
        ctx.write_log("INFO", "server", f"signal {signum}")
        ctx.shutdown_event.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _signal_handler)
        except (ValueError, OSError):
            pass

    serve_thread = threading.Thread(target=httpd.serve_forever, name="http", daemon=True)
    serve_thread.start()
    threads = [
        threading.Thread(target=ctx.metrics_loop, name="metrics", daemon=True),
        threading.Thread(target=ctx.model_health_loop, name="model-health", daemon=True),
        threading.Thread(target=ctx.monitor_heartbeats, name="heartbeat-monitor", daemon=True),
    ]
    for thread in threads:
        thread.start()
    try:
        while not ctx.shutdown_event.wait(1.0):
            pass
    except KeyboardInterrupt:  # pragma: no cover
        pass
    ctx.shutdown()
    httpd.shutdown()
    httpd.server_close()
    ctx.write_log("INFO", "server", "server stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
