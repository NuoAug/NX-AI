"""测试台工作进程：装配真实业务链路并执行综合测试 / 功能调试 / 相机预览。

协议（行 JSON，stdout 专供协议、业务 stderr 由 supervisor 落日志）：

server -> worker::

    {"id": "req1", "cmd": "run.start", "args": {...}}

worker -> server::

    {"t": "hello", "pid": 123, "capabilities": {...}}
    {"t": "ack", "id": "req1", "result": {...}}
    {"t": "error", "id": "req1", "error": {"code": "...", "message": "...", "retryable": false}}
    {"t": "event", "event": {"type": "...", "turn_id": "t1", "payload": {...}, "ts_mono_ms": 1}}
    {"t": "state", "patch": {...}}
    {"t": "preview", "jpeg_b64": "...", "frame_id": "...", "ts_mono_ms": 1}
    {"t": "log", "level": "INFO", "module": "worker", "message": "..."}
    {"t": "exit"}

命令：``run.start`` / ``run.stop`` / ``preview.start`` / ``preview.stop`` /
``debug.start`` / ``debug.stop`` / ``debug.action`` / ``status`` / ``shutdown``。

debug.py 契约（由本文件在命令处理时惰性导入）：

* ``MODULE_SPECS: list[dict]`` —— 每项
  ``{id, name, desc, fields: [{name,label,type,options?,required?,value?}], actions: [{name,label}]}``。
* ``class DebugRunner(ctx)`` —— ``start(module, input)`` 校验参数后启动后台线程；
  参数非法抛 ``DebugError(code, message)``；``stop()`` 置停止标记并等待；
  ``alive`` 属性；结束时调用 ``ctx.finished(status, error)``。
* 所有输出通过 ``ctx`` 的 ``emit/log/error/save_frame/load_image/camera_provider`` 完成。
"""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
import os
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

BENCH_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------
def mono_ms() -> int:
    return int(time.monotonic() * 1000)


def wall_ms() -> int:
    return int(time.time() * 1000)


def wall_iso(ms: int | None = None) -> str:
    moment = datetime.fromtimestamp((ms if ms is not None else wall_ms()) / 1000.0)
    return moment.astimezone().isoformat(timespec="seconds")


def load_bench_config() -> dict:
    path = BENCH_DIR / "Config" / "bench.json"
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("bench.json 顶层必须是对象")
    return data


BENCH: dict = {}
try:  # 允许在不完整工作副本中 import（仅用于单元测试的辅助函数）
    BENCH = load_bench_config()
except Exception as exc:  # noqa: BLE001
    print(f"[worker] bench.json load failed: {exc}", file=sys.stderr)


class BenchError(Exception):
    """带错误码的测试台错误。"""

    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.code = str(code)
        self.message = str(message)
        self.retryable = bool(retryable)
        super().__init__(message)


# ---------------------------------------------------------------------------
# stdout 协议
# ---------------------------------------------------------------------------
_out_lock = threading.Lock()


def emit_message(payload: dict) -> None:
    line = json.dumps(payload, ensure_ascii=False)
    with _out_lock:
        try:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
        except Exception:  # noqa: BLE001 - 管道断开时静默
            pass


def _close_quietly(obj: Any) -> None:
    if obj is None:
        return
    closer = getattr(obj, "close", None)
    if not callable(closer):
        return
    try:
        closer()
    except Exception:  # noqa: BLE001 - 关闭链每项独立
        pass


def _stop_quietly(obj: Any) -> None:
    if obj is None:
        return
    stopper = getattr(obj, "stop", None)
    if not callable(stopper):
        return
    try:
        result = stopper()
        if inspect.isawaitable(result):
            pass  # 调用方负责 await
    except Exception:  # noqa: BLE001
        pass


def check_model_health(bench: dict, *, timeout_s: float | None = None) -> tuple[bool, str]:
    url = str(bench.get("model_health_url") or "http://127.0.0.1:8080/health")
    timeout = (
        float(timeout_s)
        if timeout_s is not None
        else max(0.2, float(bench.get("model_health_timeout_ms", 2000)) / 1000.0)
    )
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(4096)
            status = getattr(response, "status", 200)
        if status != 200:
            return False, f"health HTTP {status}"
        try:
            data = json.loads(body.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return True, "health non-json"
        if isinstance(data, dict) and str(data.get("status", "")).lower() == "ok":
            return True, "ok"
        return True, f"health {data}"
    except urllib.error.URLError as exc:
        return False, f"model_unavailable: {exc.reason}"
    except Exception as exc:  # noqa: BLE001
        return False, f"model_unavailable: {exc}"


def make_camera(cfg: Any) -> Any:
    """创建真实相机（测试通过 monkeypatch 本函数注入 fake）。"""
    from aieyes.vision.camera import V4L2Camera

    return V4L2Camera(cfg.vision)


def load_image_frame(path: str | Path) -> Any:
    from aieyes.vision.file_camera import load_frame

    return load_frame(str(path))


# ---------------------------------------------------------------------------
# 运行会话
# ---------------------------------------------------------------------------
class RunSession:
    """一次综合测试：装配真实业务链路并管理启动/停止顺序。"""

    def __init__(self, worker: "Worker", run_id: str, run_dir: Path) -> None:
        self.worker = worker
        self.run_id = str(run_id)
        self.run_dir = Path(run_dir)
        self.cfg = worker.cfg
        self.bench = worker.bench
        self.started_ms = wall_ms()
        self.started_iso = wall_iso(self.started_ms)
        self.cancel_reason = ""
        self.status = "running"

        self._ready = threading.Event()
        self._done = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: BenchError | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stop_event: asyncio.Event | None = None

        self.bridge: Any = None
        self.evidence: Any = None
        self.frame_store: Any = None
        self.hub: Any = None
        self.engine: Any = None
        self.deps: Any = None
        self.scene: Any = None
        self.provider: Any = None
        self.vlm: Any = None
        self.ocr: Any = None
        self._sink_started = False
        self._engine_started = False
        self._contract_started = False
        self._frontend_started = False

    # --- 外部控制 -------------------------------------------------------
    def start(self, timeout_s: float) -> None:
        self._thread = threading.Thread(
            target=self._thread_main, name=f"run-{self.run_id}", daemon=True
        )
        self._thread.start()
        if not self._ready.wait(max(1.0, timeout_s)):
            self.cancel_reason = self.cancel_reason or "start_timeout"
            self.request_stop("start_timeout")
            raise BenchError("worker_timeout", "引擎启动超时", retryable=True)
        if self._error is not None:
            raise self._error

    def request_stop(self, reason: str = "user") -> None:
        if self.cancel_reason == "":
            self.cancel_reason = reason
        loop = self._loop
        event = self._stop_event
        if loop is not None and event is not None:
            try:
                loop.call_soon_threadsafe(event.set)
                return
            except RuntimeError:
                pass

    def wait_done(self, timeout_s: float) -> bool:
        return self._done.wait(max(0.0, timeout_s))

    # --- 线程/循环 ------------------------------------------------------
    def _thread_main(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception:  # noqa: BLE001
            traceback.print_exc()
        finally:
            self._done.set()

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()
        try:
            self._assemble()
            self.deps.sink.start()
            self._sink_started = True
            sample_rate = int(getattr(self.worker.cfg.audio, "sample_rate", 16000))
            self.worker.audio_out = {"status": "ready", "detail": f"{sample_rate} Hz 默认输出"}
            await self.engine.start()
            self._engine_started = True
            contract = getattr(self.deps, "contract", None)
            if contract is not None:
                result = contract.start()
                if inspect.isawaitable(result):
                    await result
                self._contract_started = True
            self.deps.frontend.start()
            self._frontend_started = True
            self.worker.audio_in = {"status": "ready", "detail": f"{sample_rate} Hz 默认输入"}
            self.worker.emit_state(
                {
                    "run": {
                        "id": self.run_id,
                        "mode": "live",
                        "status": "running",
                        "started_at": self.started_iso,
                    },
                    "devices": self.worker.devices_patch(),
                }
            )
            self.worker.log("INFO", "worker", f"run {self.run_id} running")
        except Exception as exc:  # noqa: BLE001
            self._error = BenchError("engine_failed", f"业务链路启动失败：{exc}")
            if self.worker.audio_out.get("status") == "ready" and not self._frontend_started:
                self.worker.audio_out = {"status": "error", "detail": str(exc)[:200]}
            if not self._frontend_started:
                self.worker.audio_in = {"status": "error", "detail": str(exc)[:200]}
            self.worker.log("WARN", "worker", f"run start failed: {exc}")
            traceback.print_exc()
        finally:
            self._ready.set()
            await self._cleanup()

    # --- 装配 -----------------------------------------------------------
    def _assemble(self) -> None:
        from aieyes.app import _make_source, build_engine
        from aieyes.audio.output import AudioSink
        from aieyes.audio.vad import EnergyVad
        from aieyes.chat import ChatSession
        from aieyes.events import EventBus
        from aieyes.logging_setup import JsonlEvidenceLogger
        from aieyes.scene.providers import CameraFrameProvider
        from aieyes.scene.service import SceneService
        from aieyes.speech.asr import StreamingAsr
        from aieyes.speech.frontend import SpeechFrontend
        from aieyes.speech.kws import SherpaKws
        from aieyes.speech.tts import PiperTts
        from aieyes.vision.ocr import OcrEngine
        from aieyes.vision.vlm import VlmClient

        from bridge import Bridge
        from media import CameraHubAdapter, FrameStore

        cfg = self.cfg
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "logs").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "media").mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.json").write_text(
            json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )

        self.frame_store = FrameStore(self.run_dir)
        self.bridge = Bridge(
            self.worker.emit_event,
            run_id=self.run_id,
            frame_store=self.frame_store,
        )
        self.evidence = JsonlEvidenceLogger(self.run_dir / "evidence.jsonl")
        self.hub = self.worker.ensure_hub()
        bus = EventBus()
        observed_bus = self.bridge.wrap_bus(bus)
        self.provider = self.bridge.wrap_provider(
            CameraFrameProvider(camera=CameraHubAdapter(self.hub))
        )
        self.vlm = VlmClient(cfg.llm, cfg.vlm)
        self.ocr = OcrEngine(cfg.ocr, ocr_dir=cfg.models.resolve(cfg.models.ocr_dir))
        self.scene = SceneService(vlm=self.vlm, ocr=self.ocr, provider=self.provider)
        observed_scene = self.bridge.wrap_scene(self.scene)
        observed_chat = self.bridge.wrap_chat(ChatSession(self.vlm))
        observed_tts = self.bridge.wrap_tts(PiperTts(cfg.tts, cfg.models))
        observed_evidence = self.bridge.wrap_evidence(self.evidence)
        sink = AudioSink(
            device=cfg.audio.output_device,
            sample_rate=cfg.audio.sample_rate,
        )
        frontend = SpeechFrontend(
            _make_source(cfg),
            kws=SherpaKws(cfg.kws, cfg.models),
            vad=EnergyVad(
                cfg.vad,
                sample_rate=cfg.audio.sample_rate,
                frame_ms=cfg.audio.frame_ms,
            ),
            asr=StreamingAsr(cfg.asr, cfg.models),
            sample_rate=cfg.audio.sample_rate,
        )
        engine, deps = build_engine(
            cfg,
            evidence=observed_evidence,
            chat=observed_chat,
            scene=observed_scene,
            sink=sink,
            tts=observed_tts,
            frontend=frontend,
            bus=observed_bus,
        )
        self.engine = engine
        self.deps = deps

    # --- 清理 -----------------------------------------------------------
    async def _cleanup(self) -> None:
        cfg = self.cfg
        errors: list[str] = []
        steps: list[tuple[str, Any]] = []

        if self._ready.is_set() and self._error is None:
            self.worker.emit_state({"run": {"id": self.run_id, "status": "stopping"}})
        self.worker.release_preview_ref("run_stop")

        async def _stop_frontend() -> None:
            if self.deps is None:
                return
            try:
                self.deps.frontend.stop()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"frontend.stop: {exc}")

        async def _stop_contract() -> None:
            if not self._contract_started:
                return
            contract = getattr(self.deps, "contract", None)
            if contract is None:
                return
            result = contract.stop()
            if inspect.isawaitable(result):
                await result

        async def _stop_engine() -> None:
            if not self._engine_started:
                return
            await self.engine.stop()

        steps.extend(
            [
                ("frontend", _stop_frontend),
                ("contract", _stop_contract),
                ("engine", _stop_engine),
            ]
        )

        for name, step in steps:
            try:
                await step()
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{name}: {exc}")
                self.worker.log("WARN", "worker", f"stop step {name} failed: {exc}")

        # 关闭链：每项独立 try，任何一项失败不影响后续
        if self.deps is not None:
            _close_quietly(getattr(self.deps, "frontend", None))
            _close_quietly(getattr(self.deps, "sink", None))
            _close_quietly(getattr(self.deps, "tts", None))
        _close_quietly(self.scene)
        _close_quietly(self.ocr)
        _close_quietly(self.vlm)
        if self.provider is not None:
            _close_quietly(self.provider)
        identity = getattr(self.deps, "identity", None) if self.deps is not None else None
        if identity is not None:
            gate = getattr(identity, "gate", identity)
            _close_quietly(gate)
        if self.bridge is not None:
            try:
                self.bridge.finish(cancelled=self.cancel_reason != "")
            except Exception:  # noqa: BLE001
                pass
        _close_quietly(self.evidence)
        self.worker.audio_in = {"status": "unknown", "detail": ""}
        self.worker.audio_out = {"status": "unknown", "detail": ""}
        self.worker.cleanup_after_run()

        status = self._status()
        self.status = status
        self._write_result(status, errors)
        self.worker.on_run_finished(self, status, errors)

    def _status(self) -> str:
        if self._error is not None:
            return "failed"
        if self.cancel_reason in ("client_disconnected", "start_timeout", "shutdown"):
            return "interrupted"
        if self.cancel_reason == "":
            return "completed"
        return "completed"

    def _write_result(self, status: str, errors: list[str]) -> None:
        ended_ms = wall_ms()
        turns = []
        if self.bridge is not None:
            try:
                turns = self.bridge.snapshot_turns()
            except Exception:  # noqa: BLE001
                turns = []
        result = {
            "run_id": self.run_id,
            "mode": "live",
            "status": status,
            "started_at": self.started_iso,
            "started_at_ms": self.started_ms,
            "ended_at": wall_iso(ended_ms),
            "ended_at_ms": ended_ms,
            "duration_ms": max(0, ended_ms - self.started_ms),
            "timings": dict(self.worker.timings),
            "turns": turns,
            "inputs": {},
            "images": {},
            "errors": errors,
            "measurement": {
                "cpu_percent": "所有核归一化 0-100%，整机容量占比",
                "audio": "模块耗时独立测量，不相加为端到端总耗时",
                "generation_ms": "流式首段文本延迟或非流式总生成耗时",
                "acoustic_first_audio_ms": "未测量（无真实声学回路采样）",
            },
            "unmeasured": ["acoustic_first_audio_ms"],
            "human_confirmation": "pending",
            "notes": [
                "本记录由测试台自动生成；人工确认结论尚未填写。",
            ],
        }
        try:
            (self.run_dir / "result.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:  # noqa: BLE001
            self.worker.log("WARN", "worker", f"result.json 写入失败: {exc}")


# ---------------------------------------------------------------------------
# 调试上下文
# ---------------------------------------------------------------------------
class DebugContext:
    """提供给 debug.py 模块的实现上下文（见文件头契约）。"""

    def __init__(
        self,
        worker: "Worker",
        run_id: str,
        run_dir: Path,
        module: str,
        input_data: dict,
        images: dict,
    ) -> None:
        from bridge import Bridge
        from media import FrameStore

        self.worker = worker
        self.run_id = str(run_id)
        self.run_dir = Path(run_dir)
        self.module = str(module)
        self.input = dict(input_data or {})
        self.images = dict(images or {})
        self.cfg = worker.cfg
        self.bench = worker.bench
        self.stop_event = threading.Event()
        self.started_ms = wall_ms()
        self.frame_store = FrameStore(self.run_dir)
        self.bridge = Bridge(
            worker.emit_event,
            run_id=self.run_id,
            frame_store=self.frame_store,
        )

    # --- 输出 -----------------------------------------------------------
    def emit(self, etype: str, payload: dict | None = None, turn_id: str | None = None) -> None:
        self.worker.emit_event(etype, turn_id, payload or {})

    def log(self, level: str, message: str) -> None:
        self.worker.log(level, "debug", message)

    def error(self, code: str, message: str, *, retryable: bool = False) -> None:
        self.emit("error", {"code": code, "message": message, "retryable": retryable})

    def timing(self, **fields: Any) -> None:
        self.emit("timing", {k: v for k, v in fields.items() if v is not None})

    # --- 控制 -----------------------------------------------------------
    @property
    def stopped(self) -> bool:
        return self.stop_event.is_set()

    def wait(self, seconds: float) -> bool:
        """可中断等待；返回 True 表示应停止。"""
        return self.stop_event.wait(max(0.0, seconds))

    def finished(self, status: str, error: str = "") -> None:
        self.worker.on_debug_finished(self, status, error)

    # --- 资源 -----------------------------------------------------------
    def ensure_hub(self) -> Any:
        return self.worker.ensure_hub()

    def camera_provider(self) -> Any:
        from aieyes.scene.providers import CameraFrameProvider
        from media import CameraHubAdapter

        return CameraFrameProvider(camera=CameraHubAdapter(self.ensure_hub(), reason="debug"))

    def load_image(self, slot: str) -> Any:
        entry = self.images.get(slot) or {}
        path = entry.get("path") if isinstance(entry, dict) else entry
        if not path:
            raise BenchError("missing_image", f"槽位 {slot} 没有已上传图片", retryable=False)
        try:
            return load_image_frame(path)
        except Exception as exc:  # noqa: BLE001
            raise BenchError("invalid_image", f"图片 {slot} 无法读取：{exc}") from exc

    def save_frame(self, frame: Any, slot: str, *, purpose: str = "debug") -> dict:
        meta = self.frame_store.save(frame)
        self.bridge.note_frame(frame, purpose=purpose, slot=slot, saved=meta)
        return meta

    # --- 组件 -----------------------------------------------------------
    def vlm_client(self) -> Any:
        from aieyes.vision.vlm import VlmClient

        return VlmClient(self.cfg.llm, self.cfg.vlm)

    def scene_service(self, *, with_ocr: bool = True) -> Any:
        from aieyes.scene.service import SceneService

        ocr = None
        if with_ocr:
            from aieyes.vision.ocr import OcrEngine

            ocr = OcrEngine(self.cfg.ocr, ocr_dir=self.cfg.models.resolve(self.cfg.models.ocr_dir))
        return SceneService(vlm=self.vlm_client(), ocr=ocr)


class DebugRun:
    """一次功能调试任务的生命周期包装。"""

    def __init__(
        self,
        worker: "Worker",
        run_id: str,
        run_dir: Path,
        module: str,
        input_data: dict,
        images: dict,
    ) -> None:
        self.worker = worker
        self.run_id = str(run_id)
        self.run_dir = Path(run_dir)
        self.module = str(module)
        self.input = dict(input_data or {})
        self.images = dict(images or {})
        self.ctx = DebugContext(worker, run_id, run_dir, module, input_data, images)
        self.runner: Any = None

    def start(self) -> dict:
        try:
            import debug as debug_module
        except Exception as exc:  # noqa: BLE001
            raise BenchError("debug_unavailable", f"调试模块不可用：{exc}") from exc
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.runner = debug_module.DebugRunner(self.ctx)
        try:
            self.runner.start(self.module, self.input)
        except BenchError:
            raise
        except Exception as exc:  # noqa: BLE001
            error_cls = getattr(debug_module, "DebugError", None)
            if error_cls is not None and isinstance(exc, error_cls):
                raise BenchError(
                    getattr(exc, "code", "invalid_request"),
                    getattr(exc, "message", str(exc)),
                ) from exc
            raise BenchError("debug_failed", f"调试任务启动失败：{exc}") from exc
        return {"run_id": self.run_id, "module": self.module, "status": "running"}

    def stop(self) -> None:
        self.ctx.stop_event.set()
        if self.runner is not None:
            try:
                self.runner.stop()
            except Exception:  # noqa: BLE001
                pass


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
class Worker:
    def __init__(self) -> None:
        self.bench = dict(BENCH)
        self.lock = threading.RLock()
        self.cfg: Any = None
        self.cfg_error = ""
        self.hub: Any = None
        self.run: RunSession | None = None
        self.debug: DebugRun | None = None
        self.timings: dict[str, Any] = {}
        self.preview_active = False
        self.preview_frames = 0
        self.preview_error = ""
        self.audio_in: dict[str, str] = {"status": "unverified", "detail": ""}
        self.audio_out: dict[str, str] = {"status": "unverified", "detail": ""}
        self.last_preview: tuple[int, str] | None = None
        self.started_ms = wall_ms()
        self.shutting_down = False
        self.worker_pid = os.getpid()
        self.project_root = Path(self.bench.get("project_root") or BENCH_DIR.parent)
        self.run_root = BENCH_DIR / "runs"
        self._load_config()

    # --- 配置 -----------------------------------------------------------
    def _load_config(self) -> None:
        try:
            from aieyes.config import AppConfig

            target = self.project_root / str(self.bench.get("target_config") or "Config/target.json")
            cfg = AppConfig.from_file(str(target))
            # 本轮覆盖：写入路径全部重定向到测试台，不使用正式设备配置
            cfg.audio.input_device = None
            cfg.audio.output_device = None
            self.cfg = cfg
        except Exception as exc:  # noqa: BLE001
            self.cfg_error = f"config_unavailable: {exc}"
            self.cfg = None

    def capabilities(self) -> dict:
        speaker = False
        contract = False
        if self.cfg is not None:
            contract = bool(getattr(self.cfg.contract, "enabled", False))
            try:
                enrolled = self.project_root / "models" / "speaker" if False else None
                from pathlib import Path as _Path

                gate = getattr(self.cfg, "speaker", None)
                if gate is not None and getattr(gate, "enabled", False):
                    enroll_dir = getattr(gate, "enroll_dir", "") or ""
                    base = _Path(self.cfg.models.root)
                    target = base / enroll_dir if enroll_dir else base / "speaker"
                    speaker = target.is_dir() and any(
                        p.suffix in (".npy", ".onnx", ".bin", ".pt", ".json")
                        for p in target.iterdir()
                    )
            except Exception:  # noqa: BLE001
                speaker = False
        return {"contract": contract, "speaker": speaker, "camera": True, "audio": True}

    # --- 状态 -----------------------------------------------------------
    def devices_patch(self) -> dict:
        camera = {"status": "closed", "detail": ""}
        if self.hub is not None:
            status = self.hub.status()
            if status.get("error"):
                camera = {"status": "error", "detail": str(status["error"])}
            elif status.get("camera_open") or status.get("active"):
                camera = {"status": "ready", "detail": "系统默认彩色节点"}
            else:
                camera = {"status": "closed", "detail": ""}
        return {
            "camera": camera,
            "audio_in": dict(self.audio_in),
            "audio_out": dict(self.audio_out),
        }

    def preview_patch(self) -> dict:
        status = self.hub.status() if self.hub is not None else {}
        return {
            "active": bool(self.preview_active),
            "last_frame_ts": status.get("last_frame_ts"),
            "error": self.preview_error or status.get("error"),
            "frames": self.preview_frames,
        }

    def emit_state(self, patch: dict) -> None:
        emit_message({"t": "state", "patch": patch})

    def emit_event(self, etype: str, turn_id: str | None, payload: dict) -> None:
        if etype == "timing":
            for key, value in payload.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    self.timings[key] = value
        emit_message(
            {
                "t": "event",
                "event": {
                    "type": str(etype),
                    "turn_id": turn_id,
                    "payload": payload,
                    "ts_mono_ms": mono_ms(),
                },
            }
        )

    def log(self, level: str, module: str, message: str) -> None:
        emit_message(
            {
                "t": "log",
                "level": str(level).upper(),
                "module": str(module),
                "message": str(message),
            }
        )

    # --- 相机中枢 -------------------------------------------------------
    def ensure_hub(self) -> Any:
        with self.lock:
            if self.hub is not None:
                return self.hub
            if self.cfg is None:
                raise BenchError("config_unavailable", self.cfg_error or "配置不可用")
            from media import CameraHub

            bench = self.bench
            hub = CameraHub(
                lambda: make_camera(self.cfg),
                jpeg_quality=int(bench.get("preview_jpeg_quality", 80)),
                preview_fps=float(bench.get("preview_max_fps", 5)),
                preview_max_side=int(bench.get("preview_max_side", 960)),
                on_preview=self._on_preview,
                log=lambda message: self.log("WARN", "camera", message),
            )
            self.hub = hub
            return hub

    def _on_preview(self, jpeg: bytes, frame_id: str, t_mono_ms: int) -> None:
        if not self.preview_active:
            return
        self.preview_frames += 1
        self.last_preview = (int(t_mono_ms), str(frame_id))
        emit_message(
            {
                "t": "preview",
                "jpeg_b64": base64.b64encode(jpeg).decode("ascii"),
                "frame_id": str(frame_id),
                "ts_mono_ms": int(t_mono_ms),
            }
        )

    def release_preview_ref(self, reason: str) -> None:
        with self.lock:
            if not self.preview_active:
                return
            self.preview_active = False
            hub = self.hub
        if hub is not None:
            try:
                hub.remove_ref("preview")
            except Exception:  # noqa: BLE001
                pass
        self.emit_state({"preview": self.preview_patch(), "devices": self.devices_patch()})

    def cleanup_after_run(self) -> None:
        self.release_preview_ref("run_stop")

    # --- 命令分发 -------------------------------------------------------
    def handle(self, message: dict) -> None:
        request_id = str(message.get("id") or "")
        cmd = str(message.get("cmd") or "")
        args = message.get("args")
        if not isinstance(args, dict):
            args = {}
        if cmd == "shutdown":
            self._handle_shutdown(request_id)
            return
        if not request_id:
            self.log("WARN", "worker", f"忽略无 id 命令: {cmd}")
            return
        thread = threading.Thread(
            target=self._run_command,
            args=(request_id, cmd, args),
            name=f"cmd-{cmd}",
            daemon=True,
        )
        thread.start()

    def _run_command(self, request_id: str, cmd: str, args: dict) -> None:
        handler = getattr(self, f"_cmd_{cmd.replace('.', '_')}", None)
        if handler is None:
            self._send_error(request_id, BenchError("invalid_request", f"未知命令: {cmd}"))
            return
        try:
            result = handler(args)
        except BenchError as exc:
            self._send_error(request_id, exc)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self._send_error(
                request_id, BenchError("internal_error", f"{cmd} 失败：{exc}", retryable=True)
            )
        else:
            emit_message({"t": "ack", "id": request_id, "result": result or {}})

    def _send_error(self, request_id: str, exc: BenchError) -> None:
        self.log("WARN", "worker", f"{exc.code}: {exc.message}")
        emit_message(
            {
                "t": "error",
                "id": request_id,
                "error": {
                    "code": exc.code,
                    "message": exc.message,
                    "retryable": exc.retryable,
                },
            }
        )

    # --- run ------------------------------------------------------------
    def _cmd_run_start(self, args: dict) -> dict:
        run_id = str(args.get("run_id") or "")
        run_dir = Path(str(args.get("run_dir") or ""))
        if not run_id or not str(run_dir):
            raise BenchError("invalid_request", "缺少 run_id/run_dir")
        with self.lock:
            if self.shutting_down:
                raise BenchError("shutting_down", "测试台正在停止", retryable=True)
            if self.run is not None:
                raise BenchError("busy", "已有综合测试运行中", retryable=True)
            if self.debug is not None:
                raise BenchError("busy", "功能调试进行中，不能开始综合测试", retryable=True)
            if self.cfg is None:
                raise BenchError("config_unavailable", self.cfg_error or "配置不可用")
            session = RunSession(self, run_id, run_dir)
            self.run = session
            self.timings = {}
        self.emit_state(
            {"run": {"id": run_id, "mode": "live", "status": "starting", "started_at": session.started_iso}}
        )
        ok, detail = check_model_health(self.bench)
        if not ok:
            with self.lock:
                self.run = None
            self.emit_state({"run": {"id": run_id, "status": "failed"}, "services": {"model": detail}})
            raise BenchError("model_unavailable", detail, retryable=True)
        try:
            session.start(float(self.bench.get("worker_start_timeout_ms", 180000)) / 1000.0)
        except BenchError:
            with self.lock:
                self.run = None
            raise
        except Exception as exc:  # noqa: BLE001
            session.request_stop("start_error")
            with self.lock:
                self.run = None
            raise BenchError("engine_failed", f"启动失败：{exc}") from exc
        return {"run_id": run_id, "status": "running"}

    def _cmd_run_stop(self, args: dict) -> dict:
        with self.lock:
            session = self.run
        if session is None:
            return {"status": "idle"}
        reason = str(args.get("reason") or "user")
        session.request_stop(reason)
        self.emit_state({"run": {"id": session.run_id, "status": "stopping"}})
        return {"run_id": session.run_id, "status": "stopping"}

    def on_run_finished(self, session: RunSession, status: str, errors: list[str]) -> None:
        with self.lock:
            if self.run is session:
                self.run = None
            self.release_preview_ref("run_stop")
        self.emit_state(
            {
                "run": {
                    "id": session.run_id,
                    "status": status,
                    "started_at": session.started_iso,
                    "ended_at": wall_iso(),
                },
                "devices": self.devices_patch(),
            }
        )
        self.log("INFO", "worker", f"run {session.run_id} finished: {status}")
        for error in errors:
            self.log("WARN", "worker", f"cleanup: {error}")
        if status == "failed":
            self.emit_event(
                "error",
                None,
                {"code": "engine_failed", "message": "综合测试未能完成", "retryable": True},
            )
        self._schedule_exit("run_finished")

    # --- preview --------------------------------------------------------
    def _cmd_preview_start(self, args: dict) -> dict:
        with self.lock:
            hub = self.ensure_hub()
            self.preview_error = ""
            if not self.preview_active:
                count = hub.add_ref("preview")
                self.preview_active = True
                self.preview_frames = 0
            else:
                count = hub.ref_count
        self.emit_state({"preview": self.preview_patch(), "devices": self.devices_patch()})
        return {"active": True, "refs": int(count)}

    def _cmd_preview_stop(self, args: dict) -> dict:
        self.release_preview_ref("user")
        return {"active": False}

    def _cmd_status(self, args: dict) -> dict:
        with self.lock:
            run = self.run
            debug = self.debug
            data = {
                "pid": self.worker_pid,
                "uptime_ms": max(0, wall_ms() - self.started_ms),
                "config_error": self.cfg_error,
                "capabilities": self.capabilities(),
                "preview": self.preview_patch(),
                "devices": self.devices_patch(),
                "run": (
                    {"id": run.run_id, "status": "running", "started_at": run.started_iso}
                    if run is not None
                    else None
                ),
                "debug": (
                    {"module": debug.module, "run_id": debug.run_id}
                    if debug is not None
                    else None
                ),
                "timings": dict(self.timings),
            }
        return data

    # --- debug ----------------------------------------------------------
    def _cmd_debug_start(self, args: dict) -> dict:
        module = str(args.get("module") or "")
        run_id = str(args.get("run_id") or "")
        run_dir = Path(str(args.get("run_dir") or ""))
        input_data = args.get("input")
        if not isinstance(input_data, dict):
            input_data = {}
        images = args.get("images")
        if not isinstance(images, dict):
            images = {}
        with self.lock:
            if self.shutting_down:
                raise BenchError("shutting_down", "测试台正在停止", retryable=True)
            if self.run is not None:
                raise BenchError("busy", "综合测试运行中，不能开始功能调试", retryable=True)
            if self.debug is not None:
                raise BenchError("busy", "已有功能调试运行中", retryable=True)
            if self.cfg is None:
                raise BenchError("config_unavailable", self.cfg_error or "配置不可用")
            if not module or not run_id or not str(run_dir):
                raise BenchError("invalid_request", "缺少 module/run_id/run_dir")
            debug_run = DebugRun(self, run_id, run_dir, module, input_data, images)
            self.debug = debug_run
        try:
            result = debug_run.start()
        except BenchError:
            with self.lock:
                if self.debug is debug_run:
                    self.debug = None
            raise
        self.emit_state(
            {
                "debug": {
                    "module": module,
                    "run_id": run_id,
                    "status": "running",
                    "started_at": wall_iso(),
                }
            }
        )
        return result

    def _cmd_debug_stop(self, args: dict) -> dict:
        with self.lock:
            debug_run = self.debug
        if debug_run is None:
            return {"status": "idle"}
        debug_run.stop()
        self.emit_state({"debug": {"run_id": debug_run.run_id, "module": debug_run.module, "status": "stopping"}})
        return {"run_id": debug_run.run_id, "status": "stopping"}

    def _cmd_debug_action(self, args: dict) -> dict:
        with self.lock:
            debug_run = self.debug
        if debug_run is None:
            raise BenchError("invalid_request", "没有运行中的调试任务", retryable=True)
        action = str(args.get("action") or "")
        input_data = args.get("input")
        if not isinstance(input_data, dict):
            input_data = {}
        runner = debug_run.runner
        handler = getattr(runner, "action", None)
        if not callable(handler):
            raise BenchError("invalid_request", "当前调试模块不支持该动作", retryable=True)
        result = handler(action, input_data)
        return result if isinstance(result, dict) else {}

    def on_debug_finished(self, debug_run: DebugRun, status: str, error: str = "") -> None:
        with self.lock:
            if self.debug is debug_run:
                self.debug = None
        patch = {
            "debug": {
                "module": debug_run.module,
                "run_id": debug_run.run_id,
                "status": status,
                "error": error,
                "ended_at": wall_iso(),
            },
            "devices": self.devices_patch(),
        }
        self.emit_state(patch)
        if error:
            self.log("WARN", "debug", f"{debug_run.module}: {status} ({error})")
        else:
            self.log("INFO", "debug", f"{debug_run.module}: {status}")

    # --- 关闭 -----------------------------------------------------------
    def _handle_shutdown(self, request_id: str) -> None:
        self.shutting_down = True
        with self.lock:
            run = self.run
            debug_run = self.debug
        if run is not None:
            run.request_stop("shutdown")
            run.wait_done(10.0)
        if debug_run is not None:
            debug_run.stop()
        if self.hub is not None:
            try:
                self.hub.stop()
            except Exception:  # noqa: BLE001
                pass
        emit_message({"t": "ack", "id": request_id, "result": {"status": "stopping"}})
        self._schedule_exit("shutdown", delay=0.05)

    def _schedule_exit(self, reason: str, delay: float = 0.2) -> None:
        def _exit() -> None:
            if delay > 0:
                time.sleep(delay)
            self.log("INFO", "worker", f"exit ({reason})")
            emit_message({"t": "exit"})
            try:
                sys.stdout.flush()
            except Exception:  # noqa: BLE001
                pass
            os._exit(0)

        threading.Thread(target=_exit, name="worker-exit", daemon=True).start()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    bench = BENCH
    if not bench:
        emit_message(
            {
                "t": "hello",
                "pid": os.getpid(),
                "capabilities": {},
                "config_error": "bench.json 加载失败",
            }
        )
        return 2
    worker = Worker()
    emit_message(
        {
            "t": "hello",
            "pid": os.getpid(),
            "capabilities": worker.capabilities(),
            "config_error": worker.cfg_error,
            "bench_dir": str(BENCH_DIR),
        }
    )
    worker.log("INFO", "worker", f"worker 启动 pid={os.getpid()}")
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                worker.log("WARN", "worker", f"忽略非法 JSON: {line[:200]}")
                continue
            if not isinstance(message, dict):
                continue
            worker.handle(message)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    finally:
        worker.shutting_down = True
        with worker.lock:
            run = worker.run
            debug_run = worker.debug
        if run is not None:
            run.request_stop("stdin_closed")
            run.wait_done(10.0)
        if debug_run is not None:
            debug_run.stop()
        if worker.hub is not None:
            try:
                worker.hub.stop()
            except Exception:  # noqa: BLE001
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
