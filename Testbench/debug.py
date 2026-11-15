"""功能调试模块：按固定白名单，用真实业务组件完成单项调试。

契约（由 ``Testbench/worker.py`` 文件头约定，本文件不反向 import worker/server）：

* ``MODULE_SPECS`` —— ``list[dict]``；每项包含 ``id`` / ``name`` / ``desc`` /
  ``fields`` / ``actions``，由 ``GET /api/debug/modules`` 原样返回给前端。
* ``class DebugError`` —— 参数/能力错误，携带 ``code`` 与 ``message``。
* ``class DebugRunner(ctx)`` —— ``ctx`` 为 ``worker.DebugContext``：``start()``
  校验参数后启动后台线程，``stop()`` 置停止标记并等待，``alive`` 表示线程是否
  仍在运行，``action()`` 处理调试动作；结束时调用 ``ctx.finished(status, error)``。

输出只走 ``ctx.emit/log/error/timing/save_frame/load_image/camera_provider``，
事件 type 仅限调试面板支持的八种：``asr.partial`` / ``asr.final`` /
``assistant.final`` / ``playback`` / ``frame`` / ``timing`` / ``log`` / ``error``。

能力缺失（模型/档案/合同链路未接入等）只展示“未接入/未启用”及配置依据，
不伪造结果；真实音频/相机/模型组件全部在模块被选中后才惰性加载，并在结束时关闭。
"""

from __future__ import annotations

import copy
import math
import queue
import threading
import time
import traceback
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 固定模块表（顺序即前端展示顺序）
# ---------------------------------------------------------------------------
_DEBUG_EVENT_TYPES = frozenset(
    {
        "asr.partial",
        "asr.final",
        "assistant.final",
        "playback",
        "frame",
        "timing",
        "log",
        "error",
    }
)

AUDIO_RECORD_SECONDS = 5.0
KWS_VAD_MAX_SECONDS = 20.0
ASR_MAX_SECONDS = 20.0
SPEAKER_SAMPLE_SECONDS = 3.0
STOP_JOIN_TIMEOUT_S = 8.0

_SOURCE_CAMERA = "相机"
_SOURCE_IMAGE_A = "上传图片 A"
_COMPARE_SOURCE_UPLOAD = "上传图片 A/B"
_COMPARE_SOURCE_CAMERA = "相机先后拍摄"


def _field(
    name: str,
    label: str,
    ftype: str = "text",
    *,
    required: bool = False,
    options: list[str] | None = None,
    value: Any = None,
) -> dict:
    item: dict[str, Any] = {"name": name, "label": label, "type": ftype}
    if required:
        item["required"] = True
    if options is not None:
        item["options"] = list(options)
    if value is not None:
        item["value"] = value
    return item


def _action(name: str, label: str) -> dict:
    return {"name": name, "label": label}


MODULE_SPECS: list[dict] = [
    {
        "id": "audio",
        "name": "音频输入/输出",
        "desc": "用系统默认麦克风录制 5 秒并保存 WAV；只有点击「播放刚录音」才会出声。",
        "fields": [],
        "actions": [_action("play", "播放刚录音")],
    },
    {
        "id": "kws_vad",
        "name": "唤醒与 VAD",
        "desc": "用当前 KWS/VAD 监听最长 20 秒，显示唤醒词与语音起止；不自动提交问答。",
        "fields": [],
        "actions": [],
    },
    {
        "id": "asr",
        "name": "语音识别",
        "desc": "用当前 StreamingAsr 识别最长 20 秒，显示中间文本与最终全文，可提前停止。",
        "fields": [],
        "actions": [],
    },
    {
        "id": "chat",
        "name": "文本对话",
        "desc": "只调用 ChatSession/VlmClient 生成文本，不自动 TTS；上下文仅本次调试保留，停止即释放。",
        "fields": [_field("text", "问题", "textarea", required=True)],
        "actions": [],
    },
    {
        "id": "tts",
        "name": "语音合成与播放",
        "desc": "用 PiperTts 合成并用默认 AudioSink 实际播放，可停止；显示合成与播放状态。",
        "fields": [_field("text", "播报文本", "textarea", required=True)],
        "actions": [],
    },
    {
        "id": "interruption",
        "name": "插话中断",
        "desc": "启动真实交互链路（KWS/VAD/ASR/对话/播报/相机），记录真实插话事件，不注入取消事件。",
        "fields": [],
        "actions": [],
    },
    {
        "id": "vision",
        "name": "视觉问答",
        "desc": "用 SceneService 对相机或已上传图片 A 做真实视觉问答，显示实际输入帧与回答。",
        "fields": [
            _field("question", "问题", "text", required=True),
            _field(
                "source",
                "图片来源",
                "select",
                options=[_SOURCE_CAMERA, _SOURCE_IMAGE_A],
                value=_SOURCE_CAMERA,
            ),
        ],
        "actions": [],
    },
    {
        "id": "ocr",
        "name": "文字识别",
        "desc": "用现有 OCR/解析读取文字、价格与日期，分别显示识别原文和业务结果，不另写计算规则。",
        "fields": [
            _field("question", "读取要求", "text", value="读取图中的文字，包括价格和日期。"),
            _field(
                "source",
                "图片来源",
                "select",
                options=[_SOURCE_CAMERA, _SOURCE_IMAGE_A],
                value=_SOURCE_CAMERA,
            ),
        ],
        "actions": [],
    },
    {
        "id": "compare",
        "name": "图片比较",
        "desc": "用现有比较能力比较上传的 A/B 或相机先后两次拍摄；相机两次拍摄必须分别由用户触发。",
        "fields": [
            _field("question", "比较问题", "text", value="比较一下这两样东西"),
            _field(
                "source",
                "图片来源",
                "select",
                options=[_COMPARE_SOURCE_UPLOAD, _COMPARE_SOURCE_CAMERA],
                value=_COMPARE_SOURCE_UPLOAD,
            ),
        ],
        "actions": [
            _action("capture_a", "拍摄图片 A"),
            _action("capture_b", "拍摄图片 B"),
        ],
    },
    {
        "id": "speaker",
        "name": "声纹验证",
        "desc": "在用户明确同意后采集本轮参考/核验音频，使用当前声纹组件临时验证分数与准入状态；免唤醒准入未启用。",
        "fields": [
            _field(
                "consent",
                "我已知晓本次采集仅用于测试，不会写入正式档案",
                "checkbox",
                required=True,
                value=False,
            )
        ],
        "actions": [],
    },
    {
        "id": "contract",
        "name": "合同联调",
        "desc": "查询合同配置并显式触发测试事件；正式配置关闭时显示未接入；调试不连接真实主控。",
        "fields": [],
        "actions": [
            _action("status", "查询合同状态"),
            _action("test_event", "发送测试事件"),
        ],
    },
]

_MODULE_BY_ID = {str(item["id"]): item for item in MODULE_SPECS}


class DebugError(Exception):
    """调试参数/能力错误；``code`` 供 worker 转为 BenchError。"""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------
def _mono_ms() -> int:
    return int(time.monotonic() * 1000)


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return False
    return not str(value).strip()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on", "是")


def _as_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _dbfs(samples: Any) -> float:
    import numpy as np

    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return -120.0
    rms = float(np.sqrt(np.mean(np.square(data))))
    return 20.0 * math.log10(max(rms, 1e-9))


def _peak_dbfs(samples: Any) -> float:
    import numpy as np

    data = np.asarray(samples, dtype=np.float32).reshape(-1)
    if data.size == 0:
        return -120.0
    peak = float(np.max(np.abs(data)))
    return 20.0 * math.log10(max(peak, 1e-9))


def _safe_close(label: str, obj: Any, ctx: Any) -> None:
    if obj is None:
        return
    close = getattr(obj, "close", None)
    if not callable(close):
        return
    try:
        close()
    except Exception as exc:  # noqa: BLE001 - 释放失败不能掩盖调试结论
        try:
            ctx.log("WARN", f"{label} 释放失败：{exc}")
        except Exception:  # noqa: BLE001
            pass


class _EmitAdapter:
    """把 ``Bridge`` 的 ``emit(type, turn_id, payload)`` 收敛到调试事件白名单。"""

    def __init__(self, ctx: Any) -> None:
        self._ctx = ctx

    def __call__(self, etype: str, turn_id: Any = None, payload: Any = None) -> None:
        if etype not in _DEBUG_EVENT_TYPES:
            return
        try:
            self._ctx.emit(etype, payload if isinstance(payload, dict) else {}, turn_id)
        except Exception:  # noqa: BLE001 - 观测失败不影响真实链路
            pass


# ---------------------------------------------------------------------------
# 调试执行器
# ---------------------------------------------------------------------------
class DebugRunner:
    """一次功能调试：负责参数校验、后台执行、动作分发与资源释放。"""

    def __init__(self, ctx: Any) -> None:
        self.ctx = ctx
        self._module = ""
        self._input: dict[str, Any] = {}
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._finished = False
        self._resources: list[tuple[str, Any]] = []
        self._cancel: Any = None
        self._audio_state: dict[str, Any] = {}
        self._compare_state: dict[str, Any] = {}
        self._speaker_state: dict[str, Any] = {}
        self._contract_state: dict[str, Any] = {}

    # -- 生命周期 -------------------------------------------------------
    @property
    def alive(self) -> bool:
        thread = self._thread
        return thread is not None and thread.is_alive()

    def start(self, module: str, input_data: dict | None = None) -> None:
        module = _as_text(module)
        with self._lock:
            if self.alive:
                raise DebugError("busy", "已有调试任务运行中")
            spec = _MODULE_BY_ID.get(module)
            if spec is None:
                raise DebugError("invalid_module", f"未知调试模块：{module or '(空)'}")
            data = dict(input_data or {})
            self._validate_fields(spec, data)
            self._validate_module(module, data)
            self._module = module
            self._input = data
            self._finished = False
            thread = threading.Thread(
                target=self._thread_main,
                name=f"debug-{module}",
                daemon=True,
            )
            self._thread = thread
        thread.start()

    def stop(self) -> None:
        try:
            self.ctx.stop_event.set()
        except Exception:  # noqa: BLE001
            pass
        cancel = self._cancel
        if callable(cancel):
            try:
                cancel()
            except Exception as exc:  # noqa: BLE001
                self._log("WARN", f"取消播放失败：{exc}")
            self._cancel = None
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(STOP_JOIN_TIMEOUT_S)
            if thread.is_alive():
                self._log(
                    "WARN",
                    f"调试线程未在 {STOP_JOIN_TIMEOUT_S:.0f} 秒内结束，已标记中断",
                )
                self._finish("interrupted", "stop_timeout")

    def action(self, action: str, input_data: dict | None = None) -> dict:
        action = _as_text(action)
        data = dict(input_data or {})
        if not self.alive:
            raise DebugError("invalid_request", "当前没有运行中的调试任务")
        if self._module == "audio" and action == "play":
            return self._action_audio_play()
        if self._module == "compare" and action in ("capture_a", "capture_b"):
            return self._action_compare_capture(action)
        if self._module == "contract" and action in ("status", "test_event"):
            return self._action_contract(action)
        raise DebugError(
            "invalid_request",
            f"当前模块 {self._module} 不支持动作：{action or '(空)'}",
        )

    # -- 参数校验 -------------------------------------------------------
    def _validate_fields(self, spec: dict, data: dict) -> None:
        for field in spec.get("fields") or []:
            name = str(field.get("name") or "")
            if not name:
                continue
            label = str(field.get("label") or name)
            ftype = str(field.get("type") or "text")
            value = data.get(name)
            if ftype == "checkbox":
                # 复选框只由模块级校验给出更明确的拒绝原因
                continue
            if field.get("required") and _is_blank(value):
                raise DebugError("invalid_request", f"缺少必填参数：{label}")
            options = field.get("options")
            if ftype == "select" and isinstance(options, list) and options:
                if not _is_blank(value) and str(value) not in [str(o) for o in options]:
                    raise DebugError("invalid_request", f"{label} 取值无效：{value}")

    def _validate_module(self, module: str, data: dict) -> None:
        images = getattr(self.ctx, "images", None) or {}
        if module == "speaker":
            if not _as_bool(data.get("consent")):
                raise DebugError(
                    "consent_required",
                    "未获得用户同意：未采集任何音频。请先勾选同意项。",
                )
        elif module == "compare":
            source = _as_text(data.get("source")) or _COMPARE_SOURCE_UPLOAD
            if source == _COMPARE_SOURCE_UPLOAD:
                missing = [slot for slot in ("A", "B") if not images.get(slot)]
                if missing:
                    raise DebugError(
                        "missing_image",
                        "缺少已上传图片：%s；请先上传 A/B，或选择相机先后拍摄。"
                        % "、".join(missing),
                    )
        elif module in ("vision", "ocr"):
            source = _as_text(data.get("source")) or _SOURCE_CAMERA
            if source == _SOURCE_IMAGE_A and not images.get("A"):
                raise DebugError(
                    "missing_image",
                    "缺少已上传图片 A：请先上传图片 A，或把图片来源切换为相机。",
                )

    def _thread_main(self) -> None:
        status = "completed"
        error = ""
        try:
            handler = getattr(self, f"_run_{self._module}", None)
            if not callable(handler):
                raise DebugError("invalid_module", f"调试模块未实现：{self._module}")
            handler()
        except DebugError as exc:
            status = "failed"
            error = exc.message
            try:
                self.ctx.error(exc.code, exc.message)
            except Exception:  # noqa: BLE001
                pass
            self._log("ERROR", f"调试失败：{exc.message}")
        except Exception as exc:  # noqa: BLE001 - 设备/模型异常统一上报
            status = "failed"
            code = str(getattr(exc, "code", "") or "debug_failed")
            message = str(getattr(exc, "message", "") or exc or exc.__class__.__name__)
            error = message
            try:
                self.ctx.error(code, message)
            except Exception:  # noqa: BLE001
                pass
            self._log("ERROR", f"调试失败：{message}")
            self._log("WARN", "".join(traceback.format_exc(limit=2)).strip())
        finally:
            self._close_resources()
            if status == "completed" and self.ctx.stopped:
                status = "interrupted"
            self._finish(status, error)

    def _finish(self, status: str, error: str = "") -> None:
        with self._lock:
            if self._finished:
                return
            self._finished = True
        try:
            self.ctx.finished(status, error)
        except Exception:  # noqa: BLE001
            pass

    # -- 资源 -----------------------------------------------------------
    def _track(self, label: str, obj: Any, method: str = "close") -> Any:
        if obj is None:
            return None
        fn = getattr(obj, method, None)
        if callable(fn):
            with self._lock:
                self._resources.append((label, fn))
        return obj

    def _close_resources(self) -> None:
        with self._lock:
            resources = list(reversed(self._resources))
            self._resources.clear()
        for label, fn in resources:
            try:
                fn()
            except Exception as exc:  # noqa: BLE001
                self._log("WARN", f"{label} 释放失败：{exc}")
        self._cancel = None

    # -- 输出 -----------------------------------------------------------
    def _log(self, level: str, message: str) -> None:
        try:
            self.ctx.log(level, message)
        except Exception:  # noqa: BLE001
            pass

    def _emit(self, etype: str, payload: dict | None = None) -> None:
        if etype not in _DEBUG_EVENT_TYPES:
            return
        try:
            self.ctx.emit(etype, payload or {})
        except Exception:  # noqa: BLE001
            pass

    def _timing(self, **fields: Any) -> None:
        try:
            self.ctx.timing(**fields)
        except Exception:  # noqa: BLE001
            pass

    # -- 惰性组件工厂（测试可覆盖） -------------------------------------
    @property
    def cfg(self) -> Any:
        return self.ctx.cfg

    def _audio_cfg(self) -> Any:
        return getattr(self.cfg, "audio", None)

    def _make_source(self) -> Any:
        from aieyes.audio.capture import SoundDeviceSource

        audio = self._audio_cfg()
        return SoundDeviceSource(
            audio,
            sample_rate=int(getattr(audio, "sample_rate", 16000) or 16000),
            frame_ms=int(getattr(audio, "frame_ms", 20) or 20),
            device=None,  # 固定使用系统默认设备
        )

    def _make_sink(self, sample_rate: int) -> Any:
        from aieyes.audio.output import AudioSink

        return AudioSink(device=None, sample_rate=sample_rate)

    def _make_kws(self) -> Any:
        from aieyes.speech.kws import SherpaKws

        return SherpaKws(getattr(self.cfg, "kws", None), getattr(self.cfg, "models", None))

    def _make_vad(self) -> Any:
        from aieyes.audio.vad import EnergyVad

        audio = self._audio_cfg()
        return EnergyVad(
            getattr(self.cfg, "vad", None),
            sample_rate=int(getattr(audio, "sample_rate", 16000) or 16000),
            frame_ms=int(getattr(audio, "frame_ms", 20) or 20),
        )

    def _make_asr(self) -> Any:
        from aieyes.speech.asr import StreamingAsr

        return StreamingAsr(getattr(self.cfg, "asr", None), getattr(self.cfg, "models", None))

    def _make_tts(self) -> Any:
        from aieyes.speech.tts import PiperTts

        return PiperTts(getattr(self.cfg, "tts", None), getattr(self.cfg, "models", None))

    def _make_vlm(self) -> Any:
        return self.ctx.vlm_client()

    def _make_chat(self, vlm: Any) -> Any:
        from aieyes.chat import ChatSession

        return ChatSession(vlm)

    def _make_scene(self) -> Any:
        from aieyes.scene.service import SceneService
        from aieyes.vision.ocr import OcrEngine

        vlm = self._track("vlm", self._make_vlm())
        ocr = None
        models = getattr(self.cfg, "models", None)
        resolve = getattr(models, "resolve", None)
        ocr_dir = None
        if callable(resolve):
            try:
                ocr_dir = resolve(getattr(models, "ocr_dir", "ocr"))
            except Exception:  # noqa: BLE001
                ocr_dir = None
        try:
            ocr = OcrEngine(getattr(self.cfg, "ocr", None), ocr_dir=ocr_dir)
            self._track("ocr", ocr)
        except Exception as exc:  # noqa: BLE001 - OCR 不可用时交给 SceneService 惰性默认
            self._log("WARN", f"OCR 组件初始化失败：{exc}")
        scene = SceneService(vlm=vlm, ocr=ocr)
        self._track("scene", scene)
        return scene

    def _make_gate(self) -> Any:
        from aieyes.identity import SpeakerGate

        models = getattr(self.cfg, "models", None)
        return SpeakerGate(
            cfg=getattr(self.cfg, "speaker", None),
            models_root=getattr(models, "root", None),
        )

    def _read_wav(self, path: str | Path) -> tuple[Any, int]:
        from aieyes.audio.wav import read_wav

        return read_wav(str(path))

    def _write_wav(self, path: str | Path, samples: Any, sample_rate: int) -> None:
        from aieyes.audio.wav import write_wav

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        write_wav(str(path), samples, sample_rate)

    # -- 录音辅助 -------------------------------------------------------
    def _record_audio(self, seconds: float, label: str) -> tuple[Any, int]:
        import numpy as np

        audio = self._audio_cfg()
        sample_rate = int(getattr(audio, "sample_rate", 16000) or 16000)
        source = self._make_source()
        frames: queue.Queue = queue.Queue(maxsize=200)

        def _on_frame(frame: Any) -> None:
            try:
                frames.put_nowait(frame)
            except queue.Full:
                pass

        started = time.monotonic()
        chunks: list[Any] = []
        last_report = 0.0
        source.start(_on_frame)
        try:
            while not self.ctx.stopped:
                remaining = float(seconds) - (time.monotonic() - started)
                if remaining <= 0:
                    break
                try:
                    frame = frames.get(timeout=min(0.1, max(0.01, remaining)))
                except queue.Empty:
                    continue
                samples = np.asarray(frame.samples, dtype=np.float32).reshape(-1)
                if samples.size:
                    chunks.append(samples.copy())
                now = time.monotonic()
                if now - last_report >= 0.5:
                    last_report = now
                    self._log("INFO", f"{label}电平：{_dbfs(samples):.1f} dBFS")
        finally:
            try:
                source.stop()
            except Exception as exc:  # noqa: BLE001
                self._log("WARN", f"{label}采集停止失败：{exc}")
        if chunks:
            data = np.concatenate(chunks)
        else:
            data = np.zeros(0, dtype=np.float32)
        return data, sample_rate

    # ------------------------------------------------------------------
    # audio：录音 5 秒；仅用户点击动作时播放
    # ------------------------------------------------------------------
    def _run_audio(self) -> None:
        self._log("INFO", f"开始录音 {AUDIO_RECORD_SECONDS:.0f} 秒（系统默认输入设备）…")
        data, sample_rate = self._record_audio(AUDIO_RECORD_SECONDS, "录音")
        if self.ctx.stopped:
            self._log("INFO", "录音被提前停止。")
            return
        if getattr(data, "size", 0) == 0:
            raise DebugError("audio_empty", "没有采集到音频：请检查系统默认输入设备。")
        duration = float(getattr(data, "size", 0)) / float(sample_rate or 16000)
        wav_path = Path(self.ctx.run_dir) / "audio" / "recording.wav"
        self._write_wav(wav_path, data, sample_rate)
        self._audio_state.update(
            {"path": str(wav_path), "samples": data, "sample_rate": sample_rate}
        )
        self._timing(record_ms=int(duration * 1000))
        self._log(
            "INFO",
            "录音完成：%.2f 秒，峰值 %.1f dBFS，WAV 已保存到 %s"
            % (duration, _peak_dbfs(data), wav_path),
        )
        self._log("INFO", "点击「播放刚录音」才会出声；点停止结束本次调试。")
        while not self.ctx.stopped:
            if self.ctx.wait(0.2):
                break
        self._log("INFO", "音频调试结束。")

    def _action_audio_play(self) -> dict:
        state = self._audio_state
        path = state.get("path")
        if not path:
            raise DebugError("invalid_request", "尚未完成录音，无法播放。")
        try:
            samples, sample_rate = self._read_wav(path)
        except Exception as exc:  # noqa: BLE001
            raise DebugError("audio_failed", f"读取录音失败：{exc}") from exc
        sink = self._track("sink", self._make_sink(sample_rate))
        sink.start()
        piece_id = f"audio-{_mono_ms()}"
        self._cancel = lambda: sink.cancel(piece_id)
        self._emit(
            "playback",
            {
                "state": "started",
                "piece_id": piece_id,
                "text": "刚录音",
                "kind": "answer",
            },
        )
        started = time.monotonic()
        try:
            handle = sink.play(samples, sample_rate, piece_id=piece_id, text="刚录音")
        except Exception as exc:  # noqa: BLE001
            self._cancel = None
            raise DebugError("audio_failed", f"播放失败：{exc}") from exc
        while not self.ctx.stopped:
            done = True
            wait = getattr(handle, "wait", None)
            if callable(wait):
                done = bool(wait(0.1))
            if done:
                break
        cancelled = bool(self.ctx.stopped)
        if cancelled:
            try:
                sink.cancel(piece_id)
            except Exception:  # noqa: BLE001
                pass
        self._cancel = None
        played_ms = int((time.monotonic() - started) * 1000)
        self._emit(
            "playback",
            {
                "state": "finished",
                "piece_id": piece_id,
                "played_ms": played_ms,
                "total_ms": int(getattr(handle, "total_ms", 0) or played_ms),
                "cancelled": cancelled,
            },
        )
        self._log("INFO", "播放%s：%d ms" % ("已取消" if cancelled else "完成", played_ms))
        return {
            "played": not cancelled,
            "cancelled": cancelled,
            "played_ms": played_ms,
        }

    # ------------------------------------------------------------------
    # kws_vad：最长 20 秒，只观测唤醒与语音起止
    # ------------------------------------------------------------------
    def _run_kws_vad(self) -> None:
        self._log(
            "INFO",
            f"启动 KWS/VAD（最长 {KWS_VAD_MAX_SECONDS:.0f} 秒，可点停止提前结束）…",
        )
        kws = self._track("kws", self._make_kws())
        vad = self._track("vad", self._make_vad())
        source = self._make_source()
        frames: queue.Queue = queue.Queue(maxsize=200)

        def _on_frame(frame: Any) -> None:
            try:
                frames.put_nowait(frame)
            except queue.Full:
                pass

        started = time.monotonic()
        wake_count = 0
        speech_count = 0
        in_speech = False
        last_level = 0.0
        source.start(_on_frame)
        try:
            while not self.ctx.stopped:
                remaining = KWS_VAD_MAX_SECONDS - (time.monotonic() - started)
                if remaining <= 0:
                    break
                try:
                    frame = frames.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                word = kws.accept(frame)
                if word:
                    wake_count += 1
                    self._log("INFO", f"唤醒：{word}")
                segment = vad.accept(frame)
                now_in_speech = bool(vad.in_speech)
                if now_in_speech != in_speech:
                    in_speech = now_in_speech
                    self._log("INFO", "语音开始" if in_speech else "语音结束")
                if segment is not None:
                    speech_count += 1
                    duration = int(segment.t_end_ms - segment.t_start_ms)
                    self._log(
                        "INFO",
                        f"语音段：{duration} ms（{getattr(segment, 'cause', 'vad_end')}）",
                    )
                now = time.monotonic()
                if now - last_level >= 1.0:
                    last_level = now
                    self._log("INFO", f"电平：{_dbfs(frame.samples):.1f} dBFS")
        finally:
            try:
                source.stop()
            except Exception as exc:  # noqa: BLE001
                self._log("WARN", f"音频采集停止失败：{exc}")
        elapsed = time.monotonic() - started
        reason = "已停止" if self.ctx.stopped else f"达到 {KWS_VAD_MAX_SECONDS:.0f} 秒上限"
        self._timing(kws_vad_ms=int(elapsed * 1000))
        self._log(
            "INFO",
            f"KWS/VAD 结束（{reason}）：唤醒 {wake_count} 次，语音段 {speech_count} 段；"
            "本模块不自动提交问答。",
        )

    # ------------------------------------------------------------------
    # asr：最长 20 秒，中间文本 + 最终全文
    # ------------------------------------------------------------------
    def _run_asr(self) -> None:
        self._log(
            "INFO",
            f"启动 StreamingAsr（最长 {ASR_MAX_SECONDS:.0f} 秒，可点停止提前结束）…",
        )
        asr = self._track("asr", self._make_asr())
        source = self._make_source()
        frames: queue.Queue = queue.Queue(maxsize=200)

        def _on_frame(frame: Any) -> None:
            try:
                frames.put_nowait(frame)
            except queue.Full:
                pass

        started = time.monotonic()
        last_text = ""
        last_partial_log = 0.0
        source.start(_on_frame)
        asr.start_utterance()
        try:
            while not self.ctx.stopped:
                remaining = ASR_MAX_SECONDS - (time.monotonic() - started)
                if remaining <= 0:
                    break
                try:
                    frame = frames.get(timeout=min(0.1, remaining))
                except queue.Empty:
                    continue
                text = asr.accept(frame)
                if text and text != last_text:
                    last_text = str(text)
                    self._emit("asr.partial", {"text": last_text})
                    now = time.monotonic()
                    if now - last_partial_log >= 0.25:
                        last_partial_log = now
                        self._log("INFO", f"中间识别：{last_text}")
        finally:
            try:
                source.stop()
            except Exception as exc:  # noqa: BLE001
                self._log("WARN", f"音频采集停止失败：{exc}")
        final_text = last_text
        try:
            utterance = asr.finalize()
            if utterance is not None:
                final_text = _as_text(getattr(utterance, "text", "")) or last_text
        except Exception as exc:  # noqa: BLE001
            self._log("WARN", f"ASR finalize 失败：{exc}")
        elapsed = time.monotonic() - started
        self._emit("asr.final", {"text": final_text})
        self._timing(asr_ms=int(elapsed * 1000))
        if final_text:
            self._log("INFO", f"最终识别：{final_text}")
        else:
            self._log("WARN", "未识别到语音文本。")

    # ------------------------------------------------------------------
    # chat：只生成文本，不自动 TTS
    # ------------------------------------------------------------------
    def _run_chat(self) -> None:
        text = _as_text(self._input.get("text"))
        self._log("INFO", "调用 ChatSession 生成文本回答（不自动 TTS）…")
        vlm = self._track("vlm", self._make_vlm())
        chat = self._make_chat(vlm)
        started = time.monotonic()
        answer = chat.ask(text)
        generation_ms = int((time.monotonic() - started) * 1000)
        answer_text = _as_text(getattr(answer, "text", ""))
        self._emit(
            "assistant.final",
            {"text": answer_text, "cancelled": False, "source": "debug"},
        )
        self._timing(generation_ms=generation_ms)
        self._log(
            "INFO",
            f"生成完成：{generation_ms} ms，{len(answer_text)} 字；"
            "调试上下文仅本次运行保留，停止即释放。",
        )

    # ------------------------------------------------------------------
    # tts：合成 + 实际播放，可停止
    # ------------------------------------------------------------------
    def _run_tts(self) -> None:
        text = _as_text(self._input.get("text"))
        self._log("INFO", "PiperTts 合成中…")
        tts = self._track("tts", self._make_tts())
        started = time.monotonic()
        audio = tts.synth(text)
        synth_ms = int((time.monotonic() - started) * 1000)
        self._timing(tts_ms=synth_ms)
        self._log(
            "INFO",
            f"合成完成：{synth_ms} ms，音频 {int(getattr(audio, 'duration_ms', 0) or 0)} ms",
        )
        if self.ctx.stopped:
            self._log("INFO", "已停止：跳过播放。")
            return
        sample_rate = int(getattr(audio, "sample_rate", 16000) or 16000)
        pcm = getattr(audio, "pcm", None)
        if pcm is None:
            raise DebugError("tts_failed", "TTS 未返回音频数据。")
        sink = self._track("sink", self._make_sink(sample_rate))
        sink.start()
        piece_id = f"tts-{_mono_ms()}"
        self._cancel = lambda: sink.cancel(piece_id)
        self._emit(
            "playback",
            {
                "state": "started",
                "piece_id": piece_id,
                "text": text,
                "kind": "answer",
            },
        )
        play_started = time.monotonic()
        try:
            handle = sink.play(pcm, sample_rate, piece_id=piece_id, text=text)
        except Exception as exc:  # noqa: BLE001
            self._cancel = None
            raise DebugError("tts_failed", f"播放失败：{exc}") from exc
        while not self.ctx.stopped:
            done = True
            wait = getattr(handle, "wait", None)
            if callable(wait):
                done = bool(wait(0.1))
            if done:
                break
        cancelled = bool(self.ctx.stopped)
        if cancelled:
            try:
                sink.cancel(piece_id)
            except Exception:  # noqa: BLE001
                pass
        self._cancel = None
        played_ms = int((time.monotonic() - play_started) * 1000)
        self._emit(
            "playback",
            {
                "state": "finished",
                "piece_id": piece_id,
                "played_ms": played_ms,
                "total_ms": int(getattr(audio, "duration_ms", 0) or played_ms),
                "cancelled": cancelled,
            },
        )
        self._log("INFO", "播放%s：%d ms" % ("已取消" if cancelled else "完成", played_ms))

    # ------------------------------------------------------------------
    # 取帧辅助（vision / ocr / compare 共用）
    # ------------------------------------------------------------------
    def _acquire_frame(self, source: str, purpose: str) -> tuple[Any, dict]:
        if source == _SOURCE_IMAGE_A:
            frame = self.ctx.load_image("A")
            slot = "A"
        else:
            provider = self.ctx.camera_provider()
            try:
                frame = provider.acquire(purpose)
            finally:
                try:
                    provider.close()
                except Exception:  # noqa: BLE001
                    pass
            slot = "model"
        meta = self.ctx.save_frame(frame, slot, purpose=purpose)
        self._log(
            "INFO",
            "输入帧：ID %s，%s×%s，来源 %s"
            % (
                _as_text(getattr(frame, "frame_id", "")) or "无 ID",
                meta.get("width", getattr(frame, "width", "?")),
                meta.get("height", getattr(frame, "height", "?")),
                source,
            ),
        )
        return frame, meta

    # ------------------------------------------------------------------
    # vision：SceneService 真实视觉问答
    # ------------------------------------------------------------------
    def _run_vision(self) -> None:
        question = _as_text(self._input.get("question"))
        source = _as_text(self._input.get("source")) or _SOURCE_CAMERA
        self._log("INFO", f"SceneService 视觉问答（来源：{source}）…")
        frame, _meta = self._acquire_frame(source, "vision")
        scene = self._make_scene()
        started = time.monotonic()
        answer = scene.answer(question, frame=frame)
        elapsed = int((time.monotonic() - started) * 1000)
        text = _as_text(getattr(answer, "text", ""))
        self._emit(
            "assistant.final",
            {"text": text, "cancelled": False, "source": "debug"},
        )
        self._timing(vision_ms=elapsed)
        self._log(
            "INFO",
            "视觉回答完成：%d ms；意图 %s；降级 %s"
            % (
                elapsed,
                _as_text(getattr(answer, "intent", "")) or "未知",
                bool(getattr(answer, "degraded", False)),
            ),
        )

    # ------------------------------------------------------------------
    # ocr：识别原文 + 现有解析/业务结果
    # ------------------------------------------------------------------
    def _run_ocr(self) -> None:
        question = _as_text(self._input.get("question")) or "读取图中的文字，包括价格和日期。"
        source = _as_text(self._input.get("source")) or _SOURCE_CAMERA
        self._log("INFO", f"OCR/解析（来源：{source}）…")
        frame, _meta = self._acquire_frame(source, "ocr")
        scene = self._make_scene()
        ocr_started = time.monotonic()
        result = scene.read_text_result(frame)
        ocr_ms = int(getattr(result, "t_ms", 0) or ((time.monotonic() - ocr_started) * 1000))
        lines = list(getattr(result, "lines", ()) or ())
        if lines:
            self._log("INFO", f"OCR 原文（{len(lines)} 行）：")
            for line in lines[:30]:
                self._log(
                    "INFO",
                    "  %s  [%.2f]"
                    % (
                        _as_text(getattr(line, "text", "")),
                        float(getattr(line, "score", 0.0) or 0.0),
                    ),
                )
        else:
            self._log("WARN", "OCR 未识别到文字。")
        self._timing(ocr_ms=ocr_ms)
        started = time.monotonic()
        answer = scene.answer(question, frame=frame)
        elapsed = int((time.monotonic() - started) * 1000)
        text = _as_text(getattr(answer, "text", ""))
        self._emit(
            "assistant.final",
            {"text": text, "cancelled": False, "source": "debug"},
        )
        self._timing(vision_ms=elapsed)
        self._log(
            "INFO",
            "业务结果完成：%d ms；意图 %s；降级 %s"
            % (
                elapsed,
                _as_text(getattr(answer, "intent", "")) or "未知",
                bool(getattr(answer, "degraded", False)),
            ),
        )

    # ------------------------------------------------------------------
    # compare：上传 A/B 或相机先后两次拍摄
    # ------------------------------------------------------------------
    def _run_compare(self) -> None:
        question = _as_text(self._input.get("question")) or "比较一下这两样东西"
        source = _as_text(self._input.get("source")) or _COMPARE_SOURCE_UPLOAD
        self._compare_state = {"source": source, "a": None, "b": None}
        if source == _COMPARE_SOURCE_UPLOAD:
            frame_a = self.ctx.load_image("A")
            frame_b = self.ctx.load_image("B")
            meta_a = self.ctx.save_frame(frame_a, "A", purpose="compare")
            meta_b = self.ctx.save_frame(frame_b, "B", purpose="compare")
            self._compare_state.update({"a": frame_a, "b": frame_b})
            self._log(
                "INFO",
                "图片 A：ID %s（上传）；图片 B：ID %s（上传）；文件 %s / %s"
                % (
                    _as_text(getattr(frame_a, "frame_id", "")) or "无 ID",
                    _as_text(getattr(frame_b, "frame_id", "")) or "无 ID",
                    meta_a.get("file"),
                    meta_b.get("file"),
                ),
            )
        else:
            self._log(
                "INFO",
                "请依次点击「拍摄图片 A」「拍摄图片 B」；两次拍摄都由你手动触发。",
            )
            while not self.ctx.stopped:
                if self._compare_state.get("a") is not None and self._compare_state.get("b") is not None:
                    break
                self.ctx.wait(0.1)
            if self.ctx.stopped:
                self._log("INFO", "比较调试被停止。")
                return
        frame_a = self._compare_state.get("a")
        frame_b = self._compare_state.get("b")
        if frame_a is None or frame_b is None:
            raise DebugError("missing_image", "缺少图片 A 或 B，无法比较。")
        self._compare_frames(frame_a, frame_b, question)

    def _compare_frames(self, frame_a: Any, frame_b: Any, question: str) -> None:
        self._log("INFO", "调用 SceneService 比较能力…")
        scene = self._make_scene()
        started = time.monotonic()
        answer = scene.compare(frame_a, frame_b, question)
        elapsed = int((time.monotonic() - started) * 1000)
        text = _as_text(getattr(answer, "text", ""))
        self._emit(
            "assistant.final",
            {"text": text, "cancelled": False, "source": "debug"},
        )
        self._timing(vision_ms=elapsed)
        self._log(
            "INFO",
            "比较结果：A=%s；B=%s；降级 %s"
            % (
                _as_text(getattr(answer, "a_frame_id", ""))
                or _as_text(getattr(frame_a, "frame_id", ""))
                or "无 ID",
                _as_text(getattr(answer, "b_frame_id", ""))
                or _as_text(getattr(frame_b, "frame_id", ""))
                or "无 ID",
                bool(getattr(answer, "degraded", False)),
            ),
        )
        for item in list(getattr(answer, "items", ()) or ())[:10]:
            self._log(
                "INFO",
                "  %s：A=%s B=%s%s"
                % (
                    _as_text(getattr(item, "name", "")),
                    _as_text(getattr(item, "a", "")),
                    _as_text(getattr(item, "b", "")),
                    _as_text(getattr(item, "note", "")),
                ),
            )

    def _action_compare_capture(self, action: str) -> dict:
        state = self._compare_state
        if state.get("source") != _COMPARE_SOURCE_CAMERA:
            raise DebugError(
                "invalid_request",
                "当前是上传图片模式，无需拍摄；请以「相机先后拍摄」重新开始调试。",
            )
        slot = "A" if action == "capture_a" else "B"
        if state.get(slot.lower()) is not None:
            raise DebugError("invalid_request", f"图片 {slot} 已拍摄；请点停止后重新开始。")
        self._log("INFO", f"拍摄图片 {slot}…")
        provider = self.ctx.camera_provider()
        try:
            frame = provider.acquire("compare")
        except Exception as exc:  # noqa: BLE001
            raise DebugError("camera_failed", f"拍摄图片 {slot} 失败：{exc}") from exc
        finally:
            try:
                provider.close()
            except Exception:  # noqa: BLE001
                pass
        meta = self.ctx.save_frame(frame, slot, purpose=f"compare_{slot.lower()}")
        state[slot.lower()] = frame
        frame_id = _as_text(getattr(frame, "frame_id", ""))
        self._log(
            "INFO",
            "已拍摄图片 %s：ID %s，%s×%s"
            % (slot, frame_id or "无 ID", meta.get("width"), meta.get("height")),
        )
        return {"slot": slot, "frame_id": frame_id, "file": meta.get("file")}

    # ------------------------------------------------------------------
    # speaker：用户同意后，本轮临时参考 + 核验样本
    # ------------------------------------------------------------------
    def _run_speaker(self) -> None:
        self._log(
            "INFO",
            "已获得用户同意：仅采集本次调试的临时样本，不写入正式声纹档案。",
        )
        speaker_cfg = getattr(self.cfg, "speaker", None)
        if not bool(getattr(speaker_cfg, "enabled", False)):
            self._log(
                "WARN",
                "声纹验证未启用：配置 speaker.enabled=false（依据：正式配置项 speaker.enabled）。",
            )
            self._log(
                "WARN",
                "免唤醒准入未启用：调试未建立话题窗口，也不满足「话题窗口 + 声纹 VERIFIED + directed」条件。",
            )
            return
        gate = None
        try:
            gate = self._track("speaker_gate", self._make_gate())
            count = int(gate.enroll_models_dir() or 0)
        except Exception as exc:  # noqa: BLE001
            models = getattr(self.cfg, "models", None)
            enroll_dir = _as_text(getattr(speaker_cfg, "enroll_dir", "")) or "speaker/enroll"
            self._log(
                "WARN",
                "声纹组件未接入：%s（依据：%s/%s 下没有可用声纹档案）"
                % (exc, getattr(models, "root", "models.root"), enroll_dir),
            )
            self._log(
                "WARN",
                "免唤醒准入未启用：调试未建立话题窗口，也不满足「话题窗口 + 声纹 VERIFIED + directed」条件。",
            )
            return
        if count <= 0:
            models = getattr(self.cfg, "models", None)
            enroll_dir = _as_text(getattr(speaker_cfg, "enroll_dir", "")) or "speaker/enroll"
            self._log(
                "WARN",
                "声纹验证未启用：档案为空（依据：%s/%s 下没有已注册样本）。"
                % (getattr(models, "root", "models.root"), enroll_dir),
            )
            self._log(
                "WARN",
                "免唤醒准入未启用：调试未建立话题窗口，也不满足「话题窗口 + 声纹 VERIFIED + directed」条件。",
            )
            return
        self._log("INFO", f"声纹档案已加载：{count} 条。开始采集临时参考音频（3 秒）…")
        reference, sample_rate = self._record_audio(SPEAKER_SAMPLE_SECONDS, "参考音频")
        if self.ctx.stopped:
            return
        if getattr(reference, "size", 0) == 0:
            raise DebugError("speaker_sample_invalid", "参考音频为空，无法建立临时样本。")
        enrolled = bool(gate.enroll(reference, sample_rate))
        self._log(
            "INFO",
            "参考音频注册（仅内存临时样本）：%s" % ("成功" if enrolled else "失败"),
        )
        if not enrolled:
            raise DebugError(
                "speaker_sample_invalid",
                "参考音频过短或无效，未能建立临时声纹样本。",
            )
        self._log("INFO", "开始采集临时核验音频（3 秒）…")
        verify_samples, _verify_rate = self._record_audio(SPEAKER_SAMPLE_SECONDS, "核验音频")
        if self.ctx.stopped:
            return
        started = time.monotonic()
        result = gate.verify(verify_samples)
        elapsed = int((time.monotonic() - started) * 1000)
        status = getattr(result, "status", "")
        status_name = _as_text(getattr(status, "name", status)) or "UNKNOWN"
        score = float(getattr(result, "score", 0.0) or 0.0)
        reason = _as_text(getattr(result, "reason", "")) or "无"
        threshold = float(getattr(speaker_cfg, "threshold", 0.0) or 0.0)
        self._timing(speaker_ms=elapsed)
        self._log(
            "INFO",
            "声纹核验：分数 %.3f，阈值 %.3f，状态 %s，原因 %s"
            % (score, threshold, status_name, reason),
        )
        self._log(
            "INFO",
            "准入状态：%s" % ("通过（VERIFIED）" if status_name == "VERIFIED" else "未通过"),
        )
        self._log(
            "WARN",
            "免唤醒准入未启用：需要话题窗口打开且 directed=True，调试未建立该条件。",
        )
        self._log("INFO", "临时参考/核验样本仅保存在内存中，未写入正式声纹档案。")

    # ------------------------------------------------------------------
    # contract：只读配置 + 显式测试事件（不连接真实主控）
    # ------------------------------------------------------------------
    def _contract_config(self) -> Any:
        return getattr(self.cfg, "contract", None)

    def _run_contract(self) -> None:
        contract = self._contract_config()
        enabled = bool(getattr(contract, "enabled", False))
        host = _as_text(getattr(contract, "host", "")) or "-"
        port = getattr(contract, "port", "-")
        role = _as_text(getattr(contract, "role", "")) or "-"
        self._contract_state = {
            "enabled": enabled,
            "host": host,
            "port": port,
            "role": role,
        }
        if not enabled:
            self._log(
                "WARN",
                "合同未接入：contract.enabled=false（依据：正式配置未启用合同链路；"
                "调试不连接真实主控，不发送任何指令）。",
            )
        else:
            self._log(
                "WARN",
                "合同配置已启用（%s:%s，role=%s），但功能调试按约定不连接真实主控、"
                "不发送真实控制指令；这里只展示状态。" % (host, port, role),
            )
        self._log("INFO", "可点击「查询合同状态」；点停止结束调试。")
        while not self.ctx.stopped:
            if self.ctx.wait(0.2):
                break
        self._log("INFO", "合同联调调试结束。")

    def _action_contract(self, action: str) -> dict:
        state = dict(self._contract_state)
        enabled = bool(state.get("enabled", False))
        if action == "status":
            self._log(
                "INFO",
                "合同状态：%s；host=%s；port=%s；role=%s；调试连接=false"
                % (
                    "已启用" if enabled else "未接入",
                    state.get("host", "-"),
                    state.get("port", "-"),
                    state.get("role", "-"),
                ),
            )
            payload = dict(state)
            payload.update({"connected": False, "note": "调试不连接真实主控"})
            return payload
        if not enabled:
            self._log("WARN", "合同未接入：未发送测试事件（contract.enabled=false）。")
            return {"sent": False, "reason": "contract_disabled"}
        self._log(
            "WARN",
            "按调试约束不连接真实主控，未发送测试事件；如使用替身，请标记替身并在独立环境显式验证。",
        )
        return {"sent": False, "reason": "debug_offline"}

    # ------------------------------------------------------------------
    # interruption：真实交互链路，记录真实事件，不注入取消
    # ------------------------------------------------------------------
    def _build_interruption_bundle(self) -> dict[str, Any]:
        from aieyes.app import _make_source, build_engine
        from aieyes.audio.output import AudioSink
        from aieyes.audio.vad import EnergyVad
        from aieyes.chat import ChatSession
        from aieyes.events import EventBus
        from aieyes.scene.providers import CameraFrameProvider
        from aieyes.scene.service import SceneService
        from aieyes.speech.asr import StreamingAsr
        from aieyes.speech.frontend import SpeechFrontend
        from aieyes.speech.kws import SherpaKws
        from aieyes.speech.tts import PiperTts
        from aieyes.vision.ocr import OcrEngine
        from aieyes.vision.vlm import VlmClient

        from bridge import Bridge
        from media import CameraHubAdapter

        # 调试绝不连接真实主控：只在内存副本里关闭合同，不改正式配置
        cfg = copy.deepcopy(self.cfg)
        try:
            cfg.contract.enabled = False
        except Exception:  # noqa: BLE001
            pass
        bridge = Bridge(
            _EmitAdapter(self.ctx),
            run_id=self.ctx.run_id,
            frame_store=getattr(self.ctx, "frame_store", None),
        )
        hub = self.ctx.ensure_hub()
        bus = EventBus()
        observed_bus = bridge.wrap_bus(bus)
        provider = bridge.wrap_provider(
            CameraFrameProvider(camera=CameraHubAdapter(hub, reason="debug"))
        )
        vlm = VlmClient(cfg.llm, cfg.vlm)
        ocr = OcrEngine(cfg.ocr, ocr_dir=cfg.models.resolve(cfg.models.ocr_dir))
        scene = SceneService(vlm=vlm, ocr=ocr, provider=provider)
        observed_scene = bridge.wrap_scene(scene)
        observed_chat = bridge.wrap_chat(ChatSession(vlm))
        observed_tts = bridge.wrap_tts(PiperTts(cfg.tts, cfg.models))
        sink = AudioSink(device=None, sample_rate=cfg.audio.sample_rate)
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
            bus=observed_bus,
            chat=observed_chat,
            scene=observed_scene,
            sink=sink,
            tts=observed_tts,
            frontend=frontend,
        )
        return {
            "cfg": cfg,
            "bridge": bridge,
            "engine": engine,
            "deps": deps,
            "sink": sink,
            "frontend": frontend,
            "tts": observed_tts,
            "scene": scene,
            "provider": provider,
            "vlm": vlm,
            "ocr": ocr,
        }

    def _run_interruption(self) -> None:
        import asyncio
        import inspect

        self._log("INFO", "装配真实交互链路（KWS/VAD/ASR/对话/播报/相机）…")
        bundle = self._build_interruption_bundle()
        engine = bundle["engine"]
        deps = bundle["deps"]
        sink = bundle["sink"]
        frontend = bundle["frontend"]

        async def _cleanup() -> None:
            async def _call(label: str, fn: Any) -> None:
                try:
                    result = fn()
                    if inspect.isawaitable(result):
                        await result
                except Exception as exc:  # noqa: BLE001
                    self._log("WARN", f"{label} 释放失败：{exc}")

            await _call("engine.stop", engine.stop)
            await _call("frontend.stop", frontend.stop)
            _safe_close("sink", sink, self.ctx)
            _safe_close("tts", bundle["tts"], self.ctx)
            _safe_close("scene", bundle["scene"], self.ctx)
            _safe_close("provider", bundle["provider"], self.ctx)
            _safe_close("ocr", bundle["ocr"], self.ctx)
            _safe_close("vlm", bundle["vlm"], self.ctx)
            identity = deps.get("identity") if isinstance(deps, dict) else None
            if identity is not None:
                _safe_close("identity.gate", getattr(identity, "gate", identity), self.ctx)
            self._log("INFO", "交互链路已关闭。")

        async def _drive() -> None:
            # 清理必须与 engine.start 在同一个 event loop：引擎的消费任务
            # 绑定在启动时的 loop 上，跨 loop await 会失败。
            sink.start()
            try:
                await engine.start()
                self._log(
                    "INFO",
                    "交互链路已启动。请让设备回答，然后插话/说停止/继续；"
                    "调试只记录真实事件，不注入取消事件。点停止结束调试。",
                )
                frontend.start()
                while not self.ctx.stopped:
                    await asyncio.sleep(0.1)
            finally:
                try:
                    await _cleanup()
                except Exception as exc:  # noqa: BLE001
                    self._log("WARN", f"交互链路清理失败：{exc}")

        asyncio.run(_drive())
