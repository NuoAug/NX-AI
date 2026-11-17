"""测试台媒体层：共享相机中枢、帧落盘、上传图片管理。

职责边界（见 ``testbench/执行方案.md`` 8.1/8.2）：

* :class:`CameraHub` —— 进程内唯一相机所有者。首个引用时启动唯一的采集线程并
  ``open()`` 相机，持续 ``grab()`` 缓存最新帧；按 ``preview_fps`` 上限把最新帧编码
  为 JPEG 并回调 ``on_preview``；最后一个引用移除时停止线程并 ``close()`` 相机。
* :class:`CameraHubAdapter` —— 把 hub 适配成 ``aieyes`` 的 ``Camera`` 协议，
  可直接交给现有 ``CameraFrameProvider``，使质量门控/重拍继续生效。
* :class:`FrameStore` —— 把实际进入业务的帧写入 ``runs/<run_id>/media/``。
* :class:`UploadStore` —— A/B 两槽上传图片的校验、EXIF 规范化、持久化与延迟回收。
* :func:`image_to_jpeg_bytes` —— 预览/缩略图 JPEG 编码辅助函数。

依赖：标准库 + Pillow（``ImageFrame.rgb`` 是 numpy 数组，PIL 可直接从数组构造，
因此本模块不导入 numpy）。

线程模型：``on_preview`` 不在采集线程内执行，而是由单槽预览线程转发；
慢速消费方（HTTP 推送）不会阻塞设备取帧，过时预览被直接丢弃。
"""

from __future__ import annotations

import io
import json
import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image, ImageOps, UnidentifiedImageError

__all__ = [
    "CameraHub",
    "CameraHubAdapter",
    "FrameStore",
    "UploadStore",
    "UploadError",
    "image_to_jpeg_bytes",
]


# 采集线程单次 grab 的等待上限：既要覆盖低帧率设备，又要保证 stop() 能及时返回。
_GRAB_TIMEOUT_MS = 1000
_ERROR_BACKOFF_START_S = 0.1
_ERROR_BACKOFF_MAX_S = 2.0
_JOIN_TIMEOUT_S = 5.0
# 连续这么多次 grab 失败就释放并重开设备（断连/拔插后恢复）。
_RECONNECT_AFTER_FAILURES = 3

_FRAME_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_FRAME_ID_LEN = 120

_ALLOWED_UPLOAD_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
_SLOT_NAMES = ("A", "B")
_UPLOAD_IMAGE_ID = re.compile(r"^[0-9a-f]{32}$")
_UPLOAD_JPEG_QUALITY = 95


def _pillow_rgb(rgb: Any) -> Image.Image:
    """把 ``ImageFrame.rgb``（uint8 HxWx3 ndarray）或 PIL 图像转成 RGB 图像。"""
    if isinstance(rgb, Image.Image):
        image = rgb
    else:
        image = Image.fromarray(rgb)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _clamp_quality(quality: int) -> int:
    return max(1, min(95, int(quality)))


def _safe_filename(value: Any, *, fallback: str = "frame") -> str:
    """把外部 frame_id 收敛成安全文件名（仅 ``[A-Za-z0-9._-]``，无路径分隔符）。"""
    text = _FRAME_ID_UNSAFE.sub("_", str(value or "").strip())
    text = text.strip("._")[:_MAX_FRAME_ID_LEN].strip("._")
    return text or fallback


def image_to_jpeg_bytes(
    rgb: Any, quality: int = 80, max_side: int | None = None
) -> bytes:
    """把 RGB 帧编码为 JPEG 字节；``max_side`` 限制长边（等比缩小）。"""
    image = _pillow_rgb(rgb)
    if max_side is not None:
        limit = int(max_side)
        if limit > 0 and max(image.size) > limit:
            image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_clamp_quality(quality))
    return buffer.getvalue()


class CameraProtocol(Protocol):
    """与 ``aieyes.interfaces.Camera`` 同形（只用于类型说明，不做运行时检查）。"""

    def open(self) -> None: ...

    def grab(self, timeout_ms: int = 10_000) -> Any: ...

    def close(self) -> None: ...


class CameraHub:
    """进程内唯一的相机持有者：引用计数启停、单线程采集、限速预览。

    Parameters
    ----------
    camera_factory:
        无参可调用对象，返回待打开的相机对象（如 ``lambda: V4L2Camera()``）。
        工厂在采集线程内调用，返回值必须有 ``open()/grab(timeout_ms)/close()``，
        可选 ``describe()``。
    jpeg_quality:
        预览 JPEG 质量（1~95）。
    preview_fps:
        预览编码上限（帧/秒）；``<=0`` 关闭预览回调。
    preview_max_side:
        预览长边上限（等比缩小），``None``/0 表示不缩放。
    clock:
        单调时钟，返回秒；用于 ``t_mono_ms`` 与超时计算。
    on_preview:
        ``on_preview(jpeg_bytes, frame_id, t_mono_ms)``；在独立预览线程调用，
        调用方只应做轻量转发，不得阻塞。
    log:
        可选日志回调，接收单行字符串。
    """

    def __init__(
        self,
        camera_factory: Callable[[], Any],
        *,
        jpeg_quality: int = 80,
        preview_fps: float = 5,
        preview_max_side: int | None = 960,
        clock: Callable[[], float] = time.monotonic,
        on_preview: Callable[[bytes, str, int], None] | None = None,
        log: Callable[[str], None] | None = None,
    ) -> None:
        if not callable(camera_factory):
            raise TypeError("camera_factory must be callable")
        self._factory = camera_factory
        self._jpeg_quality = _clamp_quality(jpeg_quality)
        self._preview_fps = float(preview_fps)
        self._preview_interval = (
            1.0 / self._preview_fps if self._preview_fps > 0 else 0.0
        )
        self._preview_max_side = int(preview_max_side) if preview_max_side else None
        self._clock = clock
        self._on_preview = on_preview
        self._log_fn = log

        self._cond = threading.Condition()
        self._refs: dict[str, int] = {}
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._camera: Any = None
        self._camera_open = False
        self._frame: Any = None
        self._frame_t: int | None = None
        self._last_frame_ts: int | None = None
        self._error: str | None = None
        self._closing = False
        self._stopped = False
        self._stopped_event = threading.Event()

        self._next_preview_at = 0.0
        self._preview_lock = threading.Lock()
        self._preview_pending: tuple[Any, int] | None = None
        self._preview_event = threading.Event()
        self._preview_stop = threading.Event()
        self._preview_thread: threading.Thread | None = None
        self._logged_error = ""

    # --- 引用计数 -------------------------------------------------------
    def add_ref(self, reason: str = "task") -> int:
        """增加一个引用；首次引用时启动采集线程并打开相机。返回当前总引用数。"""
        reason = str(reason or "task")
        with self._cond:
            if self._stopped:
                raise RuntimeError("CameraHub has been stopped")
            while self._closing:
                self._cond.wait(0.1)
                if self._stopped:
                    raise RuntimeError("CameraHub has been stopped")
            self._refs[reason] = self._refs.get(reason, 0) + 1
            total = sum(self._refs.values())
            if total == 1 and self._thread is None:
                self._start_capture_locked()
            return total

    def remove_ref(self, reason: str = "task") -> int:
        """减少一个引用；最后一个引用移除时停止采集线程并关闭相机。"""
        reason = str(reason or "task")
        thread: threading.Thread | None = None
        with self._cond:
            count = self._refs.get(reason, 0)
            if count <= 0:
                return sum(self._refs.values())
            if count == 1:
                del self._refs[reason]
            else:
                self._refs[reason] = count - 1
            total = sum(self._refs.values())
            if total == 0 and self._thread is not None:
                thread = self._begin_shutdown_locked()
        if thread is not None:
            self._join_capture(thread)
        return 0 if thread is not None else total

    def ref_count(self) -> int:
        with self._cond:
            return sum(self._refs.values())

    def _start_capture_locked(self) -> None:
        self._stop_event = threading.Event()
        self._next_preview_at = 0.0
        self._logged_error = ""
        # 上一轮采集的缓存帧不属于本次请求，重新开流时清掉。
        self._frame = None
        self._frame_t = None
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="camera-hub-capture",
            daemon=True,
        )
        self._thread.start()

    def _begin_shutdown_locked(self) -> threading.Thread | None:
        thread = self._thread
        if thread is None:
            return None
        self._closing = True
        self._stop_event.set()
        # 丢弃未转发的预览，避免停止后还回调 on_preview。
        with self._preview_lock:
            self._preview_pending = None
        self._preview_event.clear()
        self._cond.notify_all()
        return thread

    def _join_capture(self, thread: threading.Thread) -> None:
        thread.join(timeout=_JOIN_TIMEOUT_S)
        if thread.is_alive():
            self._log_error("camera capture thread did not stop in time")
        with self._cond:
            if self._thread is thread:
                self._thread = None
            self._closing = False
            self._camera_open = False
            # 采集线程退出后不再持有可复用帧，latest() 返回 None。
            self._frame = None
            self._frame_t = None
            self._cond.notify_all()

    # --- 状态 / 取帧 ----------------------------------------------------
    def latest(self) -> tuple[Any, int] | None:
        """返回 ``(ImageFrame, t_mono_ms)``；尚无帧时返回 ``None``。"""
        with self._cond:
            if self._frame is None or self._frame_t is None:
                return None
            return self._frame, int(self._frame_t)

    def wait_newer_than(self, t_mono_ms: int, timeout_ms: int) -> Any:
        """等待一帧捕获时间晚于 ``t_mono_ms``；超时或未采集时抛 ``TimeoutError``。"""
        return self._wait_newer_than(t_mono_ms, timeout_ms)[0]

    def _wait_newer_than(self, t_mono_ms: int, timeout_ms: int) -> tuple[Any, int]:
        threshold = int(t_mono_ms)
        deadline = self._clock() + max(0, int(timeout_ms)) / 1000.0
        with self._cond:
            while True:
                if self._frame is not None and self._frame_t is not None:
                    if self._frame_t > threshold:
                        return self._frame, int(self._frame_t)
                if self._thread is None or not self._thread.is_alive():
                    raise TimeoutError("camera hub is not capturing")
                remaining = deadline - self._clock()
                if remaining <= 0:
                    raise TimeoutError(
                        f"no camera frame newer than {threshold} within {timeout_ms} ms"
                    )
                self._cond.wait(remaining)

    def status(self) -> dict[str, Any]:
        with self._cond:
            return {
                "active": sum(self._refs.values()) > 0,
                "camera_open": bool(self._camera_open),
                "last_frame_ts": (
                    int(self._last_frame_ts)
                    if self._last_frame_ts is not None
                    else None
                ),
                "error": self._error,
            }

    def describe(self) -> str:
        with self._cond:
            camera = self._camera
        if camera is None:
            return "camera-hub(closed)"
        describe = getattr(camera, "describe", None)
        if callable(describe):
            try:
                return str(describe())
            except Exception:  # noqa: BLE001 - 描述失败不该影响采样
                pass
        return "camera-hub(open)"

    def stop(self) -> None:
        """永久停止：关闭采集/预览线程与相机，之后不再回调 ``on_preview``。"""
        self._stopped_event.set()
        with self._cond:
            if self._stopped:
                already = True
            else:
                already = False
                self._stopped = True
                self._refs.clear()
                self._cond.notify_all()
                thread = self._begin_shutdown_locked()
            if already:
                thread = None
        if thread is not None:
            self._join_capture(thread)
        self._preview_stop.set()
        self._preview_event.set()
        preview = self._preview_thread
        if preview is not None and preview.is_alive():
            preview.join(timeout=_JOIN_TIMEOUT_S)
        with self._cond:
            self._camera = None
            self._camera_open = False
            self._cond.notify_all()

    # --- 采集线程 -------------------------------------------------------
    def _capture_loop(self) -> None:
        camera: Any = None
        stop_event = self._stop_event
        try:
            with self._cond:
                camera = self._camera
            if camera is None:
                try:
                    camera = self._factory()
                except Exception as exc:  # noqa: BLE001 - 工厂失败要上报到 status
                    self._set_error(f"camera factory failed: {exc}")
                    return
                # 一台设备只创建一个实例，预览/任务交替启停时复用同一对象。
                with self._cond:
                    self._camera = camera
            backoff = _ERROR_BACKOFF_START_S
            opened = False
            # 连续失败到阈值就认为句柄已失效（拔线/设备重置/驱动崩溃）并重开，
            # 否则会永远在坏句柄上重试，画面再也回不来。
            grab_failures = 0
            while not stop_event.is_set():
                if not opened:
                    try:
                        camera.open()
                    except Exception as exc:  # noqa: BLE001 - 设备忙/拔插可恢复
                        self._set_error(f"camera open failed: {exc}")
                        backoff = self._sleep_backoff(stop_event, backoff)
                        continue
                    opened = True
                    backoff = _ERROR_BACKOFF_START_S
                    with self._cond:
                        self._camera_open = True
                        self._error = None
                        self._logged_error = ""
                        self._cond.notify_all()
                try:
                    frame = camera.grab(timeout_ms=_GRAB_TIMEOUT_MS)
                except Exception as exc:  # noqa: BLE001 - grab 失败必须记录并重试
                    self._set_error(f"camera grab failed: {exc}")
                    grab_failures += 1
                    if grab_failures >= _RECONNECT_AFTER_FAILURES:
                        grab_failures = 0
                        opened = False
                        with self._cond:
                            self._camera_open = False
                            self._cond.notify_all()
                        try:
                            camera.close()
                        except Exception as close_exc:  # noqa: BLE001 - 释放失败只记录
                            self._log_error(f"camera close failed: {close_exc}")
                    backoff = self._sleep_backoff(stop_event, backoff)
                    continue
                grab_failures = 0
                backoff = _ERROR_BACKOFF_START_S
                now = self._clock()
                t_ms = int(now * 1000)
                with self._cond:
                    self._frame = frame
                    self._frame_t = t_ms
                    self._last_frame_ts = t_ms
                    self._error = None
                    self._logged_error = ""
                    self._cond.notify_all()
                self._schedule_preview(frame, t_ms, now)
        finally:
            if camera is not None:
                try:
                    camera.close()
                except Exception as exc:  # noqa: BLE001 - 释放失败只记录
                    self._log_error(f"camera close failed: {exc}")
            with self._cond:
                self._camera_open = False
                self._cond.notify_all()

    def _sleep_backoff(self, stop_event: threading.Event, backoff: float) -> float:
        # 用 Event.wait 而不是 sleep，保证 stop()/零引用能立即唤醒退出。
        stop_event.wait(backoff)
        return min(backoff * 2.0, _ERROR_BACKOFF_MAX_S)

    def _set_error(self, message: str) -> None:
        with self._cond:
            self._error = message[:300]
            self._cond.notify_all()
        self._log_error(message)

    def _log_error(self, message: str) -> None:
        # 同一错误只记一次，避免告警刷屏。
        if message == self._logged_error:
            return
        self._logged_error = message
        if self._log_fn is not None:
            try:
                self._log_fn(message)
            except Exception:  # noqa: BLE001 - 日志失败不得影响采集
                pass

    # --- 预览 -----------------------------------------------------------
    def _schedule_preview(self, frame: Any, t_ms: int, now: float) -> None:
        if self._on_preview is None or self._preview_interval <= 0:
            return
        if now < self._next_preview_at:
            return
        self._next_preview_at = now + self._preview_interval
        with self._preview_lock:
            self._preview_pending = (frame, t_ms)
        self._ensure_preview_thread()
        self._preview_event.set()

    def _ensure_preview_thread(self) -> None:
        thread = self._preview_thread
        if thread is not None and thread.is_alive():
            return
        thread = threading.Thread(
            target=self._preview_loop, name="camera-hub-preview", daemon=True
        )
        self._preview_thread = thread
        thread.start()

    def _preview_loop(self) -> None:
        callback = self._on_preview
        if callback is None:
            return
        while not self._preview_stop.is_set():
            if not self._preview_event.wait(0.1):
                continue
            self._preview_event.clear()
            with self._preview_lock:
                pending = self._preview_pending
                self._preview_pending = None
            if pending is None:
                continue
            if self._stopped_event.is_set():
                break
            frame, t_ms = pending
            try:
                jpeg = image_to_jpeg_bytes(
                    frame.rgb,
                    quality=self._jpeg_quality,
                    max_side=self._preview_max_side,
                )
            except Exception as exc:  # noqa: BLE001 - 单帧编码失败不致命
                self._log_error(f"preview encode failed: {exc}")
                continue
            if self._stopped_event.is_set():
                break
            frame_id = str(getattr(frame, "frame_id", "") or f"cam-{t_ms}")
            try:
                callback(jpeg, frame_id, t_ms)
            except Exception as exc:  # noqa: BLE001 - 消费方异常不得杀死预览线程
                self._log_error(f"on_preview callback failed: {exc}")


class CameraHubAdapter:
    """把 :class:`CameraHub` 适配成 ``aieyes`` 的 ``Camera`` 协议。

    ``grab()`` 只返回本次请求开始之后捕获的新帧，保留帧原始的捕获时间、
    分辨率与 ``source``；``CameraFrameProvider`` 的质量门控/重拍逻辑不受影响。
    """

    def __init__(self, hub: CameraHub, *, reason: str = "task") -> None:
        self._hub = hub
        self._reason = str(reason or "task")
        self._last_t = 0
        self._open = False

    def open(self) -> None:
        if self._open:
            return
        self._hub.add_ref(self._reason)
        self._open = True
        self._last_t = 0

    def grab(self, timeout_ms: int = 10_000) -> Any:
        if not self._open:
            self.open()
        frame, t_ms = self._hub._wait_newer_than(self._last_t, timeout_ms)
        self._last_t = t_ms
        return frame

    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        self._hub.remove_ref(self._reason)

    def describe(self) -> str:
        return self._hub.describe()


class FrameStore:
    """把实际进入业务的帧写入 ``<run_dir>/media/<frame_id>.jpg``。"""

    def __init__(self, run_dir: str | Path, *, jpeg_quality: int = 85) -> None:
        self._run_dir = Path(run_dir)
        self._media_dir = self._run_dir / "media"
        self._jpeg_quality = _clamp_quality(jpeg_quality)

    @property
    def media_dir(self) -> Path:
        return self._media_dir

    def save(self, frame: Any) -> dict[str, Any]:
        image = _pillow_rgb(frame.rgb)
        frame_id = str(getattr(frame, "frame_id", "") or "")
        if frame_id.strip():
            name = _safe_filename(frame_id)
        else:
            name = _safe_filename(f"frame-{int(getattr(frame, 't_capture_ms', 0) or 0)}")
        self._media_dir.mkdir(parents=True, exist_ok=True)
        path = self._media_dir / f"{name}.jpg"
        image.save(path, format="JPEG", quality=self._jpeg_quality)
        return {
            "file": f"media/{path.name}",
            "width": int(image.width),
            "height": int(image.height),
        }


class UploadError(Exception):
    """上传校验失败；``code`` 携带 HTTP 语义码（413/415/422）。"""

    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)

    def __str__(self) -> str:
        return self.message


class UploadStore:
    """A/B 两槽上传图片：校验、EXIF 规范化、索引持久化与延迟回收。

    索引为 ``<upload_dir>/index.json``；图片统一以随机 ``image_id`` 命名并另存为
    JPEG。客户端文件名/路径一律忽略。启动时加载索引，文件与索引不一致时以实际
    文件为准（缺失项清理、多余文件收养）；索引损坏则按目录内容重建（槽位归属
    丢失，图片仍可读）。
    """

    def __init__(
        self,
        upload_dir: str | Path,
        *,
        max_bytes: int = 20 * 1024 * 1024,
        max_pixels: int = 24_000_000,
    ) -> None:
        self._dir = Path(upload_dir)
        self._max_bytes = int(max_bytes)
        self._max_pixels = int(max_pixels)
        self._index_path = self._dir / "index.json"
        self._lock = threading.RLock()
        self._items: dict[str, dict[str, Any]] = {}
        self._slots: dict[str, str | None] = {name: None for name in _SLOT_NAMES}
        self._dir.mkdir(parents=True, exist_ok=True)
        self._load()

    # --- 上传 -----------------------------------------------------------
    def save(self, raw: bytes, slot: str) -> dict[str, Any]:
        """校验并保存一张上传图片到槽位 ``slot``（"A"/"B"），返回公开 meta。"""
        if slot not in _SLOT_NAMES:
            raise ValueError(f"slot must be 'A' or 'B', got {slot!r}")
        if not isinstance(raw, (bytes, bytearray, memoryview)):
            raise UploadError(415, "uploaded payload is not binary image data")
        data = bytes(raw)
        if not data:
            raise UploadError(415, "empty upload")
        if len(data) > self._max_bytes:
            raise UploadError(413, f"image larger than {self._max_bytes} bytes")
        try:
            with Image.open(io.BytesIO(data)) as probe:
                fmt = (probe.format or "").upper()
                if fmt not in _ALLOWED_UPLOAD_FORMATS:
                    raise UploadError(415, f"unsupported image format: {fmt or 'unknown'}")
                width, height = probe.size
                if width <= 0 or height <= 0:
                    raise UploadError(415, "image has invalid dimensions")
                if width * height > self._max_pixels:
                    raise UploadError(
                        413, f"image has too many pixels ({width}x{height})"
                    )
                probe.load()  # 真正解码：识别伪图片、截断图片与解码炸弹
                image = ImageOps.exif_transpose(probe)
        except UploadError:
            raise
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as exc:
            raise UploadError(415, f"image decode failed: {exc}") from exc

        if image.mode != "RGB":
            image = image.convert("RGB")
        out_w, out_h = image.size
        image_id = secrets.token_hex(16)
        destination = self._dir / f"{image_id}.jpg"
        # 4:4:4 不做色度抽样：红色小字在 4:2:0 下会糊掉，OCR 读不出“高压”
        image.save(destination, format="JPEG", quality=_UPLOAD_JPEG_QUALITY, subsampling=0)

        meta = {
            "image_id": image_id,
            "slot": slot,
            "width": int(out_w),
            "height": int(out_h),
            "format": "jpeg",
        }
        with self._lock:
            previous = self._slots[slot]
            if previous is not None and previous != image_id:
                self._retire_locked(previous)
            self._items[image_id] = {
                **meta,
                "in_use": False,
                "pending": False,
            }
            self._slots[slot] = image_id
            self._persist_locked()
        return dict(meta)

    # --- 读取 / 槽位 ----------------------------------------------------
    def get(self, image_id: str) -> tuple[Path, str]:
        """返回 ``(path, "image/jpeg")``；不存在抛 ``KeyError``。"""
        with self._lock:
            record = self._items.get(image_id)
            if record is None:
                raise KeyError(image_id)
            path = self._dir / f"{image_id}.jpg"
            if not path.is_file():
                # 索引与文件不一致：以实际文件为准
                self._unlink_locked(image_id)
                self._persist_locked()
                raise KeyError(image_id)
            return path, "image/jpeg"

    def slots(self) -> dict[str, dict[str, Any] | None]:
        with self._lock:
            result: dict[str, dict[str, Any] | None] = {}
            for name in _SLOT_NAMES:
                image_id = self._slots[name]
                record = self._items.get(image_id) if image_id else None
                result[name] = self._public(record) if record is not None else None
            return result

    def set_in_use(self, image_id: str, in_use: bool) -> bool:
        """标记图片是否被活动任务引用；释放最后一个引用时真正回收待删文件。"""
        with self._lock:
            record = self._items.get(image_id)
            if record is None:
                return False
            record["in_use"] = bool(in_use)
            if not in_use and record.get("pending"):
                self._unlink_locked(image_id)
            self._persist_locked()
            return True

    def is_pending(self, image_id: str) -> bool:
        """该图片是否已从槽位删除、但因仍被引用而待回收。"""
        with self._lock:
            record = self._items.get(image_id)
            return bool(record and record.get("pending"))

    def delete(self, image_id: str) -> bool:
        """删除图片；仍被引用时不删文件，只标记 pending 并清空槽位。"""
        with self._lock:
            record = self._items.get(image_id)
            if record is None:
                return False
            if record.get("in_use"):
                record["pending"] = True
                for name, assigned in self._slots.items():
                    if assigned == image_id:
                        self._slots[name] = None
                record["slot"] = None
                self._persist_locked()
                return True
            self._unlink_locked(image_id)
            self._persist_locked()
            return True

    # --- 内部：索引与回收 -----------------------------------------------
    @staticmethod
    def _public(record: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_id": record["image_id"],
            "slot": record.get("slot"),
            "width": int(record["width"]),
            "height": int(record["height"]),
            "format": "jpeg",
        }

    def _retire_locked(self, image_id: str) -> None:
        """把图片移出槽位：无人引用则立即删文件，否则挂起 pending。"""
        record = self._items.get(image_id)
        if record is None:
            return
        for name, assigned in self._slots.items():
            if assigned == image_id:
                self._slots[name] = None
        record["slot"] = None
        if record.get("in_use"):
            record["pending"] = True
            return
        self._unlink_locked(image_id)

    def _unlink_locked(self, image_id: str) -> None:
        self._items.pop(image_id, None)
        for name, assigned in self._slots.items():
            if assigned == image_id:
                self._slots[name] = None
        try:
            (self._dir / f"{image_id}.jpg").unlink()
        except OSError:
            pass

    def _persist_locked(self) -> None:
        payload = {
            "version": 1,
            "slots": dict(self._slots),
            "items": {image_id: dict(record) for image_id, record in self._items.items()},
        }
        temporary = self._index_path.with_name(self._index_path.name + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        os.replace(temporary, self._index_path)

    def _load(self) -> None:
        data: dict[str, Any] | None = None
        try:
            loaded = json.loads(self._index_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and int(loaded.get("version", 0)) == 1:
                data = loaded
        except (OSError, ValueError, TypeError):
            data = None

        items: dict[str, dict[str, Any]] = {}
        slots: dict[str, str | None] = {name: None for name in _SLOT_NAMES}
        if data is not None:
            raw_items = data.get("items")
            if isinstance(raw_items, dict):
                for image_id, record in raw_items.items():
                    if not isinstance(image_id, str) or not _UPLOAD_IMAGE_ID.match(image_id):
                        continue
                    if not isinstance(record, dict):
                        continue
                    items[image_id] = {
                        "image_id": image_id,
                        "slot": None,
                        "width": int(record.get("width") or 0),
                        "height": int(record.get("height") or 0),
                        "format": "jpeg",
                        # 上一进程的引用关系随进程结束，启动时不保留 in_use
                        "in_use": False,
                        "pending": bool(record.get("pending")),
                    }
            raw_slots = data.get("slots")
            if isinstance(raw_slots, dict):
                for name in _SLOT_NAMES:
                    assigned = raw_slots.get(name)
                    if isinstance(assigned, str) and assigned in items:
                        slots[name] = assigned
            for name in _SLOT_NAMES:
                record = items.get(slots[name]) if slots[name] else None
                if record is not None:
                    record["slot"] = name

        # 以实际文件为准：清理缺失项，收养索引外的合法图片。
        present = {
            path.stem
            for path in self._dir.glob("*.jpg")
            if _UPLOAD_IMAGE_ID.match(path.stem)
        }
        for image_id in list(items):
            if image_id not in present:
                items.pop(image_id)
        for image_id in sorted(present - set(items)):
            try:
                with Image.open(self._dir / f"{image_id}.jpg") as image:
                    width, height = image.size
            except Exception:  # noqa: BLE001 - 无法解码的孤儿文件直接清理
                try:
                    (self._dir / f"{image_id}.jpg").unlink()
                except OSError:
                    pass
                continue
            items[image_id] = {
                "image_id": image_id,
                "slot": None,
                "width": int(width),
                "height": int(height),
                "format": "jpeg",
                "in_use": False,
                "pending": False,
            }

        # 延迟回收：pending 且无引用的文件在启动时清理。
        for image_id in list(items):
            record = items[image_id]
            if record.get("pending") and not record.get("in_use"):
                try:
                    (self._dir / f"{image_id}.jpg").unlink()
                except OSError:
                    pass
                items.pop(image_id)

        # 槽位一致性：同一 id 只归属一个槽位，指向已删图片的槽位清空。
        assigned_ids: set[str] = set()
        for name in _SLOT_NAMES:
            image_id = slots[name]
            if image_id is None or image_id not in items or image_id in assigned_ids:
                slots[name] = None
            else:
                assigned_ids.add(image_id)
        for record in items.values():
            record["slot"] = None
        for name, image_id in slots.items():
            if image_id is not None:
                items[image_id]["slot"] = name

        self._items = items
        self._slots = slots
        self._persist_locked()
