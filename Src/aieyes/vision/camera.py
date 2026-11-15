"""V4L2 取图相机：``fcntl`` ioctl + ``mmap``，不引入新依赖。

实现 ``interfaces.Camera`` 协议，输出 ``ImageFrame(rgb=uint8 HxWx3,
t_capture_ms=clock.mono_ms())``。

设计要点
--------
* **自动选节点/格式/分辨率**：``VisionConfig.device`` 默认 ``/dev/video0``，
  但它可能是 Intel RealSense 的深度/红外节点。``open()`` 会枚举
  ``/dev/video*`` 的 QUERYCAP/ENUM_FMT，优先使用配置节点（若它是
  彩色节点），否则挑选支持 ``YUYV``（其次 ``MJPG``）的节点，并按
  ``VisionConfig.width/height/fps`` 选最接近的档位。
* **捕获时间**：优先用内核随帧返回的 monotonic 时间戳（CLOCK_MONOTONIC，
  与 ``clock.mono_ms()`` 同基准），取不到再退化为出队时刻，便于上层
  执行 ``max_age_ms`` 时效判断。
* **超时**：``grab(timeout_ms)`` 默认 10s（T09）。超时会拆掉流以便下次
  重新打开，并抛 ``CameraUnavailable``，消息里带设备列表与支持格式。
* **线程安全**：公开方法内部加锁（接口约定允许在 asyncio 线程调用）。
"""

from __future__ import annotations

import ctypes
import dataclasses
import errno
import fcntl
import glob
import mmap
import os
import select
import threading
from dataclasses import dataclass

import numpy as np

from aieyes.clock import mono_ms
from aieyes.config import VisionConfig
from aieyes.errors import CameraUnavailable
from aieyes.interfaces import ImageFrame

from . import image_ops

# ---------------------------------------------------------------------------
# ioctl 编号（linux/videodev2.h）
# ---------------------------------------------------------------------------
_IOC_NRBITS, _IOC_TYPEBITS, _IOC_SIZEBITS, _IOC_DIRBITS = 8, 8, 14, 2
_IOC_NRSHIFT, _IOC_TYPESHIFT, _IOC_SIZESHIFT, _IOC_DIRSHIFT = 0, 8, 16, 30
_IOC_NONE, _IOC_WRITE, _IOC_READ = 0, 1, 2


def _IOC(direction: int, type_: str, nr: int, size: int) -> int:
    return (
        (direction << _IOC_DIRSHIFT)
        | (ord(type_) << _IOC_TYPESHIFT)
        | (nr << _IOC_NRSHIFT)
        | (size << _IOC_SIZESHIFT)
    )


def _IOR(type_: str, nr: int, size: int) -> int:
    return _IOC(_IOC_READ, type_, nr, size)


def _IOW(type_: str, nr: int, size: int) -> int:
    return _IOC(_IOC_WRITE, type_, nr, size)


def _IOWR(type_: str, nr: int, size: int) -> int:
    return _IOC(_IOC_READ | _IOC_WRITE, type_, nr, size)


def _fourcc(text: str) -> int:
    b = text.encode("ascii")
    if len(b) != 4:
        raise ValueError(f"fourcc must be 4 chars: {text!r}")
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)


def _fourcc_str(value: int) -> str:
    return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4))


# 像素格式
FMT_YUYV = _fourcc("YUYV")
FMT_MJPG = _fourcc("MJPG")
FMT_UYVY = _fourcc("UYVY")  # 仅探测/诊断用，当前不取图
FMT_NV12 = _fourcc("NV12")
FMT_Z16 = _fourcc("Z16 ")
FMT_GREY = _fourcc("GREY")

#: 当前可解码的格式
SUPPORTED_FOURCC = {"YUYV": FMT_YUYV, "MJPG": FMT_MJPG}

# 缓冲类型/内存类型/能力位
V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
V4L2_MEMORY_MMAP = 1
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_DEVICE_CAPS = 0x80000000
V4L2_FIELD_NONE = 1

# 帧时间戳来源
V4L2_BUF_FLAG_TIMESTAMP_MASK = 0x0000E000
V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC = 0x00002000


# ---------------------------------------------------------------------------
# ctypes 结构体（大小必须与内核 ABI 一致，注释里给出 64-bit 字节数）
# ---------------------------------------------------------------------------
class _Timeval(ctypes.Structure):
    _fields_ = [("tv_sec", ctypes.c_long), ("tv_usec", ctypes.c_long)]  # 16


class _Timecode(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("frames", ctypes.c_uint8),
        ("seconds", ctypes.c_uint8),
        ("minutes", ctypes.c_uint8),
        ("hours", ctypes.c_uint8),
        ("userbits", ctypes.c_uint8 * 4),
    ]  # 16


class _Capability(ctypes.Structure):
    _fields_ = [
        ("driver", ctypes.c_char * 16),
        ("card", ctypes.c_char * 32),
        ("bus_info", ctypes.c_char * 32),
        ("version", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("device_caps", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]  # 104


class _FmtDesc(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("description", ctypes.c_char * 32),
        ("pixelformat", ctypes.c_uint32),
        ("mbus_code", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 3),
    ]  # 64


class _PixFormat(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pixelformat", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("bytesperline", ctypes.c_uint32),
        ("sizeimage", ctypes.c_uint32),
        ("colorspace", ctypes.c_uint32),
        ("priv", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("ycbcr_enc", ctypes.c_uint32),
        ("quantization", ctypes.c_uint32),
        ("xfer_func", ctypes.c_uint32),
    ]  # 48


class _Format(ctypes.Structure):
    # 内核 union 因 v4l2_window 里的指针而 8 字节对齐：大小 208（64-bit）
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("_pad", ctypes.c_uint32),
        ("pix", _PixFormat),
        ("_reserved", ctypes.c_uint8 * 152),
    ]


class _FrmSizeDiscrete(ctypes.Structure):
    _fields_ = [("width", ctypes.c_uint32), ("height", ctypes.c_uint32)]


class _FrmSizeStepwise(ctypes.Structure):
    _fields_ = [
        ("min_width", ctypes.c_uint32),
        ("max_width", ctypes.c_uint32),
        ("step_width", ctypes.c_uint32),
        ("min_height", ctypes.c_uint32),
        ("max_height", ctypes.c_uint32),
        ("step_height", ctypes.c_uint32),
    ]


class _FrmSizeUnion(ctypes.Union):
    _fields_ = [("discrete", _FrmSizeDiscrete), ("stepwise", _FrmSizeStepwise)]


class _FrmSizeEnum(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("pixel_format", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("size", _FrmSizeUnion),
        ("reserved", ctypes.c_uint32 * 2),
    ]  # 44


class _RequestBuffers(ctypes.Structure):
    _fields_ = [
        ("count", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("capabilities", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 1),
    ]  # 20


class _Buffer(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("type", ctypes.c_uint32),
        ("bytesused", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("field", ctypes.c_uint32),
        ("timestamp", _Timeval),
        ("timecode", _Timecode),
        ("sequence", ctypes.c_uint32),
        ("memory", ctypes.c_uint32),
        ("m", ctypes.c_uint64),
        ("length", ctypes.c_uint32),
        ("reserved2", ctypes.c_uint32),
        ("request_fd", ctypes.c_int32),
    ]  # 88


class _CaptureParm(ctypes.Structure):
    _fields_ = [
        ("capability", ctypes.c_uint32),
        ("capturemode", ctypes.c_uint32),
        ("timeperframe_num", ctypes.c_uint32),
        ("timeperframe_denom", ctypes.c_uint32),
        ("extendedmode", ctypes.c_uint32),
        ("readbuffers", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 4),
    ]  # 40


class _StreamParm(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_uint32),
        ("capture", _CaptureParm),
        ("_reserved", ctypes.c_uint8 * 160),
    ]  # 204


VIDIOC_QUERYCAP = _IOR("V", 0, ctypes.sizeof(_Capability))
VIDIOC_ENUM_FMT = _IOWR("V", 2, ctypes.sizeof(_FmtDesc))
VIDIOC_G_FMT = _IOWR("V", 4, ctypes.sizeof(_Format))
VIDIOC_S_FMT = _IOWR("V", 5, ctypes.sizeof(_Format))
VIDIOC_REQBUFS = _IOWR("V", 8, ctypes.sizeof(_RequestBuffers))
VIDIOC_QUERYBUF = _IOWR("V", 9, ctypes.sizeof(_Buffer))
VIDIOC_QBUF = _IOWR("V", 15, ctypes.sizeof(_Buffer))
VIDIOC_DQBUF = _IOWR("V", 17, ctypes.sizeof(_Buffer))
VIDIOC_STREAMON = _IOW("V", 18, ctypes.sizeof(ctypes.c_int))
VIDIOC_STREAMOFF = _IOW("V", 19, ctypes.sizeof(ctypes.c_int))
VIDIOC_G_PARM = _IOWR("V", 21, ctypes.sizeof(_StreamParm))
VIDIOC_S_PARM = _IOWR("V", 22, ctypes.sizeof(_StreamParm))
VIDIOC_ENUM_FRAMESIZES = _IOWR("V", 74, ctypes.sizeof(_FrmSizeEnum))

#: 支持的像素格式名（用于配置校验/诊断）
PIXEL_FORMAT_CHOICES = ("auto", "yuyv", "mjpeg")


def _ioctl(fd: int, request: int, arg) -> None:
    """对 ctypes 结构体执行 ioctl；失败抛 ``OSError``（保留 errno）。"""
    try:
        fcntl.ioctl(fd, request, arg)
    except OSError:
        raise


# ---------------------------------------------------------------------------
# 设备探测
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FormatInfo:
    fourcc: str
    pixel_format: int
    sizes: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class VideoDeviceInfo:
    path: str
    card: str = ""
    driver: str = ""
    is_capture: bool = False
    formats: tuple[FormatInfo, ...] = ()
    error: str = ""

    @property
    def color_formats(self) -> tuple[FormatInfo, ...]:
        return tuple(f for f in self.formats if f.fourcc in SUPPORTED_FOURCC)

    def describe(self) -> str:
        if self.error:
            return f"{self.path} ERROR({self.error})"
        fmts = ", ".join(
            f"{f.fourcc}[{'/'.join(f'{w}x{h}' for w, h in f.sizes[:6])}"
            f"{'...' if len(f.sizes) > 6 else ''}]"
            for f in self.formats
        )
        return f"{self.path} card={self.card!r} formats={{{fmts}}}"


def _querycap(fd: int) -> _Capability:
    cap = _Capability()
    _ioctl(fd, VIDIOC_QUERYCAP, cap)
    return cap


def _enum_formats(fd: int) -> tuple[FormatInfo, ...]:
    formats: list[FormatInfo] = []
    index = 0
    while index < 32:
        desc = _FmtDesc(index=index, type=V4L2_BUF_TYPE_VIDEO_CAPTURE)
        try:
            _ioctl(fd, VIDIOC_ENUM_FMT, desc)
        except OSError as exc:
            if exc.errno in (errno.EINVAL, errno.ENOTTY):
                break
            raise
        fourcc = _fourcc_str(desc.pixelformat)
        sizes = _enum_sizes(fd, fourcc, desc.pixelformat)
        formats.append(FormatInfo(fourcc=fourcc, pixel_format=desc.pixelformat, sizes=sizes))
        index += 1
    return tuple(formats)


def _enum_sizes(
    fd: int, fourcc: str, pixel_format: int
) -> tuple[tuple[int, int], ...]:
    sizes: list[tuple[int, int]] = []
    index = 0
    while index < 64:
        fs = _FrmSizeEnum(index=index, pixel_format=pixel_format)
        try:
            _ioctl(fd, VIDIOC_ENUM_FRAMESIZES, fs)
        except OSError as exc:
            if exc.errno in (errno.EINVAL, errno.ENOTTY):
                break
            raise
        if fs.type != 1:  # V4L2_FRMSIZE_TYPE_DISCRETE
            break
        sizes.append((int(fs.size.discrete.width), int(fs.size.discrete.height)))
        index += 1
    return tuple(sizes)


def _probe_device(path: str) -> VideoDeviceInfo:
    fd = -1
    try:
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        cap = _querycap(fd)
        card = bytes(cap.card).split(b"\x00", 1)[0].decode("utf-8", "replace")
        driver = bytes(cap.driver).split(b"\x00", 1)[0].decode("utf-8", "replace")
        caps = cap.device_caps if cap.capabilities & V4L2_CAP_DEVICE_CAPS else cap.capabilities
        is_capture = bool(caps & V4L2_CAP_VIDEO_CAPTURE)
        formats = _enum_formats(fd) if is_capture else ()
        return VideoDeviceInfo(
            path=path,
            card=card,
            driver=driver,
            is_capture=is_capture,
            formats=formats,
        )
    except OSError as exc:
        return VideoDeviceInfo(path=path, error=f"{exc.strerror or exc} (errno={exc.errno})")
    finally:
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass


def probe_video_devices(paths: list[str] | None = None) -> list[VideoDeviceInfo]:
    """枚举 ``/dev/video*`` 的能力与支持的格式/分辨率（不启动取流）。"""
    if paths is None:
        paths = sorted(glob.glob("/dev/video*"))
    return [_probe_device(p) for p in paths]


# ---------------------------------------------------------------------------
# 选设备 / 选格式
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class _Candidate:
    device: VideoDeviceInfo
    format: FormatInfo
    size: tuple[int, int]
    device_rank: int
    format_rank: int
    size_rank: int

    @property
    def key(self) -> tuple[int, int, int]:
        return (self.device_rank, self.format_rank, self.size_rank)


def _pick_size(
    sizes: tuple[tuple[int, int], ...], want_w: int, want_h: int
) -> tuple[tuple[int, int], int]:
    def score(size: tuple[int, int]) -> int:
        w, h = size
        return abs(w - want_w) + abs(h - want_h)

    # 优先精确匹配
    for size in sizes:
        if size == (want_w, want_h):
            return size, 0
    best = min(sizes, key=score)
    return best, 1 + score(best)


def select_capture_candidate(
    config: VisionConfig, infos: list[VideoDeviceInfo]
) -> list[_Candidate]:
    """按配置偏好给可用的 (设备, 格式, 分辨率) 排序；返回候选列表。"""
    want_w, want_h = int(config.width), int(config.height)
    fmt_pref = (config.pixel_format or "auto").lower()
    if fmt_pref not in PIXEL_FORMAT_CHOICES:
        raise CameraUnavailable(
            f"VisionConfig.pixel_format={config.pixel_format!r} unsupported; "
            f"choose from {PIXEL_FORMAT_CHOICES}"
        )
    requested = os.path.realpath(config.device) if config.device else ""
    candidates: list[_Candidate] = []
    for info in infos:
        if info.error or not info.is_capture:
            continue
        device_rank = 0 if (requested and os.path.realpath(info.path) == requested) else 1
        for fmt in info.color_formats:
            if fmt_pref == "yuyv" and fmt.fourcc != "YUYV":
                continue
            if fmt_pref == "mjpeg" and fmt.fourcc != "MJPG":
                continue
            # 自动模式：YUYV 优先（免解码、时延低），MJPG 次之
            format_rank = 0 if fmt.fourcc == "YUYV" else 1
            if fmt_pref != "auto":
                format_rank = 0
            if not fmt.sizes:
                continue
            size, size_rank = _pick_size(fmt.sizes, want_w, want_h)
            candidates.append(
                _Candidate(
                    device=info,
                    format=fmt,
                    size=size,
                    device_rank=device_rank,
                    format_rank=format_rank,
                    size_rank=size_rank,
                )
            )
    candidates.sort(key=lambda c: c.key)
    return candidates


# ---------------------------------------------------------------------------
# 相机实现
# ---------------------------------------------------------------------------
class V4L2Camera:
    """V4L2 彩色相机（YUYV 直转 RGB / MJPG 用 Pillow 解码）。"""

    #: 内核缓冲数量（4 个够连续取图又不占太多内存）
    NUM_BUFFERS = 4

    def __init__(
        self,
        config: VisionConfig | None = None,
        *,
        device: str | None = None,
    ) -> None:
        self.config = config or VisionConfig()
        if device:
            self.config = dataclasses.replace(self.config, device=device)
        self._lock = threading.RLock()
        self._fd = -1
        self._streaming = False
        self._buffers: list[mmap.mmap] = []
        self._buf_lengths: list[int] = []
        self._buf_queued = 0
        self._seq = 0
        self._closed = True
        # 解析后的实际取流参数
        self.device_path = ""
        self.pixel_format = ""
        self.width = 0
        self.height = 0
        self.bytes_per_line = 0
        self.size_image = 0
        self.fps = 0.0
        self.last_error = ""
        self.open_ms = 0
        self.first_frame_ms = 0

    # --- Camera 协议 ---------------------------------------------------
    def open(self) -> None:
        with self._lock:
            if self._streaming:
                return
            infos = probe_video_devices()
            candidates = select_capture_candidate(self.config, infos)
            errors: list[str] = []
            for cand in candidates:
                try:
                    self._open_candidate(cand)
                    self._closed = False
                    self.last_error = ""
                    return
                except CameraUnavailable as exc:
                    errors.append(f"{cand.device.path}/{cand.format.fourcc}: {exc}")
                    self._teardown()
                except OSError as exc:
                    errors.append(
                        f"{cand.device.path}/{cand.format.fourcc}: "
                        f"{exc.strerror or exc} (errno={exc.errno})"
                    )
                    self._teardown()
            raise CameraUnavailable(self._diagnose("open failed", infos, errors))

    def grab(self, timeout_ms: int = 10_000) -> ImageFrame:
        with self._lock:
            if not self._streaming:
                self.open()
            timeout_ms = int(timeout_ms)
            if timeout_ms <= 0:
                raise CameraUnavailable("grab timeout_ms must be > 0")
            try:
                frame = self._grab_locked(timeout_ms)
            except CameraUnavailable:
                # 拆流，允许下一次 grab 自动重开（T14 设备恢复）
                self._teardown()
                raise
            self._closed = False
            return frame

    def close(self) -> None:
        with self._lock:
            self._teardown()
            self._closed = True

    # --- 内部：打开/配置 ----------------------------------------------
    def _open_candidate(self, cand: _Candidate) -> None:
        t0 = mono_ms()
        path = cand.device.path
        fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
        self._fd = fd
        width, height = cand.size
        # 1) S_FMT：设置像素格式与分辨率
        fmt = _Format(type=V4L2_BUF_TYPE_VIDEO_CAPTURE)
        fmt.pix.width = width
        fmt.pix.height = height
        fmt.pix.pixelformat = cand.format.pixel_format
        fmt.pix.field = V4L2_FIELD_NONE
        _ioctl(fd, VIDIOC_S_FMT, fmt)
        pix = fmt.pix
        actual_w, actual_h = int(pix.width), int(pix.height)
        fourcc = _fourcc_str(pix.pixelformat)
        if fourcc not in SUPPORTED_FOURCC:
            raise CameraUnavailable(
                f"driver picked unsupported format {fourcc!r} (want {cand.format.fourcc})"
            )
        if actual_w != width or actual_h != height:
            raise CameraUnavailable(
                f"driver adjusted size {width}x{height} -> {actual_w}x{actual_h}; "
                "refusing silent resize (would break frame identity)"
            )
        self.device_path = path
        self.pixel_format = fourcc
        self.width = actual_w
        self.height = actual_h
        self.bytes_per_line = int(pix.bytesperline) or actual_w * 2
        self.size_image = int(pix.sizeimage) or self.bytes_per_line * actual_h
        # 2) 帧率（不支持就算了，按驱动默认）
        self.fps = self._set_fps(fd, int(self.config.fps))
        # 3) 申请内核缓冲并 mmap
        req = _RequestBuffers(
            count=self.NUM_BUFFERS,
            type=V4L2_BUF_TYPE_VIDEO_CAPTURE,
            memory=V4L2_MEMORY_MMAP,
        )
        _ioctl(fd, VIDIOC_REQBUFS, req)
        if req.count < 1:
            raise CameraUnavailable(f"driver allocated no buffers (count={req.count})")
        self._buffers = []
        self._buf_lengths = []
        try:
            for index in range(req.count):
                buf = _Buffer(index=index, type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
                _ioctl(fd, VIDIOC_QUERYBUF, buf)
                length = int(buf.length) or self.size_image
                offset = int(buf.m)
                mm = mmap.mmap(
                    fd,
                    length,
                    mmap.MAP_SHARED,
                    mmap.PROT_READ | mmap.PROT_WRITE,
                    offset=offset,
                )
                self._buffers.append(mm)
                self._buf_lengths.append(length)
        except OSError as exc:
            raise CameraUnavailable(
                f"mmap buffers failed: {exc.strerror or exc} (errno={exc.errno}), "
                f"sizeimage={self.size_image}, bpl={self.bytes_per_line}"
            ) from exc
        # 4) 入队全部缓冲 + 开流
        for index in range(len(self._buffers)):
            self._qbuf(index)
        stream_on = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
        _ioctl(fd, VIDIOC_STREAMON, stream_on)
        self._streaming = True
        self.open_ms = mono_ms() - t0
        # 5) warmup：丢弃前 N 帧，等自动曝光/白平衡稳定
        warmup = int(self.config.warmup_frames or 0)
        warmup_deadline = mono_ms() + min(4000, max(1000, int(self.config.grab_timeout_ms)))
        first_t0 = mono_ms()
        try:
            for _ in range(warmup):
                remaining = warmup_deadline - mono_ms()
                if remaining <= 0:
                    raise CameraUnavailable("warmup exceeded 4s budget")
                buf, _ = self._dequeue_frame(remaining)
                self._qbuf(buf.index)
                if not self.first_frame_ms:
                    self.first_frame_ms = mono_ms() - first_t0
        except CameraUnavailable as exc:
            raise CameraUnavailable(
                f"warmup failed after {warmup} frame(s): {exc}; {self.describe()}"
            ) from exc

    def _set_fps(self, fd: int, fps: int) -> float:
        if fps <= 0:
            return 0.0
        parm = _StreamParm(type=V4L2_BUF_TYPE_VIDEO_CAPTURE)
        parm.capture.timeperframe_num = 1
        parm.capture.timeperframe_denom = fps
        try:
            _ioctl(fd, VIDIOC_S_PARM, parm)
            got = _StreamParm(type=V4L2_BUF_TYPE_VIDEO_CAPTURE)
            _ioctl(fd, VIDIOC_G_PARM, got)
            num = got.capture.timeperframe_num
            den = got.capture.timeperframe_denom
            return (den / num) if num else 0.0
        except OSError:
            return 0.0

    # --- 内部：取帧 ----------------------------------------------------
    def _grab_locked(self, timeout_ms: int) -> ImageFrame:
        buf, t_capture_ms = self._dequeue_frame(timeout_ms)
        try:
            rgb = self._decode(buf)
            return ImageFrame(
                rgb=rgb,
                t_capture_ms=t_capture_ms,
                frame_id=self._next_frame_id(t_capture_ms),
                source="camera",
            )
        finally:
            self._qbuf(buf.index)
            self._buf_queued += 1

    def _dequeue_frame(self, timeout_ms: int) -> tuple[_Buffer, int]:
        """poll 等待可读帧后非阻塞 DQBUF；超时抛 ``CameraUnavailable``。"""
        deadline = mono_ms() + timeout_ms
        poller = select.poll()
        poller.register(self._fd, select.POLLIN)
        while True:
            remaining = deadline - mono_ms()
            if remaining <= 0:
                raise CameraUnavailable(
                    f"grab timed out after {timeout_ms} ms; {self.describe()}"
                )
            events = poller.poll(int(remaining))
            if not events:
                continue
            buf = _Buffer(type=V4L2_BUF_TYPE_VIDEO_CAPTURE, memory=V4L2_MEMORY_MMAP)
            try:
                _ioctl(self._fd, VIDIOC_DQBUF, buf)
            except OSError as exc:
                if exc.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                    # 内核已可读但队列暂时空（多缓冲竞争），继续等
                    continue
                raise CameraUnavailable(
                    f"DQBUF failed: {exc.strerror or exc} (errno={exc.errno})"
                ) from exc
            self._seq += 1
            return buf, self._capture_ms(buf)

    def _capture_ms(self, buf: _Buffer) -> int:
        flags = int(buf.flags)
        if (flags & V4L2_BUF_FLAG_TIMESTAMP_MASK) == V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC:
            ts = buf.timestamp
            ms = int(ts.tv_sec) * 1000 + int(ts.tv_usec) // 1000
            # 与进程单调时钟同基准（CLOCK_MONOTONIC）；偏差过大说明驱动不可信
            if abs(ms - mono_ms()) <= 5000:
                return ms
        return mono_ms()

    def _decode(self, buf: _Buffer) -> np.ndarray:
        index = int(buf.index)
        if index < 0 or index >= len(self._buffers):
            raise CameraUnavailable(f"driver returned invalid buffer index {index}")
        mm = self._buffers[index]
        length = self._buf_lengths[index]
        bytesused = int(buf.bytesused) or length
        bytesused = min(bytesused, length)
        if self.pixel_format == "YUYV":
            return self._decode_yuyv(mm, length, bytesused)
        if self.pixel_format == "MJPG":
            data = bytes(mm[:bytesused])
            try:
                return image_ops.decode_image(data)
            except Exception as exc:
                raise CameraUnavailable(f"JPEG decode failed ({bytesused} bytes): {exc}")
        raise CameraUnavailable(f"unsupported pixel format {self.pixel_format}")

    def _decode_yuyv(self, mm: mmap.mmap, length: int, bytesused: int) -> np.ndarray:
        w, h, bpl = self.width, self.height, self.bytes_per_line
        need = w * h * 2
        if bytesused < need or length < need:
            raise CameraUnavailable(
                f"YUYV frame too small: bytesused={bytesused}, buffer={length}, "
                f"expect>={need} for {w}x{h}"
            )
        if bpl == w * 2:
            data = np.frombuffer(mm, dtype=np.uint8, count=need)
            return image_ops.yuyv_to_rgb(data, w, h)
        # bytesperline 有行尾填充：逐行取有效像素
        rows = np.frombuffer(mm, dtype=np.uint8, count=bpl * h).reshape(h, bpl)[:, : w * 2]
        return image_ops.yuyv_to_rgb(rows, w, h)

    def _qbuf(self, index: int) -> None:
        buf = _Buffer(
            index=int(index),
            type=V4L2_BUF_TYPE_VIDEO_CAPTURE,
            memory=V4L2_MEMORY_MMAP,
        )
        try:
            _ioctl(self._fd, VIDIOC_QBUF, buf)
        except OSError as exc:
            raise CameraUnavailable(
                f"QBUF buffer {index} failed: {exc.strerror or exc} (errno={exc.errno})"
            ) from exc

    def _next_frame_id(self, t_capture_ms: int) -> str:
        return f"cam-{t_capture_ms}-{self._seq}"

    # --- 内部：清理/诊断 ----------------------------------------------
    def _teardown(self) -> None:
        fd = self._fd
        if self._streaming and fd >= 0:
            try:
                stream_off = ctypes.c_int(V4L2_BUF_TYPE_VIDEO_CAPTURE)
                _ioctl(fd, VIDIOC_STREAMOFF, stream_off)
            except OSError:
                pass
        self._streaming = False
        for mm in self._buffers:
            try:
                mm.close()
            except (OSError, ValueError):
                pass
        self._buffers = []
        self._buf_lengths = []
        if fd >= 0:
            try:
                os.close(fd)
            except OSError:
                pass
        self._fd = -1

    def _diagnose(
        self, reason: str, infos: list[VideoDeviceInfo], errors: list[str]
    ) -> str:
        devices = "; ".join(info.describe() for info in infos) or "(no /dev/video*)"
        attempts = " | ".join(errors) if errors else "-"
        return (
            f"camera unavailable: {reason}; requested={self.config.device!r} "
            f"pixel_format={self.config.pixel_format!r} "
            f"size={self.config.width}x{self.config.height}; devices=[{devices}]; "
            f"attempts=[{attempts}]"
        )

    def describe(self) -> str:
        """当前实际取流参数（日志/报告用）。"""
        if not self.device_path:
            return "camera(not opened)"
        return (
            f"camera({self.device_path} {self.pixel_format} "
            f"{self.width}x{self.height}@{self.fps or '?'}fps "
            f"bpl={self.bytes_per_line} sizeimage={self.size_image})"
        )

    @property
    def is_open(self) -> bool:
        return self._streaming and self._fd >= 0

    # 兼容上下文管理器写法（测试/离线脚本方便）
    def __enter__(self) -> "V4L2Camera":
        self.open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
