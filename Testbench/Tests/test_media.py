"""testbench/media.py 单测：全部使用 fake camera 与临时目录，不碰真实设备。"""

from __future__ import annotations

import io
import json
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT / "testbench"), str(_REPO_ROOT / "src")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from aieyes.interfaces import ImageFrame  # noqa: E402

from media import (  # noqa: E402
    CameraHub,
    CameraHubAdapter,
    FrameStore,
    UploadError,
    UploadStore,
    image_to_jpeg_bytes,
)


# ---------------------------------------------------------------------------
# 测试夹具
# ---------------------------------------------------------------------------
def make_frame(
    seq: int = 0,
    width: int = 64,
    height: int = 48,
    frame_id: str | None = None,
) -> ImageFrame:
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = seq % 256
    rgb[0, 0] = (10, 20, 30)
    return ImageFrame(
        rgb=rgb,
        t_capture_ms=1000 + seq,
        frame_id=frame_id if frame_id is not None else f"frame-{seq}",
        source="synthetic",
    )


class FakeCamera:
    """按 ``interval`` 产出新帧的假相机；可冻结（模拟无帧）或注入 grab 异常。"""

    def __init__(self, *, width: int = 64, height: int = 48, interval: float = 0.01):
        self.width = width
        self.height = height
        self.interval = interval
        self.open_count = 0
        self.close_count = 0
        self.fail_always = False
        self.fail_remaining = 0
        self.frozen = threading.Event()
        self._seq = 0
        self._lock = threading.Lock()

    def open(self) -> None:
        self.open_count += 1

    def close(self) -> None:
        self.close_count += 1

    def describe(self) -> str:
        return f"fake-camera {self.width}x{self.height}"

    def grab(self, timeout_ms: int = 10_000) -> ImageFrame:
        deadline = time.monotonic() + max(0, int(timeout_ms)) / 1000.0
        while self.frozen.is_set():
            if time.monotonic() >= deadline:
                raise TimeoutError("fake camera has no frame")
            time.sleep(0.005)
        if self.fail_always or self.fail_remaining > 0:
            if self.fail_remaining > 0:
                self.fail_remaining -= 1
            raise RuntimeError("fake-camera-grab-failed")
        time.sleep(max(0.0, min(self.interval, deadline - time.monotonic())))
        with self._lock:
            self._seq += 1
            seq = self._seq
        return make_frame(seq, self.width, self.height, frame_id=f"fake-{seq}")


def image_bytes(
    fmt: str = "JPEG",
    size: tuple[int, int] = (40, 30),
    color: tuple[int, int, int] = (180, 40, 40),
) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def wait_until(predicate, timeout_s: float = 3.0, interval_s: float = 0.01) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval_s)
    return predicate()


# ---------------------------------------------------------------------------
# CameraHub
# ---------------------------------------------------------------------------
class CameraHubTests(unittest.TestCase):
    def make_hub(self, camera: FakeCamera, **kwargs) -> CameraHub:
        hub = CameraHub(lambda: camera, clock=time.monotonic, log=lambda _m: None, **kwargs)
        self.addCleanup(hub.stop)
        return hub

    def test_first_frame_and_camera_lifecycle(self) -> None:
        camera = FakeCamera(interval=0.005)
        factory_calls: list[int] = []

        def factory() -> FakeCamera:
            factory_calls.append(1)
            return camera

        hub = CameraHub(factory, log=lambda _m: None)
        self.addCleanup(hub.stop)

        self.assertTrue(hub.status()["active"] is False)
        self.assertEqual(hub.add_ref("preview"), 1)
        self.assertEqual(hub.add_ref("preview"), 2)  # 同 reason 重复计数
        frame = hub.wait_newer_than(0, 3000)
        self.assertIsInstance(frame, ImageFrame)
        self.assertEqual((frame.width, frame.height), (64, 48))
        status = hub.status()
        self.assertTrue(status["active"])
        self.assertTrue(status["camera_open"])
        self.assertIsInstance(status["last_frame_ts"], int)
        self.assertIsNone(status["error"])
        self.assertEqual(camera.open_count, 1)

        self.assertEqual(hub.remove_ref("preview"), 1)
        self.assertTrue(hub.status()["active"])  # 还有引用
        self.assertEqual(hub.remove_ref("preview"), 0)
        self.assertFalse(hub.status()["active"])
        self.assertFalse(hub.status()["camera_open"])
        self.assertEqual(camera.close_count, 1)
        self.assertEqual(len(factory_calls), 1)  # 一台设备只创建一个实例

        hub.remove_ref("preview")  # 已归零，幂等
        self.assertFalse(hub.status()["active"])

        # 再次 add_ref 复用同一个相机实例，而不是另开设备
        hub.add_ref("task")
        hub.wait_newer_than(0, 3000)
        self.assertEqual(camera.open_count, 2)
        self.assertEqual(len(factory_calls), 1)
        hub.remove_ref("task")
        self.assertEqual(camera.close_count, 2)

    def test_latest_and_no_active_capture(self) -> None:
        camera = FakeCamera(interval=0.005)
        idle = self.make_hub(camera)
        self.assertIsNone(idle.latest())
        with self.assertRaises(TimeoutError):
            idle.wait_newer_than(0, 100)  # 未采集时立即失败，不忙等

        idle.add_ref("task")
        frame = idle.wait_newer_than(0, 3000)
        latest = idle.latest()
        self.assertIsNotNone(latest)
        assert latest is not None
        self.assertIs(latest[0], frame)
        idle.remove_ref("task")
        self.assertTrue(
            wait_until(lambda: idle.latest() is None, 2.0), "零引用停止后不应保留上一轮帧"
        )

    def test_wait_newer_than_returns_new_frame_and_times_out(self) -> None:
        camera = FakeCamera(interval=0.005)
        hub = self.make_hub(camera)
        hub.add_ref("task")
        first = hub.wait_newer_than(0, 3000)
        latest = hub.latest()
        assert latest is not None
        first_t = latest[1]

        newer = hub.wait_newer_than(first_t, 3000)
        self.assertNotEqual(newer.frame_id, first.frame_id)

        camera.frozen.set()
        time.sleep(0.05)  # 等在途 grab 结束并把最后一帧写入缓存
        frozen_t = hub.latest()[1]  # type: ignore[index]
        started = time.monotonic()
        with self.assertRaises(TimeoutError):
            hub.wait_newer_than(frozen_t, 150)
        elapsed = time.monotonic() - started
        self.assertGreaterEqual(elapsed, 0.1)
        self.assertLess(elapsed, 1.0)
        camera.frozen.clear()
        hub.remove_ref("task")

    def test_preview_callback_rate_limited_and_off_capture_thread(self) -> None:
        camera = FakeCamera(interval=0.002)
        received: list[tuple[bytes, str, int, str]] = []

        def on_preview(jpeg: bytes, frame_id: str, t_ms: int) -> None:
            received.append((jpeg, frame_id, t_ms, threading.current_thread().name))

        hub = self.make_hub(
            camera, preview_fps=5, jpeg_quality=70, preview_max_side=64, on_preview=on_preview
        )
        hub.add_ref("preview")
        time.sleep(1.15)
        hub.remove_ref("preview")
        self.assertGreaterEqual(len(received), 2)
        self.assertLessEqual(len(received), 7, "预览帧率不得超过 preview_fps 上限")
        jpeg, frame_id, t_ms, thread_name = received[0]
        self.assertEqual(jpeg[:2], b"\xff\xd8")
        self.assertTrue(frame_id)
        self.assertIsInstance(t_ms, int)
        self.assertEqual(thread_name, "camera-hub-preview")  # 不在采集线程回调
        with Image.open(io.BytesIO(jpeg)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertLessEqual(max(image.size), 64)

    def test_preview_stops_after_stop(self) -> None:
        camera = FakeCamera(interval=0.002)
        received: list[bytes] = []
        hub = self.make_hub(
            camera, preview_fps=10, on_preview=lambda jpeg, _f, _t: received.append(jpeg)
        )
        hub.add_ref("preview")
        self.assertTrue(wait_until(lambda: len(received) >= 1, 3.0))
        hub.stop()
        count = len(received)
        time.sleep(0.4)
        self.assertEqual(len(received), count)
        hub.stop()  # 幂等
        with self.assertRaises(RuntimeError):
            hub.add_ref("preview")

    def test_grab_error_recorded_and_recovered(self) -> None:
        camera = FakeCamera(interval=0.005)
        camera.fail_always = True
        hub = self.make_hub(camera)
        hub.add_ref("task")
        self.assertTrue(
            wait_until(lambda: hub.status()["error"] is not None, 2.0), hub.status()
        )
        self.assertTrue(hub.status()["active"])
        self.assertTrue(hub.status()["camera_open"])
        self.assertIn("grab", hub.status()["error"] or "")

        camera.fail_always = False
        frame = hub.wait_newer_than(0, 6000)
        self.assertEqual(frame.width, 64)
        self.assertIsNone(hub.status()["error"])
        hub.remove_ref("task")

    def test_camera_is_reopened_after_repeated_grab_failures(self) -> None:
        camera = FakeCamera(interval=0.005)
        camera.fail_always = True
        hub = self.make_hub(camera)
        hub.add_ref("task")
        # 连续失败到阈值必须释放已失效的句柄并重开，否则断连后画面永远回不来。
        self.assertTrue(
            wait_until(lambda: camera.close_count >= 1 and camera.open_count >= 2, 4.0),
            (camera.open_count, camera.close_count, hub.status()),
        )
        self.assertTrue(hub.status()["active"])
        camera.fail_always = False
        frame = hub.wait_newer_than(0, 6000)
        self.assertEqual(frame.width, 64)
        self.assertIsNone(hub.status()["error"])
        self.assertTrue(hub.status()["camera_open"])
        hub.remove_ref("task")
        self.assertGreaterEqual(camera.close_count, 1)

    def test_adapter_returns_progressive_new_frames(self) -> None:
        camera = FakeCamera(interval=0.005)
        hub = self.make_hub(camera)
        adapter = CameraHubAdapter(hub)
        adapter.open()
        first = adapter.grab(3000)
        second = adapter.grab(3000)
        self.assertIsInstance(first, ImageFrame)
        self.assertNotEqual(first.frame_id, second.frame_id)
        self.assertGreaterEqual(second.t_capture_ms, 1000)
        self.assertIn("fake-camera", adapter.describe())
        adapter.close()
        adapter.close()  # 幂等
        self.assertFalse(hub.status()["active"])
        self.assertEqual(camera.open_count, 1)
        self.assertEqual(camera.close_count, 1)


# ---------------------------------------------------------------------------
# FrameStore
# ---------------------------------------------------------------------------
class FrameStoreTests(unittest.TestCase):
    def test_save_writes_jpeg_under_media_dir(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FrameStore(directory)
            meta = store.save(make_frame(3, width=40, height=30, frame_id="cam-0003"))
            self.assertEqual(meta["file"], "media/cam-0003.jpg")
            self.assertEqual((meta["width"], meta["height"]), (40, 30))
            path = Path(directory) / meta["file"]
            self.assertTrue(path.is_file())
            with Image.open(path) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, (40, 30))
            # 缺省 frame_id 用捕获时间兜底
            fallback = store.save(make_frame(4, frame_id=""))
            self.assertTrue(fallback["file"].startswith("media/frame-"))

    def test_frame_id_is_sanitized_against_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = FrameStore(directory)
            media_root = Path(directory) / "media"
            for hostile in ("../../evil", "..\\..\\evil", "..", "/etc/passwd", "a" * 300):
                meta = store.save(make_frame(1, 8, 8, frame_id=hostile))
                relative = Path(meta["file"])
                self.assertEqual(relative.parts[0], "media")
                self.assertEqual(len(relative.parts), 2)
                self.assertNotIn("..", relative.parts)
                saved = Path(directory) / relative
                self.assertTrue(saved.is_file())
                self.assertEqual(saved.parent, media_root)
                self.assertLessEqual(len(saved.stem), 120)


# ---------------------------------------------------------------------------
# image_to_jpeg_bytes
# ---------------------------------------------------------------------------
class ImageEncodeTests(unittest.TestCase):
    def test_encode_and_downscale(self) -> None:
        rgb = np.zeros((48, 64, 3), dtype=np.uint8)
        rgb[:, :, 2] = 200
        data = image_to_jpeg_bytes(rgb, quality=70, max_side=32)
        self.assertEqual(data[:2], b"\xff\xd8")
        with Image.open(io.BytesIO(data)) as image:
            self.assertEqual(image.format, "JPEG")
            self.assertEqual(image.size, (32, 24))

        full = image_to_jpeg_bytes(rgb)
        with Image.open(io.BytesIO(full)) as image:
            self.assertEqual(image.size, (64, 48))


# ---------------------------------------------------------------------------
# UploadStore
# ---------------------------------------------------------------------------
class UploadStoreTests(unittest.TestCase):
    def make_store(self, directory: str, **kwargs) -> UploadStore:
        return UploadStore(directory, **kwargs)

    def test_accepts_jpeg_png_webp_and_normalizes_to_jpeg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            jpeg_meta = store.save(image_bytes("JPEG", (40, 30)), "A")
            self.assertEqual(set(jpeg_meta), {"image_id", "slot", "width", "height", "format"})
            self.assertEqual(jpeg_meta["format"], "jpeg")
            self.assertEqual((jpeg_meta["width"], jpeg_meta["height"]), (40, 30))
            self.assertRegex(jpeg_meta["image_id"], r"^[0-9a-f]{32}$")

            png_meta = store.save(image_bytes("PNG", (50, 20)), "A")
            webp_meta = store.save(image_bytes("WEBP", (12, 10)), "B")
            self.assertEqual((png_meta["width"], png_meta["height"]), (50, 20))
            self.assertEqual((webp_meta["width"], webp_meta["height"]), (12, 10))

            slots = store.slots()
            self.assertEqual(slots["A"]["image_id"], png_meta["image_id"])  # type: ignore[index]
            self.assertEqual(slots["B"]["image_id"], webp_meta["image_id"])  # type: ignore[index]

            for meta in (png_meta, webp_meta):
                path, media_type = store.get(meta["image_id"])
                self.assertEqual(media_type, "image/jpeg")
                self.assertTrue(path.is_file())
                self.assertEqual(path.name, f"{meta['image_id']}.jpg")
                with Image.open(path) as image:
                    self.assertEqual(image.format, "JPEG")
                    self.assertEqual(image.size, (meta["width"], meta["height"]))
            # 被覆盖的槽位旧图已回收
            with self.assertRaises(KeyError):
                store.get(jpeg_meta["image_id"])

    def test_rejects_oversized_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory, max_bytes=1024)
            with self.assertRaises(UploadError) as context:
                store.save(b"x" * 2048, "A")
            self.assertEqual(context.exception.code, 413)
            self.assertEqual(list(Path(directory).glob("*.jpg")), [])

    def test_rejects_too_many_pixels_and_accepts_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory, max_pixels=100)
            with self.assertRaises(UploadError) as context:
                store.save(image_bytes("JPEG", (20, 20)), "A")
            self.assertEqual(context.exception.code, 413)
            accepted = store.save(image_bytes("JPEG", (10, 10)), "A")  # 100 px，不超限
            self.assertEqual((accepted["width"], accepted["height"]), (10, 10))

    def test_rejects_fake_images_and_svg(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            svg = b'<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10"></svg>'
            truncated = image_bytes("JPEG", (40, 30))[:40]
            payloads = [
                b"plain text pretending to be a png",
                svg,
                truncated,
                image_bytes("GIF", (10, 10)),  # 合法但不是允许格式
                image_bytes("BMP", (10, 10)),
            ]
            for payload in payloads:
                with self.assertRaises(UploadError, msg=str(payload[:16])) as context:
                    store.save(payload, "A")
                self.assertEqual(context.exception.code, 415)
            with self.assertRaises(ValueError):
                store.save(image_bytes("JPEG", (10, 10)), "C")
            self.assertEqual(list(Path(directory).glob("*.jpg")), [])
            self.assertEqual(store.slots(), {"A": None, "B": None})

    def test_exif_orientation_is_normalized(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            source = Image.new("RGB", (60, 30), (10, 200, 10))
            exif = Image.Exif()
            exif[274] = 6  # 需要顺时针旋转 90°，显示尺寸 30x60
            buffer = io.BytesIO()
            source.save(buffer, format="JPEG", exif=exif)
            raw = buffer.getvalue()
            with Image.open(io.BytesIO(raw)) as probe:
                self.assertEqual(probe.size, (60, 30))

            meta = store.save(raw, "B")
            self.assertEqual((meta["width"], meta["height"]), (30, 60))
            path, _ = store.get(meta["image_id"])
            with Image.open(path) as stored:
                self.assertEqual(stored.size, (30, 60))
                self.assertIn(stored.getexif().get(274), (None, 1))  # 方向已烘焙进像素

    def test_delete_with_in_use_is_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            meta = store.save(image_bytes("JPEG"), "A")
            image_id = meta["image_id"]
            path, _ = store.get(image_id)
            self.assertTrue(store.set_in_use(image_id, True))

            self.assertTrue(store.delete(image_id))
            self.assertTrue(path.is_file())  # 仍被引用，不删文件
            self.assertTrue(store.is_pending(image_id))
            self.assertIsNone(store.slots()["A"])
            self.assertEqual(store.get(image_id)[0], path)  # 待回收期间仍可读

            self.assertTrue(store.set_in_use(image_id, False))
            self.assertFalse(path.is_file())
            self.assertFalse(store.is_pending(image_id))
            with self.assertRaises(KeyError):
                store.get(image_id)
            self.assertFalse(store.delete(image_id))
            self.assertFalse(store.set_in_use(image_id, True))

    def test_delete_without_use_removes_file_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            meta = store.save(image_bytes("JPEG"), "B")
            path, _ = store.get(meta["image_id"])
            self.assertTrue(store.delete(meta["image_id"]))
            self.assertFalse(path.is_file())
            self.assertEqual(store.slots(), {"A": None, "B": None})
            self.assertFalse(store.is_pending(meta["image_id"]))

    def test_index_persists_across_instances(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_store(directory)
            a_meta = first.save(image_bytes("JPEG", (40, 30)), "A")

            second = self.make_store(directory)
            slots = second.slots()
            self.assertEqual(slots["A"]["image_id"], a_meta["image_id"])  # type: ignore[index]
            self.assertEqual(
                (slots["A"]["width"], slots["A"]["height"]), (40, 30)  # type: ignore[index]
            )
            self.assertTrue(second.get(a_meta["image_id"])[0].is_file())

            b_meta = second.save(image_bytes("PNG", (20, 20)), "B")
            third = self.make_store(directory)
            self.assertEqual(third.slots()["A"]["image_id"], a_meta["image_id"])  # type: ignore[index]
            self.assertEqual(third.slots()["B"]["image_id"], b_meta["image_id"])  # type: ignore[index]

    def test_corrupt_index_rebuilds_from_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_store(directory)
            meta = first.save(image_bytes("JPEG"), "A")
            Path(directory, "index.json").write_text("{not-json", encoding="utf-8")

            rebuilt = self.make_store(directory)
            path, _ = rebuilt.get(meta["image_id"])  # 文件仍在，按实际文件收养
            self.assertTrue(path.is_file())
            self.assertIsNone(rebuilt.slots()["A"])  # 槽位归属不可恢复

    def test_missing_file_is_dropped_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_store(directory)
            meta = first.save(image_bytes("JPEG"), "A")
            (Path(directory) / f"{meta['image_id']}.jpg").unlink()

            reloaded = self.make_store(directory)
            self.assertIsNone(reloaded.slots()["A"])
            with self.assertRaises(KeyError):
                reloaded.get(meta["image_id"])

    def test_pending_reclaimed_on_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = self.make_store(directory)
            meta = first.save(image_bytes("JPEG"), "A")
            path, _ = first.get(meta["image_id"])
            first.set_in_use(meta["image_id"], True)
            first.delete(meta["image_id"])
            self.assertTrue(path.is_file())

            reloaded = self.make_store(directory)  # 进程重启时 in_use 不再成立
            self.assertFalse(path.is_file())
            with self.assertRaises(KeyError):
                reloaded.get(meta["image_id"])

    def test_index_file_is_valid_json(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = self.make_store(directory)
            meta = store.save(image_bytes("JPEG"), "A")
            payload = json.loads(Path(directory, "index.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["version"], 1)
            self.assertEqual(payload["slots"]["A"], meta["image_id"])
            self.assertIn(meta["image_id"], payload["items"])


if __name__ == "__main__":
    unittest.main()
