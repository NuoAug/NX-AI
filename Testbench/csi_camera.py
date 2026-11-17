"""IMX708 CAM0 capture through NVIDIA Argus; never fall back to a USB camera."""
import os
import select
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np
from aieyes.interfaces import ImageFrame
from aieyes.clock import mono_ms

_SYSFS = Path('/sys/class/video4linux')
# Argus 首帧需要约 1—2 秒建立；上层 CameraHub 每次 grab 只等 1 秒，
# 因此 open() 必须先预热到首帧，否则会被反复判为取景超时并重启流水线。
_STARTUP_TIMEOUT_MS = 4000


def _sensor_id(device):
    selected = _SYSFS / Path(device).name / 'name'
    if not selected.is_file() or 'imx708' not in selected.read_text().lower():
        raise RuntimeError('所选设备不是已连接的IMX708；请连接CAM0摄像头后刷新，不会切换其他相机')
    sensors = [p for p in _SYSFS.glob('video*/name')
               if p.read_text().lower().startswith('vi-output')]
    if len(sensors) != 1:
        raise RuntimeError('CSI传感器映射尚不能确认，拒绝猜测Argus sensor-id')
    if '10-001a' not in selected.read_text().lower():
        raise RuntimeError('当前仅核验过J20/CAM0的IMX708映射，请先确认接口')
    return 0


class CsiCamera:
    def __init__(self, device='/dev/video0'):
        self.device_path = device
        self.proc = None
        self.width, self.height = 1280, 720
        self._pending = None

    def open(self):
        if self.proc is not None:
            return
        sensor = _sensor_id(self.device_path)
        from devices import require_available
        require_available(self.device_path)
        logs = Path(__file__).resolve().parent / 'logs'
        logs.mkdir(exist_ok=True)
        with (logs / 'csi.log').open('ab') as log:
            self.proc = subprocess.Popen([
                'gst-launch-1.0', '-q', 'nvarguscamerasrc', f'sensor-id={sensor}',
                '!', f'video/x-raw(memory:NVMM),format=NV12,width={self.width},height={self.height},framerate=15/1',
                '!', 'nvvidconv', '!', 'video/x-raw,format=BGRx',
                '!', 'videoconvert', '!', 'video/x-raw,format=RGB',
                '!', 'fdsink', 'fd=1', 'sync=false',
            ], stdout=subprocess.PIPE, stderr=log)
        try:
            self._pending = self._read_frame(_STARTUP_TIMEOUT_MS)
        except Exception:
            self.close()
            raise

    def _read_frame(self, timeout_ms):
        data = bytearray()
        expected = self.width * self.height * 3
        end = time.monotonic() + timeout_ms / 1000
        while len(data) < expected:
            remaining = end - time.monotonic()
            if remaining <= 0 or not select.select([self.proc.stdout], [], [], remaining)[0]:
                raise TimeoutError('我还没看到摄像头画面，取景超时了；请检查相机连接')
            chunk = os.read(self.proc.stdout.fileno(), expected - len(data))
            if not chunk:
                raise RuntimeError('我还没看到摄像头画面；Argus采集未成功，请查看logs/csi.log')
            data.extend(chunk)
        return ImageFrame(np.frombuffer(data, dtype=np.uint8).reshape(
            self.height, self.width, 3).copy(), mono_ms(), uuid.uuid4().hex, 'camera')

    def grab(self, timeout_ms=10000):
        self.open()
        try:
            if self._pending is not None:
                frame, self._pending = self._pending, None
                return frame
            return self._read_frame(timeout_ms)
        except Exception:
            self.close()
            raise

    def close(self):
        proc, self.proc = self.proc, None
        self._pending = None
        if proc is not None:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
            proc.stdout.close()
