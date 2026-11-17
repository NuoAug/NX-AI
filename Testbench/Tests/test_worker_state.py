from .test_server import ServerTestBase, FakeWorker

class WorkerStateTests(ServerTestBase):
    def test_exit_and_restart_state(self):
        self.ctx.state["services"]["worker"] = "alive"
        state = self.ctx.public_state()
        self.assertEqual(state["services"]["worker"], "none")
        worker = FakeWorker(self.ctx)
        self.ctx._worker = worker
        self.assertEqual(self.ctx.public_state()["services"]["worker"], "alive")
        self.ctx.stop_worker("idle_recycle")
        state = self.ctx.public_state()
        self.assertEqual(state["services"]["worker"], "none")
        self.assertIsNone(state["worker"]["pid"])
        self.ctx._worker = FakeWorker(self.ctx)
        self.assertEqual(self.ctx.public_state()["services"]["worker"], "alive")

    def test_completed_debug_schedules_idle_cleanup(self):
        from unittest.mock import patch
        self.ctx._debug_active = True
        self.ctx.state["debug"] = {"status": "running"}
        with patch.object(self.ctx, "_watch_worker_exit") as watch:
            self.ctx._finalize_debug("completed")
            watch.assert_called_once()
        self.assertFalse(self.ctx._debug_active)

import unittest
from types import SimpleNamespace
from unittest.mock import patch


class DeviceGuardTests(unittest.TestCase):
    """设备"核实失败"不得读成"被占用"；设备不可用必须分类并停在装配门口。

    真机路径：worker.make_camera() 里 `holders()` 在 fuser 缺失/超时时返回
    [{'pid': None, 'process': '占用状态无法核实'}]，旧代码把它当"非空即被占用"，
    于是既误报被占用、又把内部 dict 打进错误串。devices.py 已有诚实话术
    （UNVERIFIED_PROCESS / occupancy_unverifiable / occupancy_text / require_available）。
    """

    def _cfg(self, device: str = "/dev/video0"):
        return SimpleNamespace(vision=SimpleNamespace(device=device))

    def test_unverifiable_occupancy_is_not_reported_as_busy(self) -> None:
        import devices
        import worker as worker_mod

        choices = {"camera": [{"id": "/dev/video0"}], "microphone": [], "speaker": []}
        with patch.object(devices, "inventory", return_value=(choices, [])), patch.object(
            devices, "holders", return_value=[{"pid": None, "process": devices.UNVERIFIED_PROCESS}]
        ):
            with self.assertRaises(worker_mod.DeviceUnavailable) as caught:
                worker_mod.make_camera(self._cfg())
        text = str(caught.exception)
        self.assertIn("无法核实", text)  # 如实说"核实失败"
        self.assertNotIn("被占用", text)  # 不得读成"被占用"
        self.assertNotIn("'pid'", text)  # 不得把内部 dict 打进错误串
        self.assertNotIn("{", text)

    def test_real_owner_is_still_reported_as_busy(self) -> None:
        import devices
        import worker as worker_mod

        choices = {"camera": [{"id": "/dev/video0"}], "microphone": [], "speaker": []}
        with patch.object(devices, "inventory", return_value=(choices, [])), patch.object(
            devices, "holders", return_value=[{"pid": 4321, "process": "kworker"}]
        ):
            with self.assertRaises(worker_mod.DeviceUnavailable) as caught:
                worker_mod.make_camera(self._cfg())
        text = str(caught.exception)
        self.assertIn("被占用", text)
        self.assertIn("kworker", text)

    def test_camera_missing_from_inventory_is_a_device_error(self) -> None:
        """设备不在清单是设备类事实（E 的口径），不能报 engine_failed 让人以为是程序 bug。"""

        import devices
        import worker as worker_mod

        choices = {"camera": [{"id": "/dev/video9"}], "microphone": [], "speaker": []}
        with patch.object(devices, "inventory", return_value=(choices, [])):
            with self.assertRaises(worker_mod.DeviceUnavailable) as caught:
                worker_mod.make_camera(self._cfg("/dev/video0"))
        text = str(caught.exception)
        self.assertIn("所选摄像头不可用，请刷新设备列表；不会自动切换其他相机", text)
        self.assertEqual(
            worker_mod.classify_start_error(caught.exception).code, "device_unavailable"
        )

    def test_camera_not_selected_stays_an_engine_error(self) -> None:
        """未选设备是配置选择错误，保持 engine_failed —— 不要把配置问题也算成设备不可用。"""

        import worker as worker_mod

        with self.assertRaises(RuntimeError) as caught:
            worker_mod.make_camera(self._cfg(""))
        self.assertNotIsInstance(caught.exception, worker_mod.DeviceUnavailable)
        self.assertEqual(worker_mod.classify_start_error(caught.exception).code, "engine_failed")

    def test_unavailable_audio_device_stops_before_build_engine(self) -> None:
        import aieyes.app as app_mod
        import devices
        import worker as worker_mod

        built: list[str] = []

        def fake_build_engine(*args, **kwargs):
            built.append("build_engine")
            raise AssertionError("设备不可用时不得进入 build_engine")

        session = worker_mod.RunSession.__new__(worker_mod.RunSession)
        with patch.object(
            devices,
            "require_audio",
            side_effect=RuntimeError("麦克风未连接蓝牙或USB外接设备；不会使用板载空通道"),
        ), patch.object(app_mod, "build_engine", fake_build_engine):
            with self.assertRaises(worker_mod.DeviceUnavailable) as caught:
                session._assemble()
        self.assertEqual(built, [])
        self.assertIn("麦克风", str(caught.exception))

    def test_device_errors_are_classified_as_device_unavailable(self) -> None:
        import worker as worker_mod

        device_error = worker_mod.classify_start_error(
            worker_mod.DeviceUnavailable("摄像头占用无法核实（fuser 不可用或超时）")
        )
        self.assertEqual(device_error.code, "device_unavailable")
        self.assertIn("设备不可用", device_error.message)

        other = worker_mod.classify_start_error(RuntimeError("asr 初始化失败"))
        self.assertEqual(other.code, "engine_failed")
