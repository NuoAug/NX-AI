"""设备占用必须说真话：核实不了就说核实不了，不能显示成空闲，也不能显示成被占用。"""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import devices

SENTINEL = [{'pid': None, 'process': devices.UNVERIFIED_PROCESS}]


def _run_result(returncode=1, stdout=''):
    return SimpleNamespace(returncode=returncode, stdout=stdout, stderr='')


class OccupancyTextTests(TestCase):
    def test_sentinel_is_not_treated_as_free(self):
        self.assertTrue(devices.occupancy_unverifiable(SENTINEL))
        text = devices.occupancy_text(SENTINEL, '当前未发现占用')
        self.assertIn('占用无法核实', text)
        self.assertNotIn('当前未发现占用', text)
        self.assertFalse(text.startswith('被占用：'), text)

    def test_missing_pid_from_pipewire_is_still_a_real_owner(self):
        # PipeWire 没报 pid 不等于“核实失败”；这类占用必须照实说被占用。
        owners = [{'process': 'teammate-doubao', 'pid': None}]
        self.assertFalse(devices.occupancy_unverifiable(owners))
        self.assertIn('被占用：teammate-doubao', devices.occupancy_text(owners, '当前未发现占用'))

    def test_empty_owners_uses_the_given_free_text(self):
        self.assertEqual(devices.occupancy_text([], '未发现 PipeWire 应用占用'), '未发现 PipeWire 应用占用')

    def test_busy_owners_list_each_process(self):
        owners = [{'process': 'ffplay', 'pid': 123}, {'process': 'python', 'pid': 456}]
        self.assertEqual(devices.occupancy_text(owners, 'x'), '被占用：ffplay PID 123, python PID 456')


class RequireAvailableTests(TestCase):
    def test_unverifiable_occupancy_refuses_without_claiming_busy(self):
        with patch.object(devices, 'holders', return_value=SENTINEL):
            with self.assertRaises(RuntimeError) as caught:
                devices.require_available('/dev/video0')
        message = str(caught.exception)
        self.assertIn('摄像头占用无法核实', message)
        self.assertIn('本次不采集', message)
        # 不能把“核实失败”说成“被占用”，也不许把内部字典泄漏给人看。
        self.assertNotIn('被占用', message)
        self.assertNotIn('pid', message)

    def test_real_occupancy_is_reported_with_owner(self):
        with patch.object(devices, 'holders', return_value=[{'process': 'doubao', 'pid': 4321}]):
            with self.assertRaises(RuntimeError) as caught:
                devices.require_available('/dev/video0', label='麦克风')
        self.assertIn('麦克风被占用：doubao PID 4321', str(caught.exception))

    def test_free_device_passes(self):
        with patch.object(devices, 'holders', return_value=[]):
            self.assertIsNone(devices.require_available('/dev/video0'))


class InventoryHonestyTests(TestCase):
    def _empty_sysfs(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        return Path(tmp.name)

    def test_audio_service_down_does_not_also_claim_no_device(self):
        with patch.object(devices, 'VIDEO_SYSFS', self._empty_sysfs()), \
                patch.object(devices, 'pipewire_nodes', side_effect=OSError('pw-dump 缺失')), \
                patch.object(devices, 'audio_target', side_effect=OSError('wpctl 缺失')):
            result, errors = devices.inventory()
        self.assertTrue(any('音频服务不可用' in item for item in errors), errors)
        self.assertFalse(any('未发现在线蓝牙或USB外接设备' in item for item in errors), errors)
        self.assertNotEqual(result['microphone'][0]['occupancy'], '待核实')

    def test_reachable_audio_without_external_device_says_so(self):
        with patch.object(devices, 'VIDEO_SYSFS', self._empty_sysfs()), \
                patch.object(devices, 'pipewire_nodes', return_value=[]), \
                patch.object(devices, 'audio_target', return_value=None):
            _, errors = devices.inventory()
        self.assertIn('麦克风：未发现在线蓝牙或USB外接设备', errors)

    def test_pipewire_free_device_is_not_called_absent(self):
        nodes = [(1, {'node.name': 'bluez_input.headset', 'device.api': 'bluez5',
                      'media.class': 'Audio/Source', 'node.description': '耳机麦克风'})]
        with patch.object(devices, 'VIDEO_SYSFS', self._empty_sysfs()), \
                patch.object(devices, 'pipewire_nodes', return_value=nodes), \
                patch.object(devices, 'audio_target', return_value='bluez_input.headset'):
            result, errors = devices.inventory()
        microphone = result['microphone']
        self.assertEqual(microphone[1]['occupancy'], '未发现 PipeWire 应用占用')
        self.assertEqual(microphone[0]['occupancy'], '未发现 PipeWire 应用占用')
        self.assertFalse([item for item in errors if '麦克风' in item], errors)

    def test_pipewire_busy_device_shows_owner(self):
        nodes = [(1, {'node.name': 'bluez_input.headset', 'device.api': 'bluez5',
                      'media.class': 'Audio/Source', 'node.description': '耳机麦克风',
                      '_owners': [{'process': 'doubao', 'pid': 777}]})]
        with patch.object(devices, 'VIDEO_SYSFS', self._empty_sysfs()), \
                patch.object(devices, 'pipewire_nodes', return_value=nodes), \
                patch.object(devices, 'audio_target', return_value='bluez_input.headset'):
            result, _ = devices.inventory()
        self.assertEqual(result['microphone'][1]['occupancy'], '被占用：doubao PID 777')
        self.assertEqual(result['microphone'][0]['occupancy'], '被占用：doubao PID 777')

    def _sysfs_with_camera(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        (root / 'video0').mkdir()
        (root / 'video0' / 'name').write_text('imx708 10-001a\n')
        return root

    def test_camera_occupancy_unverifiable_is_not_reported_free(self):
        with patch.object(devices, 'VIDEO_SYSFS', self._sysfs_with_camera()), \
                patch.object(devices, 'pipewire_nodes', return_value=[]), \
                patch.object(devices, 'audio_target', return_value=None), \
                patch.object(devices, 'holders', return_value=SENTINEL), \
                patch.object(devices, 'subprocess') as sub:
            sub.run.return_value = _run_result()
            result, errors = devices.inventory()
        cameras = [item for item in result['camera'] if item['id']]
        self.assertEqual(len(cameras), 1, result['camera'])
        self.assertIn('占用无法核实', cameras[0]['occupancy'])
        self.assertNotIn('当前未发现占用', cameras[0]['occupancy'])
        self.assertFalse([item for item in errors if '摄像头枚举不完整' in item], errors)

    def test_camera_enumeration_failure_is_reported_not_hidden(self):
        with patch.object(devices, 'VIDEO_SYSFS', self._sysfs_with_camera()), \
                patch.object(devices, 'pipewire_nodes', return_value=[]), \
                patch.object(devices, 'audio_target', return_value=None), \
                patch.object(devices, 'subprocess') as sub:
            sub.run.side_effect = OSError('v4l2-ctl 缺失')
            result, errors = devices.inventory()
        self.assertIn('摄像头枚举不完整：v4l2-ctl 不可用，无法核实摄像头数量与占用', errors)
        self.assertEqual([item for item in result['camera'] if item['id']], [])


class RequireAudioTests(TestCase):
    """缺少 PipeWire 工具时，报错必须说“无法核实”，不能把工具缺失说成设备不存在。"""

    def test_audio_target_returns_none_when_wpctl_is_missing(self):
        with patch.object(devices, 'selection', return_value={'microphone': '', 'speaker': ''}), \
             patch.object(devices.subprocess, 'run', side_effect=FileNotFoundError('wpctl')):
            self.assertIsNone(devices.audio_target('microphone'))

    def test_require_audio_reports_unverifiable_instead_of_raw_traceback(self):
        with patch.object(devices, 'pipewire_nodes', side_effect=FileNotFoundError('pw-dump')):
            with self.assertRaises(RuntimeError) as ctx:
                devices.require_audio('microphone')
        message = str(ctx.exception)
        self.assertIn('音频服务不可用，无法核实麦克风', message)
        self.assertNotIn('未连接蓝牙或USB外接设备', message)
        self.assertNotIn('系统默认音频未指向在线外接设备', message)

    def test_require_audio_still_refuses_board_mic_when_service_is_up(self):
        with patch.object(devices, 'pipewire_nodes', return_value=[]), \
             patch.object(devices, 'selection', return_value={'microphone': '', 'speaker': ''}), \
             patch.object(devices.subprocess, 'run', return_value=_run_result(1, '')):
            with self.assertRaises(RuntimeError) as ctx:
                devices.require_audio('microphone')
        # 服务在线但没有外接设备：要么拒绝“系统默认”，要么拒绝“未连接”，都不能静默放行。
        message = str(ctx.exception)
        self.assertTrue(
            '未连接蓝牙或USB外接设备' in message or '系统默认音频未指向在线外接设备' in message,
            message,
        )
