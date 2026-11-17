from unittest.mock import Mock
import pytest
import csi_camera


def node(root, name, label):
    p = root / name
    p.mkdir()
    (p / 'name').write_text(label)


def test_cam0_mapping_follows_sensor_identity_not_video_number(tmp_path, monkeypatch):
    node(tmp_path, 'video0', 'Intel RealSense')
    node(tmp_path, 'video7', 'vi-output, imx708 10-001a')
    monkeypatch.setattr(csi_camera, '_SYSFS', tmp_path)
    assert csi_camera._sensor_id('/dev/video7') == 0
    with pytest.raises(RuntimeError, match='不是已连接的IMX708'):
        csi_camera._sensor_id('/dev/video0')
    node(tmp_path, 'video8', 'vi-output, imx219 9-0010')
    with pytest.raises(RuntimeError, match='映射尚不能确认'):
        csi_camera._sensor_id('/dev/video7')


def test_argus_pipeline_and_owned_child_cleanup(tmp_path, monkeypatch):
    import devices
    monkeypatch.setattr(csi_camera, '_sensor_id', lambda device: 0)
    monkeypatch.setattr(devices, 'holders', lambda device: [])
    child = Mock()
    child.poll.return_value = None
    launch = Mock(return_value=child)
    monkeypatch.setattr(csi_camera.subprocess, 'Popen', launch)
    monkeypatch.setattr(csi_camera.CsiCamera, '_read_frame', lambda self, timeout_ms: 'frame-0')
    camera = csi_camera.CsiCamera('/dev/video7')
    camera.open()
    assert camera._pending == 'frame-0'
    args = launch.call_args.args[0]
    assert 'nvarguscamerasrc' in args and 'sensor-id=0' in args
    assert 'v4l2src' not in args and 'bayer2rgb' not in args
    assert 'video/x-raw,format=RGB' in args
    camera.close()
    child.terminate.assert_called_once()
    child.wait.assert_called_once_with(timeout=3)
    child.kill.assert_not_called()
    assert camera.proc is None


def test_open_primes_first_frame_so_hub_one_second_grab_succeeds(monkeypatch):
    import devices
    monkeypatch.setattr(csi_camera, '_sensor_id', lambda device: 0)
    monkeypatch.setattr(devices, 'holders', lambda device: [])
    child = Mock()
    child.poll.return_value = None
    monkeypatch.setattr(csi_camera.subprocess, 'Popen', Mock(return_value=child))
    calls = []
    monkeypatch.setattr(csi_camera.CsiCamera, '_read_frame',
                        lambda self, timeout_ms: calls.append(timeout_ms) or 'first')
    camera = csi_camera.CsiCamera('/dev/video7')
    camera.open()
    assert calls == [csi_camera._STARTUP_TIMEOUT_MS]
    assert camera.grab(timeout_ms=1000) == 'first'
    assert camera._pending is None
    assert calls == [csi_camera._STARTUP_TIMEOUT_MS]
    camera.close()


def test_busy_camera_is_not_opened(monkeypatch):
    import devices
    monkeypatch.setattr(csi_camera, '_sensor_id', lambda device: 0)
    monkeypatch.setattr(devices, 'holders', lambda device: [{'pid': 77, 'process': 'teammate'}])
    launch = Mock()
    monkeypatch.setattr(csi_camera.subprocess, 'Popen', launch)
    with pytest.raises(RuntimeError, match='被占用'):
        csi_camera.CsiCamera('/dev/video7').open()
    launch.assert_not_called()
