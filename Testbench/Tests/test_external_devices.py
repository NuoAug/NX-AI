from unittest import TestCase
from unittest.mock import patch
import devices

class ExternalTests(TestCase):
    def test_filter_external_audio(self):
        self.assertFalse(devices.external({'node.name':'alsa_output.platform-tegra','device.api':'alsa'}))
        self.assertTrue(devices.external({'node.name':'bluez_input.test','device.api':'bluez5'}))
        self.assertTrue(devices.external({'node.name':'alsa_output.usb-headset'}))
    def test_missing_external_defaults_are_rejected(self):
        with patch.object(devices,'selection',return_value={'microphone':'','speaker':'','camera':''}),patch.object(devices,'audio_target',return_value=None),patch.object(devices,'pipewire_nodes',return_value=[]),patch.object(devices,'inventory',return_value=({'microphone':[{'id':''}]},[])):
            with self.assertRaisesRegex(RuntimeError,'系统默认音频'):
                devices.require_audio('microphone')
    def test_busy_external_audio_is_not_stolen(self):
        with patch.object(devices,'audio_target',return_value='bluez_input.test'),patch.object(devices,'pipewire_nodes',return_value=[(1,{'node.name':'bluez_input.test','_owners':[{'pid':99999999,'process':'teammate'}]})]):
            with self.assertRaisesRegex(RuntimeError,'其他应用占用'):
                devices.audio_owner_check('microphone')
