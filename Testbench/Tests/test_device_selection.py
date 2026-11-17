import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from .test_server import ServerTestBase
import devices

class DeviceSelectionTests(ServerTestBase):
    def choices(self):
        return {'camera':[{'id':'','label':'default'},{'id':'/dev/video4','label':'camera'}], 'microphone':[{'id':'','label':'default'},{'id':'2','label':'mic'}], 'speaker':[{'id':'','label':'default'},{'id':'3','label':'speaker'}]}

    def test_defaults_and_selected_config(self):
        path = self.root / 'devices.json'
        cfg = SimpleNamespace(audio=SimpleNamespace(input_device=None,output_device=None), vision=SimpleNamespace(device='/dev/video0'))
        with patch.object(devices,'SELECTION',path), patch.object(devices,'inventory',return_value=(self.choices(),[])):
            devices.apply(cfg)
            self.assertEqual(cfg.audio.input_device, 'bench_microphone')
            devices.save({'camera':'/dev/video4','microphone':'2','speaker':'3'},self.choices())
            devices.apply(cfg)
            self.assertEqual((cfg.audio.input_device,cfg.audio.output_device,cfg.vision.device),(2,3,'/dev/video4'))
            with self.assertRaises(ValueError):
                devices.save({'camera':'/etc/passwd'},self.choices())

    def test_busy_settings_rejected(self):
        self.serve()
        self.ctx.heartbeat('owner')
        self.ctx._run_active = True
        try:
            status,_,_ = self.request('POST','/api/devices',body=b'{}',headers={'Origin':f'http://127.0.0.1:{self.ctx.port}','Content-Type':'application/json','X-Bench-Client':'owner'})
            self.assertEqual(status,409)
        finally:
            self.ctx._run_active = False

    def test_stale_entries_do_not_block_clearing_one_at_a_time(self):
        path = self.root / 'devices.json'
        path.write_text('{"camera":"/dev/gone","microphone":"pw:bluez_input.gone","speaker":"pw:bluez_output.gone"}')
        choices = {'camera':[{'id':'','label':'default'},{'id':'/dev/video4','label':'camera'}], 'microphone':[{'id':'','label':'default'}], 'speaker':[{'id':'','label':'default'}]}
        with patch.object(devices,'SELECTION',path):
            devices.save({'camera':'','microphone':'pw:bluez_input.gone','speaker':'pw:bluez_output.gone'}, choices)
            self.assertEqual(devices.selection()['camera'], '')
            devices.save({'camera':'','microphone':'','speaker':'pw:bluez_output.gone'}, choices)
            self.assertEqual(devices.selection()['microphone'], '')
            devices.save({'camera':'','microphone':'','speaker':''}, choices)
            self.assertEqual(devices.selection()['speaker'], '')
            with self.assertRaises(ValueError):
                devices.save({'camera':'/dev/gone2'}, choices)

    def test_idle_settings_saved_inside_testbench_config(self):
        self.serve()
        self.ctx.heartbeat('owner')
        with patch.object(devices,'SELECTION',self.root/'devices.json'),patch.object(devices,'inventory',return_value=(self.choices(),[])):
            status,_,_=self.request('POST','/api/devices',body=b'{"microphone":"2"}',headers={'Origin':f'http://127.0.0.1:{self.ctx.port}','Content-Type':'application/json','X-Bench-Client':'owner'})
            self.assertEqual(status,200)
            self.assertEqual(devices.selection()['microphone'],'2')
