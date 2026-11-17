import json
from unittest import TestCase
from types import SimpleNamespace
from unittest.mock import patch
import devices

class PipeWireSnapshotTests(TestCase):
    def test_concatenated_snapshot_and_update(self):
        first=[{'id':1,'type':'PipeWire:Interface:Node','info':{'props':{'media.class':'Audio/Sink','node.name':'bluez_output.test'}}}]
        second=[{'id':1,'info':{'props':{'node.description':'TWS5'}}}]
        raw=json.dumps(first)+'\n'+json.dumps(second)+'\n[]\n'
        with self.assertRaises(json.JSONDecodeError):json.loads(raw)
        result=devices.parse_pipewire_dump(raw)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['info']['props']['node.name'],'bluez_output.test')
        self.assertEqual(result[0]['info']['props']['node.description'],'TWS5')
        with patch('devices.subprocess.run',return_value=SimpleNamespace(stdout=raw)):
            self.assertEqual(devices.pipewire_nodes()[0][1]['media.class'],'Audio/Sink')
    def test_removed_device_does_not_remain_online(self):
        raw='[{"id":7,"type":"PipeWire:Interface:Node","info":{"props":{}}}]\n[{"id":7,"info":null}]'
        self.assertEqual(devices.parse_pipewire_dump(raw),[])
    def test_corrupt_trailing_data_is_not_ignored(self):
        with self.assertRaises(json.JSONDecodeError):devices.parse_pipewire_dump('[]\nnot-json')
    def test_single_snapshot_and_whitespace(self):
        self.assertEqual(devices.parse_pipewire_dump(' \n [] \n'),[])
    def test_invalid_document_shape_rejected(self):
        with self.assertRaises(ValueError):devices.parse_pipewire_dump('{}')
