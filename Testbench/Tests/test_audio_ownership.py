import json
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch
import devices

def node(i,cls):return {'id':i,'type':'PipeWire:Interface:Node','info':{'props':{'media.class':cls,'application.name':'test','application.process.id':123}}}
def link(input_node,output_node):return {'type':'PipeWire:Interface:Link','info':{'props':{'link.input.node':input_node,'link.output.node':output_node}}}
class AudioOwnershipTests(TestCase):
    def read(self,data):
        with patch('devices.subprocess.run',return_value=SimpleNamespace(stdout=json.dumps(data))):return dict(devices.pipewire_nodes())
    def test_monitor_of_sink_is_not_playback_ownership(self):
        result=self.read([node(1,'Audio/Sink'),node(2,'Stream/Input/Audio'),link(2,1)])
        self.assertFalse(result[1].get('_owners'))
    def test_playing_to_sink_is_ownership(self):
        result=self.read([node(1,'Audio/Sink'),node(2,'Stream/Output/Audio'),link(1,2)])
        self.assertEqual(result[1]['_owners'][0]['pid'],123)
    def test_recording_source_is_ownership(self):
        result=self.read([node(1,'Audio/Source'),node(2,'Stream/Input/Audio'),link(2,1)])
        self.assertEqual(result[1]['_owners'][0]['pid'],123)
