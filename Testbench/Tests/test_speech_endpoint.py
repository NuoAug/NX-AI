import numpy as np
from unittest.mock import Mock,patch
from speech_vad import SpeechVad
from aieyes.audio.vad import EnergyVad
from aieyes.config import VadConfig
from aieyes.interfaces import AudioFrame

def test_endpoint_metadata_excludes_wait_but_audio_keeps_tail():
    vad=SpeechVad.__new__(SpeechVad)
    cfg=VadConfig(min_speech_ms=40,min_silence_ms=60)
    EnergyVad.__init__(vad,cfg,sample_rate=16000,frame_ms=20)
    vad.reset()
    vad._session=Mock();vad._session.run.return_value=(np.zeros((1,1)),np.zeros((2,1,128),np.float32))
    found=[]
    with patch.object(vad,'frame_level',side_effect=lambda f:EnergyVad.frame_level(f)):
        for i in range(13):
            samples=np.full(320,.5 if i<10 else 0.,np.float32)
            result=vad.accept(AudioFrame(samples=samples,t_start_ms=1000+i*20,seq=i))
            if result is not None:found.append(result)
    assert len(found)==1
    assert found[0].t_end_ms==1200
    assert len(found[0].samples)>=13*320
