"""用已有 Silero 模型核实人声，保留原能量VAD的计时、前置音频和原始样本。"""
from pathlib import Path
from dataclasses import replace
from collections import OrderedDict
import numpy as np
from aieyes.audio.vad import EnergyVad

class SpeechVad(EnergyVad):
    def __init__(self,cfg=None,*,model_path=None,sample_rate=16000,frame_ms=20):
        if sample_rate!=16000:
            raise ValueError("SpeechVad requires 16 kHz input")
        super().__init__(cfg,sample_rate=sample_rate,frame_ms=frame_ms)
        import onnxruntime as ort
        path=Path(model_path) if model_path else Path(__file__).resolve().parent.parent/'models/vad/silero-vad-v6.2.3/silero_vad.onnx'
        opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        self._session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
        self.reset()

    def reset(self):
        super().reset()
        self._model_state=np.zeros((2,1,128),np.float32)
        self._context=np.zeros((1,64),np.float32)
        self._pending=np.zeros(0,np.float32)
        self._probability=0.
        self._last_voice_end=0
        self._scores=OrderedDict()

    def accept(self,frame):
        self._pending=np.concatenate((self._pending,np.asarray(frame.samples,dtype=np.float32)))
        while self._pending.size>=512:
            chunk=self._pending[:512].reshape(1,512);self._pending=self._pending[512:]
            data=np.concatenate((self._context,chunk),axis=1)
            output,self._model_state=self._session.run(None,{'input':data,'state':self._model_state,'sr':np.array(16000,np.int64)})
            self._probability=float(output[0,0]);self._context=chunk[:,-64:].copy()
        self._scores[frame.t_start_ms]=self._probability
        while len(self._scores)>2500:self._scores.popitem(last=False)
        if self.frame_level(frame)>=self.close_level:
            self._last_voice_end=int(frame.t_start_ms+round(len(frame.samples)*1000/self.sample_rate))
        segment=super().accept(frame)
        if segment is not None and segment.t_start_ms<=self._last_voice_end<=segment.t_end_ms:
            # 样本仍保留端点静音给ASR；计时终点必须是最后人声，不能扣掉静音等待。
            segment=replace(segment,t_end_ms=self._last_voice_end)
        return segment

    def frame_level(self,frame):
        probability=self._scores.get(frame.t_start_ms,self._probability)
        # 0.35 而不是 0.5：自然说话的轻声字、句尾常被 Silero 判成不确定；
        # 门限过硬会把句中静音拉长，触发 900ms 提前收尾，切出半句。
        return EnergyVad.frame_level(frame) if probability>=.35 else 0.
