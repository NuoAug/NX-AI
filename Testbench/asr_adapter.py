"""Finish the streaming Paraformer decoder without losing the final chunk."""
from dataclasses import replace
import numpy as np
from aieyes.clock import mono_ms
from aieyes.speech.asr import StreamingAsr as ProjectStreamingAsr


class StreamingAsr(ProjectStreamingAsr):
    def finalize(self):
        if self._stream is not None and self._t_end_ms is not None:
            # Sherpa's streaming Paraformer example pads 0.3 s before
            # input_finished. Feed it directly: it is decoder padding,
            # not captured audio, and must not advance audio timestamps.
            rate = int(self.cfg.sample_rate)
            self._stream.accept_waveform(rate, np.zeros(round(rate * .3), np.float32))
        result = super().finalize()
        return replace(result, t_final_ms=mono_ms())
