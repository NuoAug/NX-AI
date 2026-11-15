"""音频 I/O：采集、回放、播放、AEC、VAD、WAV 工具。"""

from .aec import NlmsAec
from .capture import SoundDeviceSource
from .output import AudioSink
from .replay_source import ReplaySource, build_replay_playlist
from .vad import EnergyVad
from .wav import read_wav, write_wav, resample_linear, concat_segments, WavError

__all__ = [
    "AudioSink",
    "EnergyVad",
    "NlmsAec",
    "ReplaySource",
    "SoundDeviceSource",
    "WavError",
    "build_replay_playlist",
    "concat_segments",
    "read_wav",
    "resample_linear",
    "write_wav",
]

SAMPLE_RATE = 16000
