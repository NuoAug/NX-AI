from pathlib import Path
from unittest.mock import patch
import pytest
from aieyes.config import AppConfig
from aieyes.errors import ConfigError
from kws_adapter import configure_kws, SherpaKws


def assets(path):
    for name in ('encoder.int8.onnx','decoder.onnx','joiner.int8.onnx','tokens.txt'):
        (path/name).write_bytes(b'test fixture')


def test_default_profile_records_the_actual_model_and_parameters(tmp_path):
    assets(tmp_path)
    cfg=AppConfig()
    with patch('kws_adapter.PROFILE_DIR',tmp_path): configure_kws(cfg)
    assert cfg.models.kws_dir == str(tmp_path)
    assert cfg.kws.keywords_score == 1
    assert cfg.kws.num_trailing_blanks == 1
    assert cfg.kws.wake_phrases == ['小冰']
    assert cfg.kws.keywords_threshold == .25


def test_explicit_model_and_keyword_file_are_preserved(tmp_path):
    assets(tmp_path)
    cfg=AppConfig();cfg.models.kws_dir='custom-model'
    with patch('kws_adapter.PROFILE_DIR',tmp_path): configure_kws(cfg)
    assert cfg.models.kws_dir == 'custom-model'
    cfg=AppConfig();cfg.kws.keywords_file='custom-keywords.txt'
    with patch('kws_adapter.PROFILE_DIR',tmp_path): configure_kws(cfg)
    assert cfg.kws.keywords_file == 'custom-keywords.txt'
    assert cfg.kws.keywords_score == 3


def test_partial_model_install_is_reported(tmp_path):
    (tmp_path/'tokens.txt').write_text('x 0')
    with patch('kws_adapter.PROFILE_DIR',tmp_path), pytest.raises(ConfigError):
        configure_kws(AppConfig())


def test_explicit_tuning_is_preserved(tmp_path):
    assets(tmp_path);cfg=AppConfig();cfg.kws.keywords_score=2;cfg.kws.num_trailing_blanks=3
    with patch('kws_adapter.PROFILE_DIR',tmp_path): configure_kws(cfg)
    assert (cfg.kws.keywords_score,cfg.kws.num_trailing_blanks)==(2,3)


def test_warmup_uses_a_separate_stream_and_only_runs_once():
    class Stream:
        def __init__(self): self.samples=[]
        def accept_waveform(self,rate,samples): self.samples.append((rate,len(samples)))
    class Spotter:
        def __init__(self): self.streams=[];self.decoded=[]
        def create_stream(self):
            s=Stream();self.streams.append(s);return s
        def is_ready(self,s): return bool(s.samples) and s not in self.decoded
        def decode_stream(self,s): self.decoded.append(s)
    kws=SherpaKws();spotter=Spotter()
    def load(k):
        if k._spotter is None:
            k._spotter=spotter;k._stream=spotter.create_stream()
    with patch('kws_adapter.ProjectKws._ensure_loaded',load):
        kws._ensure_loaded();kws._ensure_loaded()
    assert len(spotter.streams)==2
    assert spotter.streams[0].samples==[]
    assert spotter.streams[1].samples==[(16000,16000)]
    assert spotter.decoded==[spotter.streams[1]]
