"""证明唤醒配置/热词/阈值/profile 真的进到运行时 KeywordSpotter 参数。

只做配置生效性证据（清单第 6 项）：不喂真实 PCM、不计算唤醒率、不给出“近音误唤醒已解决”。
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from aieyes.config import AppConfig, KwsConfig, ModelsConfig
from aieyes.errors import ConfigError
from kws_adapter import SherpaKws, configure_kws
import kws_adapter

ROOT = Path(__file__).resolve().parents[2]
TARGET_CONFIG = ROOT / 'Config' / 'target.json'
if not TARGET_CONFIG.is_file():
    TARGET_CONFIG = ROOT / 'config' / 'target.json'


def assets(path):
    for name in ('encoder.int8.onnx', 'decoder.onnx', 'joiner.int8.onnx'):
        (path / name).write_bytes(b'test fixture')
    (path / 'tokens.txt').write_text('x 0\niǎo 1\nb 2\nīng 3\nn 4\nǐ 5\nh 6\nǎo 7\n',
                                     encoding='utf-8')


class FakeStream:
    def __init__(self, spotter):
        self.spotter = spotter
        self.samples = []

    def accept_waveform(self, rate, samples):
        self.samples.append((rate, len(samples)))

    def is_ready(self):
        return bool(self.samples) and self not in self.spotter.decoded

    def decode(self):
        self.spotter.decoded.append(self)


class FakeSpotter:
    """记录 KeywordSpotter 收到的全部参数，不做任何声学判断。"""

    calls = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.decoded = []
        FakeSpotter.calls.append(self)

    def create_stream(self):
        return FakeStream(self)

    def is_ready(self, stream):
        return stream.is_ready()

    def decode_stream(self, stream):
        stream.decode()


@pytest.fixture
def fake_sherpa(monkeypatch):
    FakeSpotter.calls = []
    monkeypatch.setitem(sys.modules, 'sherpa_onnx',
                        SimpleNamespace(KeywordSpotter=FakeSpotter))
    return FakeSpotter


def load_spotter(cfg, model_dir):
    kws = SherpaKws(cfg.kws, cfg.models, model_dir=model_dir)
    kws._ensure_loaded()
    return kws, FakeSpotter.calls[-1]


def test_shipped_target_config_selects_the_bench_profile():
    # Config/target.json 里没有 kws 段，运行时必须由 configure_kws 落到测试台 profile。
    cfg = AppConfig.from_file(str(TARGET_CONFIG))
    assert cfg.kws == KwsConfig()
    configure_kws(cfg)
    assert cfg.models.kws_dir == str(kws_adapter.PROFILE_DIR)
    assert (cfg.kws.keywords_score, cfg.kws.keywords_threshold,
            cfg.kws.num_trailing_blanks) == (1.0, 0.25, 1)
    assert cfg.kws.wake_phrases == ['小冰']


def test_profile_parameters_and_keyword_file_reach_the_spotter(fake_sherpa, tmp_path):
    assets(tmp_path)
    cfg = AppConfig()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kws_adapter, 'PROFILE_DIR', tmp_path)
        configure_kws(cfg)
    kws, spotter = load_spotter(cfg, tmp_path)
    args = spotter.kwargs
    assert args['keywords_score'] == 1.0
    assert args['keywords_threshold'] == 0.25
    assert args['num_trailing_blanks'] == 1
    assert args['sample_rate'] == 16000 and args['feature_dim'] == 80
    assert args['provider'] == 'cpu'
    keywords = Path(args['keywords_file'])
    assert keywords == kws.keywords_path
    assert keywords.is_file()
    # 只启用显式唤醒词；不把模型自带 stock 关键词带进运行时
    assert keywords.read_text(encoding='utf-8').strip().splitlines() == ['x iǎo b īng @小冰']


def test_threshold_tuning_is_read_from_config_not_hardcoded(fake_sherpa, tmp_path):
    assets(tmp_path)
    cfg = AppConfig()
    cfg.kws.keywords_score = 2.5
    cfg.kws.keywords_threshold = 0.4
    cfg.kws.num_trailing_blanks = 3
    kws, spotter = load_spotter(cfg, tmp_path)
    assert spotter.kwargs['keywords_score'] == 2.5
    assert spotter.kwargs['keywords_threshold'] == 0.4
    assert spotter.kwargs['num_trailing_blanks'] == 3


def test_explicit_keyword_file_bypasses_the_bench_profile(tmp_path):
    cfg = AppConfig()
    custom = tmp_path / 'custom-keywords.txt'
    custom.write_text('x iǎo b īng @小冰\n', encoding='utf-8')
    cfg.kws.keywords_file = str(custom)
    configure_kws(cfg)
    assert cfg.models.kws_dir == ModelsConfig().kws_dir
    assert cfg.kws.keywords_score == KwsConfig().keywords_score


def test_custom_keyword_file_is_the_one_handed_to_the_spotter(fake_sherpa, tmp_path):
    assets(tmp_path)
    custom = tmp_path / 'custom-keywords.txt'
    custom.write_text('x iǎo b īng @小冰\n', encoding='utf-8')
    cfg = AppConfig()
    cfg.kws.keywords_file = str(custom)
    kws, spotter = load_spotter(cfg, tmp_path)
    assert Path(spotter.kwargs['keywords_file']) == custom
    assert Path(spotter.kwargs['keywords_file']).read_text(encoding='utf-8') != \
        (tmp_path / 'tokens.txt').read_text(encoding='utf-8')


def test_near_homophone_wake_word_cannot_be_silently_added(tmp_path):
    # “小斌”是近音误唤醒的真问题，不能靠在热词表里默默加一条就宣称解决：
    # 拼音表里没有“斌”，配置会显式报错而不是被吞掉。
    assets(tmp_path)
    cfg = AppConfig()
    cfg.models.kws_dir = str(tmp_path)
    cfg.kws.wake_phrases = ['小冰', '小斌']
    with pytest.raises(ConfigError):
        SherpaKws(cfg.kws, cfg.models, model_dir=tmp_path)._prepare_keywords()


def test_bench_keyword_file_never_contains_unlisted_phrases(tmp_path):
    assets(tmp_path)
    cfg = AppConfig()
    target = SherpaKws(cfg.kws, cfg.models, model_dir=tmp_path)._prepare_keywords()
    lines = target.read_text(encoding='utf-8').splitlines()
    assert [line.rsplit('@', 1)[-1] for line in lines] == ['小冰']
    assert all('小斌' not in line for line in lines)


# ---------------------------------------------------------------------------
# 真实资产离线检查（不是唤醒率/误唤醒率测量，只是“这套资产能跑出什么”的观测）
#
# 用的是仓库里已有的真实模型与真实录音，不拷贝权重、不访问目标机：
#   Testbench/Data/Models/kws-zh-en-20251220-chunk8  (chunk-8 streaming zipformer)
#   Tests/Fixtures/audio/wake/tts_xiaobing_nihao.wav (22050 Hz 合成唤醒录音)
#   NX-AI/Models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs
# 资产缺失时直接失败（不跳过），否则“能运行的离线检查”会静默退化成空断言。
# ---------------------------------------------------------------------------
import wave

import numpy as np

from aieyes.interfaces import AudioFrame

BENCH_ASSETS = kws_adapter.PROFILE_DIR
WAKE_WAV = ROOT / 'Tests' / 'Fixtures' / 'audio' / 'wake' / 'tts_xiaobing_nihao.wav'
SHORT_WAKE_WAV = ROOT / 'Tests' / 'Fixtures' / 'audio' / 'wake' / 'tts_xiaobing.wav'
if not WAKE_WAV.is_file():
    WAKE_WAV = ROOT / 'tests/fixtures/audio/wake/tts_xiaobing_nihao.wav'
if not SHORT_WAKE_WAV.is_file():
    SHORT_WAKE_WAV = ROOT / 'tests/fixtures/audio/wake/tts_xiaobing.wav'
NO_WAKE_WAVS = sorted(
    (ROOT / 'NX-AI' / 'Models' / 'kws'
     / 'sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01' / 'test_wavs').glob('*.wav'))
if not NO_WAKE_WAVS:
    NO_WAKE_WAVS = sorted((ROOT / 'models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01/test_wavs').glob('*.wav'))


def load_16k(path):
    """读任意采样率单/双声道 wav，线性插值重采样到 16 kHz float32。"""
    with wave.open(str(path)) as handle:
        rate, channels = handle.getframerate(), handle.getnchannels()
        raw = np.frombuffer(handle.readframes(handle.getnframes()), dtype=np.int16)
    data = raw.astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        target = np.arange(int(round(len(data) * 16000 / rate))) / 16000.0
        data = np.interp(target, np.arange(len(data)) / rate, data).astype(np.float32)
    return data


def feed(kws, wav, chunk=1600, cooldown_ms=0):
    """按 100 ms 帧喂完整段音频，返回 [(秒, 命中词)] + flush 出来的尾随命中。"""
    kws.cfg.cooldown_ms = cooldown_ms
    samples = load_16k(wav)
    hits, t_ms = [], 1000
    for index, start in enumerate(range(0, len(samples) - chunk + 1, chunk)):
        got = kws.accept(AudioFrame(samples=samples[start:start + chunk],
                                    t_start_ms=t_ms, seq=index))
        t_ms += int(chunk * 1000 / 16000)
        if got:
            hits.append((round(start / 16000.0, 2), got))
    tail = kws.flush()
    if tail:
        hits.append(('tail', tail))
    return hits


def real_kws():
    cfg = AppConfig()
    configure_kws(cfg)
    assert cfg.models.kws_dir == str(BENCH_ASSETS)
    return SherpaKws(cfg.kws, cfg.models, model_dir=BENCH_ASSETS)


def test_real_bench_assets_are_present():
    # 前置：没有这些文件就不是“能运行的离线检查”，必须立刻暴露而不是跳过。
    for name in ('tokens.txt', 'encoder.int8.onnx', 'decoder.onnx', 'joiner.int8.onnx'):
        assert (BENCH_ASSETS / name).is_file(), f'缺少真实唤醒资产 {BENCH_ASSETS / name}'
    assert WAKE_WAV.is_file() and SHORT_WAKE_WAV.is_file()
    assert NO_WAKE_WAVS, '缺少无唤醒词的负例语音'


def test_real_keywords_file_lands_outside_the_model_and_uses_token_string():
    kws = real_kws()
    path = kws._prepare_keywords()
    kws.close()
    text = path.read_text(encoding='utf-8')
    # token 串必须与 tokens.txt 的真实 token 一致（不是整字/整拼音）
    tokens = (BENCH_ASSETS / 'tokens.txt').read_text(encoding='utf-8').split()
    for token in ('x', 'iǎo', 'b', 'īng'):
        assert token in tokens, f'tokens.txt 里没有 token {token}'
    assert text == 'x iǎo b īng @小冰\n'
    # 不写模型目录，也不污染仓库根：落在 Testbench/.runtime 下
    assert path.parent.name == 'kws_configured'
    assert path.parent.parent.parent.parent == Path(kws_adapter.__file__).resolve().parent
    assert not list(BENCH_ASSETS.glob('*.txt')) or all(
        p.name == 'tokens.txt' for p in BENCH_ASSETS.glob('*.txt'))


def test_real_wake_recording_hits_the_wake_word():
    kws = real_kws()
    try:
        hits = feed(kws, WAKE_WAV)
    finally:
        kws.close()
    # 离线观测：真实资产 + 真实录音跑出“小冰”命中；不据此推算唤醒率。
    assert [word for _, word in hits] == ['小冰']


def test_real_short_wake_recording_does_not_invent_a_longer_sentence():
    kws = real_kws()
    try:
        hits = feed(kws, SHORT_WAKE_WAV)
    finally:
        kws.close()
    # 0.77s 只含“小冰”，命中点最多一个，且返回词就是唤醒词本身
    assert len(hits) <= 1
    assert all(word == '小冰' for _, word in hits)


def test_real_non_wake_speech_does_not_trigger_the_wake_word():
    kws = real_kws()
    try:
        observed = {}
        for wav in NO_WAKE_WAVS:
            observed[wav.name] = feed(kws, wav)
    finally:
        kws.close()
    triggered = {name: hits for name, hits in observed.items() if hits}
    # 负例观测：新闻播报语音不应唤醒；这是离线替身资产下的 0/N，不是误唤醒率。
    assert triggered == {}
