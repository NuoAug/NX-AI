"""Use only explicitly configured wake phrases; keep model assets untouched."""
from hashlib import sha256
from pathlib import Path
from aieyes.speech.kws import SherpaKws as ProjectKws, build_keywords_text, load_tokens
from aieyes.errors import ConfigError

PROFILE_DIR = Path(__file__).resolve().parent / 'data/models/kws-zh-en-20251220-chunk8'


def configure_kws(cfg):
    """Select the bench-owned tested model, preserving explicit custom models."""
    from aieyes.config import ModelsConfig, KwsConfig
    if cfg.models.kws_dir != ModelsConfig().kws_dir or cfg.kws.keywords_file:
        return
    if not PROFILE_DIR.is_dir():
        return
    for name in ('encoder.int8.onnx', 'decoder.onnx', 'joiner.int8.onnx', 'tokens.txt'):
        if not (PROFILE_DIR / name).is_file():
            raise ConfigError('测试台唤醒模型文件缺失：' + name)
    cfg.models.kws_dir = str(PROFILE_DIR)
    defaults = KwsConfig()
    if cfg.kws.keywords_score == defaults.keywords_score:
        cfg.kws.keywords_score = 1.0
    if cfg.kws.num_trailing_blanks == defaults.num_trailing_blanks:
        cfg.kws.num_trailing_blanks = 1


class SherpaKws(ProjectKws):
    def _ensure_loaded(self):
        loaded = self.ready
        super()._ensure_loaded()
        if not loaded:
            import numpy as np
            # Three frontend warmup frames do not fill a streaming model
            # chunk. Warm an independent stream, then keep live state fresh.
            warm = self._spotter.create_stream()
            warm.accept_waveform(16000, np.zeros(16000, np.float32))
            while self._spotter.is_ready(warm):
                self._spotter.decode_stream(warm)

    def _prepare_keywords(self):
        text = build_keywords_text(load_tokens(self.model_dir / 'tokens.txt'), self.cfg.wake_phrases)
        if not text.strip():
            raise ConfigError('至少配置一个唤醒词')
        cache = Path(__file__).resolve().parent / '.runtime/tmp/kws_configured'
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / (sha256(text.encode()).hexdigest()[:16] + '.txt')
        if not target.is_file() or target.read_text(encoding='utf-8') != text:
            target.write_text(text, encoding='utf-8')
        return target
