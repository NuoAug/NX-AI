"""speech 单测公共 fixture：模型加载慢，尽量 session 复用。"""

from __future__ import annotations

from pathlib import Path

import pytest

from aieyes.config import ModelsConfig


@pytest.fixture(scope="session")
def models() -> ModelsConfig:
    cfg = ModelsConfig()
    if not Path(cfg.root).is_dir():
        pytest.skip(f"模型目录不存在: {cfg.root}")
    return cfg


@pytest.fixture(scope="session")
def kws(models):
    from aieyes.speech.kws import SherpaKws

    spotter = SherpaKws(models=models)
    yield spotter
    spotter.close()


@pytest.fixture(scope="session")
def asr(models):
    from aieyes.speech.asr import StreamingAsr

    recognizer = StreamingAsr(models=models)
    yield recognizer
    recognizer.close()


@pytest.fixture(scope="session")
def tts(models):
    from aieyes.speech.tts import PiperTts

    engine = PiperTts(models=models)
    yield engine
    engine.close()


@pytest.fixture(scope="session")
def speaker(models):
    from aieyes.speech.speaker import SherpaSpeaker

    embedder = SherpaSpeaker(models=models)
    yield embedder
    embedder.close()
