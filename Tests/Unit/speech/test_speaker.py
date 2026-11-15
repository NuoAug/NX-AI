"""声纹单测：素材来自 fixtures/speaker（he/me/she 三说话人各 3 段）。"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from speech_helpers import FIXTURES

from aieyes.audio.wav import read_wav
from aieyes.interfaces import SAMPLE_RATE
from aieyes.speech.speaker import SherpaSpeaker


def _fixture_files() -> list:
    folder = FIXTURES / "speaker"
    return sorted(folder.glob("*.wav")) if folder.is_dir() else []


def _embeddings(speaker) -> dict[str, np.ndarray]:
    files = _fixture_files()
    if len(files) < 6:
        pytest.skip(f"声纹素材不足: {len(files)} 个")
    return {path.stem: speaker.embed_file(path) for path in files}


def test_embed_shape_and_norm(speaker):
    files = _fixture_files()
    if not files:
        pytest.skip("缺少 fixtures/speaker/*.wav")
    samples, rate = read_wav(files[0], target_rate=SAMPLE_RATE)
    emb = speaker.embed(samples, rate)
    assert emb.shape == (speaker.dim,)
    assert speaker.dim == 192
    assert abs(float(np.linalg.norm(emb)) - 1.0) < 1e-3


def test_same_speaker_higher_than_different(speaker):
    embs = _embeddings(speaker)
    groups: dict[str, list[str]] = {}
    for name in embs:
        groups.setdefault(name.split("_")[0], []).append(name)

    same = [
        speaker.cosine(embs[a], embs[b])
        for names in groups.values()
        for a, b in combinations(sorted(names), 2)
    ]
    cross = [
        speaker.cosine(embs[a], embs[b])
        for a, b in combinations(sorted(embs), 2)
        if embs and a.split("_")[0] != b.split("_")[0]
    ]
    assert same and cross
    assert min(same) > max(cross), (min(same), max(cross))
    assert min(same) > 0.65, min(same)
    print(
        f"[speaker] same min={min(same):.4f} mean={np.mean(same):.4f} | "
        f"cross max={max(cross):.4f} mean={np.mean(cross):.4f}"
    )


def test_embed_file_deterministic_and_cosine_static(speaker):
    files = _fixture_files()
    if not files:
        pytest.skip("缺少 fixtures/speaker/*.wav")
    first = speaker.embed_file(files[0])
    second = speaker.embed_file(files[0])
    assert speaker.cosine(first, second) > 0.999
    assert isinstance(speaker, SherpaSpeaker)
    assert SherpaSpeaker.cosine(first, first) == pytest.approx(1.0, abs=1e-5)
    assert SherpaSpeaker.cosine(np.zeros(192), first) == 0.0
