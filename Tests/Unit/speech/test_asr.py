"""流式 ASR 单测：模型自带 wav → 中文文本 + 时间戳。"""

from __future__ import annotations

from speech_helpers import frames_from_wav, has_cjk

from aieyes.interfaces import SAMPLE_RATE


def _question_wav(models):
    return models.resolve(models.asr_dir) / "test_wavs" / "0.wav"


def test_transcribe_test_wav(asr, models):
    asr.reset()
    asr.start_utterance()
    frames = frames_from_wav(_question_wav(models))
    partials = []
    for frame in frames:
        partial = asr.accept(frame)
        if partial:
            partials.append(partial)
    utt = asr.finalize()

    assert utt.text.strip(), f"ASR 输出为空（partials={partials[-3:]}）"
    assert has_cjk(utt.text), f"ASR 输出不含中文: {utt.text!r}"
    assert utt.t_audio_start_ms == frames[0].t_start_ms
    assert utt.t_audio_end_ms >= utt.t_audio_start_ms
    assert utt.t_final_ms >= utt.t_audio_end_ms
    assert utt.duration_ms > 1000
    print(f"[asr] text={utt.text!r} partials={len(partials)} duration_ms={utt.duration_ms}")


def test_reset_and_empty_finalize(asr):
    asr.reset()
    assert asr.has_utterance is False
    utt = asr.finalize()
    assert utt.text == ""
    assert utt.t_audio_start_ms >= 0
    assert asr.has_utterance is False


def test_accept_starts_utterance_implicitly(asr, models):
    asr.reset()
    frames = frames_from_wav(_question_wav(models))[:20]  # 只喂 0.4s
    out = [asr.accept(frame) for frame in frames]
    assert asr.has_utterance is True
    utt = asr.finalize()
    assert utt.t_audio_end_ms > utt.t_audio_start_ms
    assert all(p is None or isinstance(p, str) for p in out)
