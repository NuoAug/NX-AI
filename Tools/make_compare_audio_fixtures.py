"""生成阶段 4 集成测试用的固定问句音频。

背景：运行时用 PiperTts 合成问句，音色/时长每次不同，ASR 偶尔把“继续比较…”识别成
“继需比较…”，导致测试偶发失败。这里把“合成→ASR 自检→落盘”一次做掉，测试改为回放固定
wav，既确定又可重复。

运行（目标板）：
    source Tools/target_env.sh && tsync && trun "python Tools/make_compare_audio_fixtures.py"
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from aieyes.audio.wav import write_wav
from aieyes.config import AppConfig
from aieyes.clock import mono_ms
from aieyes.interfaces import FRAME_SAMPLES, SAMPLE_RATE, AudioFrame
from aieyes.speech.asr import StreamingAsr
from aieyes.speech.tts import PiperTts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "Config" / "target.json"
OUT = ROOT / "Tests" / "Fixtures" / "audio" / "questions"

TARGETS = [
    ("compare_two_which_expensive", "这两个哪个贵？", ("哪个", "那", "贵")),
    ("resume_compare", "继续比较这两样东西的价格", ("继",)),
    ("compare_which_more_expensive", "这两个哪个更贵一些呢？", ("哪个", "那")),
]

MAX_ATTEMPTS = 8


def _frames(samples: np.ndarray, t0: int) -> list[AudioFrame]:
    out: list[AudioFrame] = []
    step = FRAME_SAMPLES
    for seq, start in enumerate(range(0, len(samples) - step + 1, step)):
        out.append(
            AudioFrame(
                samples=samples[start : start + step],
                t_start_ms=t0 + int(1000 * start / SAMPLE_RATE),
                seq=seq,
            )
        )
    return out


def _transcribe(asr: StreamingAsr, samples: np.ndarray) -> str:
    asr.start_utterance()
    t0 = mono_ms()
    for frame in _frames(samples, t0):
        asr.accept(frame)
    # 结尾补 600ms 静音，让流式识别收尾
    tail = np.zeros(int(0.6 * SAMPLE_RATE), dtype=np.float32)
    for frame in _frames(tail, t0 + int(1000 * len(samples) / SAMPLE_RATE)):
        asr.accept(frame)
    return asr.finalize().text


def main() -> int:
    cfg = AppConfig.from_file(CONFIG)
    OUT.mkdir(parents=True, exist_ok=True)
    tts = PiperTts(cfg.tts)
    asr = StreamingAsr(cfg.asr)
    failed = 0
    try:
        for name, text, must in TARGETS:
            saved = False
            for attempt in range(1, MAX_ATTEMPTS + 1):
                audio = tts.synth(text)
                samples = np.asarray(audio.pcm, dtype=np.float32)
                heard = _transcribe(asr, samples)
                hit = bool(heard) and any(token in heard for token in must)
                print(f"{name} #{attempt}: asr={heard!r} hit={hit}")
                if hit:
                    path = OUT / f"{name}.wav"
                    write_wav(path, samples, audio.sample_rate)
                    print(f"SAVED {path} ({path.stat().st_size} bytes, {len(samples)} samples @ {audio.sample_rate}Hz)")
                    saved = True
                    break
            if not saved:
                failed += 1
                print(f"FAILED {name}: {MAX_ATTEMPTS} 次合成都没有得到可识别结果")
    finally:
        tts.close()
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
