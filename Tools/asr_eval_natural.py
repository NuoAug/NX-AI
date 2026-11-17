"""用项目真实录音评估流式 ASR（基线/改动同一批音频对照）。

输入：16 kHz 单声道 PCM16 wav（默认 fixtures/questions）。
输出：每个文件的原始识别文本、字符错误率（CER）、VAD 分段数和处理耗时。

用法（目标板）：
    source tools/target_env.sh && trun "python tools/asr_eval_natural.py"
    trun "python tools/asr_eval_natural.py --mode vad --no-silence-skip"
"""

from __future__ import annotations

import argparse
import re
import time
import wave
from pathlib import Path

import numpy as np

from aieyes.audio.vad import EnergyVad
from aieyes.config import AppConfig
from aieyes.interfaces import FRAME_SAMPLES, SAMPLE_RATE, AudioFrame
from aieyes.speech.asr import StreamingAsr

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "fixtures" / "questions"
PUNCT_RE = re.compile(r"[\s，。！？!?、,.；;：:\"'“”‘’()（）\[\]【】]+")


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        rate = w.getframerate()
        width = w.getsampwidth()
        channels = w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise SystemExit(f"{path}: 只支持 16-bit PCM，实际 {width * 8}-bit")
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    if rate != 16000:
        raise SystemExit(f"{path}: 需要 16 kHz，实际 {rate} Hz（请先重采样）")
    return data, rate


def expected_from_name(path: Path) -> str:
    stem = path.stem
    parts = stem.split("-", 2)
    text = parts[2] if len(parts) >= 3 else stem
    text = text.replace("Downloads", "")
    text = re.sub(r"\(\d+\)$", "", text)
    return text


def frames(samples: np.ndarray, *, start_ms: int = 0) -> list[AudioFrame]:
    step = FRAME_SAMPLES
    out: list[AudioFrame] = []
    for seq, start in enumerate(range(0, len(samples) - step + 1, step)):
        out.append(
            AudioFrame(
                samples=samples[start : start + step],
                t_start_ms=start_ms + int(1000 * start / SAMPLE_RATE),
                seq=seq,
            )
        )
    return out


def transcribe(asr: StreamingAsr, samples: np.ndarray, *, tail_ms: int = 600) -> str:
    asr.start_utterance()
    for frame in frames(samples):
        asr.accept(frame)
    tail = np.zeros(int(tail_ms / 1000 * SAMPLE_RATE), dtype=np.float32)
    for frame in frames(tail, start_ms=int(1000 * len(samples) / SAMPLE_RATE)):
        asr.accept(frame)
    return asr.finalize().text


class SenseVoiceEval:
    """离线整句识别对照（只用于评测，不接入主链路）。"""

    def __init__(self, model_dir: Path, *, num_threads: int = 2) -> None:
        import sherpa_onnx

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(model_dir / "model.int8.onnx"),
            tokens=str(model_dir / "tokens.txt"),
            num_threads=num_threads,
            use_itn=True,
            language="zh",
            debug=False,
        )

    def transcribe(self, samples: np.ndarray) -> str:
        stream = self.recognizer.create_stream()
        stream.accept_waveform(SAMPLE_RATE, np.asarray(samples, dtype=np.float32))
        self.recognizer.decode_stream(stream)
        text = str(stream.result.text or "")
        return re.sub(r"<\|[^|]*\|>", "", text).strip()


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return 0.0


def transcribe_with_vad(
    asr: StreamingAsr, vad, samples: np.ndarray, *, tail_ms: int = 600
) -> tuple[str, list[str]]:
    texts: list[str] = []
    causes: list[str] = []
    vad.reset()
    for frame in frames(samples):
        segment = vad.accept(frame)
        if segment is None:
            continue
        if len(segment.samples):
            texts.append(transcribe(asr, np.asarray(segment.samples, dtype=np.float32), tail_ms=tail_ms))
            causes.append(segment.cause)
    # 收尾：还在语音里但没到静音阈值时，补一段静音让 VAD 结束
    silence = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    for frame in frames(silence, start_ms=int(1000 * len(samples) / SAMPLE_RATE)):
        segment = vad.accept(frame)
        if segment is not None and len(segment.samples):
            texts.append(transcribe(asr, np.asarray(segment.samples, dtype=np.float32), tail_ms=tail_ms))
            causes.append(segment.cause)
    return "".join(text for text in texts if text), causes


def transcribe_with_vad_offline(engine, vad, samples: np.ndarray) -> tuple[str, list[str]]:
    texts: list[str] = []
    causes: list[str] = []
    vad.reset()
    for frame in frames(samples):
        segment = vad.accept(frame)
        if segment is not None and len(segment.samples):
            texts.append(engine.transcribe(np.asarray(segment.samples, dtype=np.float32)))
            causes.append(segment.cause)
    silence = np.zeros(int(1.0 * SAMPLE_RATE), dtype=np.float32)
    for frame in frames(silence, start_ms=int(1000 * len(samples) / SAMPLE_RATE)):
        segment = vad.accept(frame)
        if segment is not None and len(segment.samples):
            texts.append(engine.transcribe(np.asarray(segment.samples, dtype=np.float32)))
            causes.append(segment.cause)
    return "".join(text for text in texts if text), causes


def cer(ref: str, hyp: str) -> float:
    ref = PUNCT_RE.sub("", ref or "")
    hyp = PUNCT_RE.sub("", hyp or "")
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        cur = [i]
        for j, h in enumerate(hyp, start=1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1] / len(ref)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=str(DEFAULT_DIR))
    parser.add_argument("--mode", choices=("stream", "vad"), default="stream")
    parser.add_argument("--engine", choices=("paraformer", "sensevoice"), default="paraformer")
    parser.add_argument("--sv-dir", default=str(ROOT / "models" / "sensevoice-small"))
    parser.add_argument("--no-silence-skip", action="store_true")
    parser.add_argument("--tail-ms", type=int, default=600)
    parser.add_argument("--pad-ms", type=int, default=500)
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    cfg = AppConfig.from_file(Path(__file__).resolve().parents[1] / "config" / "target.json")
    if args.no_silence_skip:
        cfg.asr.silence_skip_rms = 0.0
    asr = StreamingAsr(cfg.asr) if args.engine == "paraformer" else None
    if asr is not None:
        asr._TAIL_PAD_MS = int(args.pad_ms)
    sv = SenseVoiceEval(Path(args.sv_dir)) if args.engine == "sensevoice" else None
    if args.warmup:
        silence = np.zeros(int(0.1 * SAMPLE_RATE), dtype=np.float32)
        if asr is not None:
            transcribe(asr, silence, tail_ms=100)
        elif sv is not None:
            sv.transcribe(silence)
    vad = EnergyVad(cfg.vad) if args.mode == "vad" else None
    files = sorted(Path(args.dir).glob("*.wav"))
    if args.limit:
        files = files[: args.limit]
    total_cer = 0.0
    total_splits = 0
    for path in files:
        samples, _rate = read_wav(path)
        ref = expected_from_name(path)
        t0 = time.perf_counter()
        if args.engine == "sensevoice":
            if vad is None:
                hyp = sv.transcribe(samples)
                causes = []
            else:
                hyp, causes = transcribe_with_vad_offline(sv, vad, samples)
        elif vad is None:
            hyp = transcribe(asr, samples, tail_ms=args.tail_ms)
            causes = []
        else:
            hyp, causes = transcribe_with_vad(asr, vad, samples, tail_ms=args.tail_ms)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        audio_ms = len(samples) * 1000 / SAMPLE_RATE
        score = cer(ref, hyp)
        total_cer += score
        total_splits += max(0, len(causes) - 1)
        print(
            f"{path.name}\n  ref={ref!r}\n  hyp={hyp!r}\n"
            f"  cer={score:.3f} splits={max(0, len(causes) - 1)} causes={','.join(causes)}\n"
            f"  audio_ms={audio_ms:.0f} process_ms={elapsed_ms:.0f} rtf={elapsed_ms / max(audio_ms, 1):.3f}"
        )
    if files:
        print(
            f"\nSUMMARY mode={args.mode} engine={args.engine} files={len(files)} "
            f"mean_cer={total_cer / len(files):.3f} "
            f"extra_splits={total_splits} silence_skip_rms={cfg.asr.silence_skip_rms} "
            f"tail_ms={args.tail_ms} pad_ms={args.pad_ms} rss_mb={rss_mb():.0f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
