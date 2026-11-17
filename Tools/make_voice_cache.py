"""生成固定提醒的离线音色缓存（开发期一次，运行时不联网）。

音色按产品选定：``zh-CN-XiaoxiaoNeural``，rate +0%、pitch +0Hz、volume +0%。
本工具需要联网调用 edge-tts，**只用于生成交付资产**；盒端运行时只读 WAV。
生成后用 ``aieyes.speech.tts.normalize_pcm`` 把响度对齐到参考试听
（RMS -19 dBFS，峰值不超过 -1 dBFS），并写 ``index.json`` 供运行时校验。

运行（目标板）：
    python3 -m venv /tmp/edge_venv
    /tmp/edge_venv/bin/pip install edge-tts
    /tmp/edge_venv/bin/python Tools/make_voice_cache.py

依赖：edge-tts、gst-launch-1.0（目标机 ffmpeg 不带 mp3 解码时用它）。
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from aieyes.audio.wav import read_wav, write_wav  # noqa: E402
from aieyes.speech.tts import _peak_dbfs, _rms_dbfs, normalize_pcm  # noqa: E402

DEFAULT_CATALOG = ROOT / "NX-AI/Models/tts-fixed/xiaoxiao/fixed_prompts.json"
SAMPLE_RATE = 22050


def _mp3_to_wav(mp3: Path, wav: Path) -> None:
    """MP3 → 22050Hz 单声道 WAV；优先 GStreamer，失败再试 ffmpeg。"""
    gst = shutil.which("gst-launch-1.0")
    if gst:
        cmd = [
            gst, "-q",
            "filesrc", f"location={mp3}",
            "!", "mpegaudioparse", "!", "mpg123audiodec",
            "!", "audioconvert", "!", "audioresample",
            "!", f"audio/x-raw,format=S16LE,rate={SAMPLE_RATE},channels=1",
            "!", "wavenc", "!", "filesink", f"location={wav}",
        ]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode == 0 and wav.is_file():
            return
        gst_err = (done.stderr or done.stdout or "").strip()
    else:
        gst_err = "gst-launch-1.0 不存在"
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-i", str(mp3),
               "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le", str(wav)]
        done = subprocess.run(cmd, capture_output=True, text=True)
        if done.returncode == 0 and wav.is_file():
            return
    raise RuntimeError(f"MP3 解码失败：{gst_err}")


def _synth_mp3(text: str, voice: str, rate: str, pitch: str, volume: str, mp3: Path) -> None:
    try:
        import edge_tts  # noqa: PLC0415 - 仅开发期依赖
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "缺少 edge-tts（仅生成资产需要）：/tmp/edge_venv/bin/pip install edge-tts"
        ) from exc

    async def _run() -> None:
        communicate = edge_tts.Communicate(
            text, voice, rate=rate, pitch=pitch, volume=volume
        )
        await communicate.save(str(mp3))

    asyncio.run(_run())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="生成固定提醒的离线音色缓存")
    parser.add_argument("--catalog", default=str(DEFAULT_CATALOG))
    parser.add_argument("--out", default="")
    parser.add_argument("--voice", default="")
    parser.add_argument("--rate", default="")
    parser.add_argument("--pitch", default="")
    parser.add_argument("--volume", default="")
    parser.add_argument("--reference", default="", help="参考试听 WAV（可选）")
    args = parser.parse_args()

    catalog_path = Path(args.catalog)
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out) if args.out else catalog_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    voice = args.voice or str(catalog.get("voice") or "zh-CN-XiaoxiaoNeural")
    rate = args.rate or str(catalog.get("rate") or "+0%")
    pitch = args.pitch or str(catalog.get("pitch") or "+0Hz")
    volume = args.volume or str(catalog.get("volume") or "+0%")
    target_rms = float(catalog.get("target_rms_dbfs", -19.0))
    peak_ceiling = float(catalog.get("peak_ceiling_dbfs", -1.0))

    entries: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="voice_cache_") as tmp:
        tmp_dir = Path(tmp)
        for item in catalog.get("prompts", []):
            text = str(item.get("text") or "").strip()
            name = str(item.get("id") or "").strip()
            if not text or not name:
                print(f"SKIP 非法条目: {item!r}")
                continue
            mp3 = tmp_dir / f"{name}.mp3"
            raw_wav = tmp_dir / f"{name}.wav"
            _synth_mp3(text, voice, rate, pitch, volume, mp3)
            _mp3_to_wav(mp3, raw_wav)
            samples, _ = read_wav(raw_wav, target_rate=None, mono=True)
            samples = normalize_pcm(
                samples, target_rms_dbfs=target_rms, peak_ceiling_dbfs=peak_ceiling
            )
            file_name = f"{name}.wav"
            final = out_dir / file_name
            write_wav(final, samples, SAMPLE_RATE)
            entries[text] = {
                "file": file_name,
                "category": str(item.get("category") or ""),
                "sha256": _sha256(final),
                "sample_rate": SAMPLE_RATE,
                "duration_ms": int(round(len(samples) * 1000 / SAMPLE_RATE)),
                "rms_dbfs": _rms_dbfs(samples),
                "peak_dbfs": _peak_dbfs(samples),
            }
            print(
                f"OK {name}: {file_name} {entries[text]['duration_ms']}ms "
                f"rms={entries[text]['rms_dbfs']}dBFS peak={entries[text]['peak_dbfs']}dBFS"
            )

    index = {
        "voice": voice,
        "rate": rate,
        "pitch": pitch,
        "volume": volume,
        "target_rms_dbfs": target_rms,
        "peak_ceiling_dbfs": peak_ceiling,
        "sample_rate": SAMPLE_RATE,
        "generated_by": "Tools/make_voice_cache.py",
        "runtime": "offline_wav_cache",
        "entries": entries,
    }
    ref = Path(args.reference) if args.reference else out_dir / "reference.wav"
    if ref.is_file():
        ref_audio, _ = read_wav(ref, target_rate=None, mono=True)
        index["reference"] = {
            "file": ref.name,
            "sha256": _sha256(ref),
            "sample_rate": SAMPLE_RATE,
            "duration_ms": int(round(len(ref_audio) * 1000 / SAMPLE_RATE)),
            "rms_dbfs": _rms_dbfs(ref_audio),
            "peak_dbfs": _peak_dbfs(ref_audio),
            "note": "用户提供，在线生成，仅作响度参考",
        }

    index_path = out_dir / "index.json"
    index_path.write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"WROTE {index_path} ({len(entries)} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
