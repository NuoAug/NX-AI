"""阶段 7 性能探针：整栈常驻内存 + 单次真实调用时延（目标板运行）。

用法：
    trun "python Tools/mem_probe.py --config Config/target.json --image Tests/Fixtures/images/text_cn_price.png"

输出一行 JSON：各阶段 RSS/VmHWM、llama-server RSS、总占用、各阶段耗时。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from aieyes.clock import mono_ms
from aieyes.config import AppConfig
from aieyes.interfaces import AudioFrame, FRAME_SAMPLES, SAMPLE_RATE


def _rss_kb() -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def _peak_kb() -> int:
    with open("/proc/self/status", "r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


def _server_rss_kb() -> int:
    out = subprocess.run(
        ["ps", "-eo", "rss,cmd"], capture_output=True, text=True, check=False
    ).stdout
    total = 0
    for line in out.splitlines():
        if "llama-server" in line and "grep" not in line:
            try:
                total += int(line.split(None, 1)[0])
            except (ValueError, IndexError):
                continue
    return total


def _server_peak_kb() -> int:
    """llama-server 的 VmHWM（从 /proc/<pid>/status 读，多个进程取和）。"""
    out = subprocess.run(
        ["ps", "-eo", "pid,cmd"], capture_output=True, text=True, check=False
    ).stdout
    total = 0
    for line in out.splitlines():
        if "llama-server" not in line or "grep" in line:
            continue
        try:
            pid = line.split(None, 1)[0]
        except IndexError:
            continue
        try:
            with open(f"/proc/{pid}/status", "r", encoding="utf-8") as fh:
                for entry in fh:
                    if entry.startswith("VmHWM:"):
                        total += int(entry.split()[1])
                        break
        except OSError:
            continue
    return total


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Config/target.json")
    parser.add_argument("--image", default="Tests/Fixtures/images/text_cn_price.png")
    parser.add_argument(
        "--image2", default="Tests/Fixtures/images/scene_red_square.png"
    )
    args = parser.parse_args(argv)

    cfg = AppConfig.from_file(args.config)
    image_path = Path(args.image)
    marks: dict[str, float] = {}
    timings: dict[str, float] = {}

    def mark(name: str) -> None:
        marks[name] = _rss_kb() / 1024.0

    def timed(name: str, fn):
        t0 = mono_ms()
        value = fn()
        timings[name] = float(mono_ms() - t0)
        return value

    mark("baseline")

    from aieyes.speech.tts import PiperTts

    tts = timed("load_tts", lambda: PiperTts(cfg.tts))
    mark("tts_loaded")
    timed("synth_wozai", lambda: tts.synth("我在"))

    from aieyes.speech.kws import SherpaKws

    kws = timed("load_kws", lambda: SherpaKws(cfg.kws))
    mark("kws_loaded")
    silence = np.zeros(FRAME_SAMPLES, dtype=np.float32)

    def kws_frames() -> None:
        for i in range(5):
            kws.accept(AudioFrame(silence, i * 20, i))

    timed("kws_5frames", kws_frames)

    from aieyes.speech.asr import StreamingAsr

    asr = timed("load_asr", lambda: StreamingAsr(cfg.asr))
    mark("asr_loaded")

    def asr_once() -> None:
        asr.start_utterance()
        for i in range(10):
            asr.accept(AudioFrame(np.zeros(FRAME_SAMPLES, dtype=np.float32), i * 20, i))
        asr.finalize()

    timed("asr_10frames", asr_once)

    from aieyes.vision.ocr import OcrEngine

    ocr = timed("load_ocr", lambda: OcrEngine(cfg.ocr))
    mark("ocr_loaded")
    rgb = np.asarray(Image.open(image_path).convert("RGB"))
    timed("ocr_once", lambda: ocr.recognize(rgb))

    from aieyes.speech.speaker import SherpaSpeaker

    speaker = timed("load_speaker", lambda: SherpaSpeaker(cfg.speaker, cfg.models))
    mark("speaker_loaded")
    timed("speaker_embed", lambda: speaker.embed(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE))

    from aieyes.scene.providers import FileFrameProvider
    from aieyes.scene.service import SceneService
    from aieyes.vision.vlm import VlmClient

    vlm = timed("load_vlm_client", lambda: VlmClient(cfg.llm, cfg.vlm))
    provider = FileFrameProvider([str(image_path)])
    scene = SceneService(vlm=vlm, ocr=ocr, provider=provider)
    frame = provider.acquire(purpose="vqa")
    mark("scene_ready")
    answer = timed("scene_price_answer", lambda: scene.answer("这个多少钱？", frame=frame))
    mark("after_scene")

    rgb2 = np.asarray(Image.open(Path(args.image2)).convert("RGB"))
    timed(
        "vlm_text_once",
        lambda: vlm.complete("用一个词回答：一加一等于几？", max_tokens=12),
    )
    timed(
        "vlm_image_once",
        lambda: vlm.complete(
            "这张图里主要是什么颜色？", images=[rgb2], max_tokens=12
        ),
    )
    mark("after_vlm_image")

    server_kb = _server_rss_kb()
    server_peak_kb = _server_peak_kb()
    process_kb = _rss_kb()
    total = (server_kb + process_kb) / 1024.0 / 1024.0
    total_peak = (server_peak_kb + _peak_kb()) / 1024.0 / 1024.0
    result = {
        "process_rss_mib": round(process_kb / 1024.0, 1),
        "process_peak_mib": round(_peak_kb() / 1024.0, 1),
        "llama_server_rss_mib": round(server_kb / 1024.0, 1),
        "llama_server_peak_mib": round(server_peak_kb / 1024.0, 1),
        "stack_total_gib": round(total, 3),
        "stack_peak_gib": round(total_peak, 3),
        "cap_gib": 4.0,
        "within_cap": total <= 4.0,
        "within_cap_peak": total_peak <= 4.0,
        "marks_mib": {k: round(v, 1) for k, v in marks.items()},
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "answer": getattr(answer, "text", ""),
        "answer_sources": [str(t) for t in (getattr(answer, "source_tags", ()) or ())],
        "answer_intent": getattr(answer, "intent", ""),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
