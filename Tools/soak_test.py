#!/usr/bin/env python3
"""阶段 8：30 分钟持续交互检查（目标板，回放输入 + 真实模型，不需要麦克风）。

按固定节奏回放「唤醒 → 静音 → 真人问题 → 静音」循环，驱动完整软件栈：
ReplaySource → KWS/VAD/ASR → InteractionEngine → ChatSession/VLM 流式 → PiperTts → AudioSink。

只做观测，不做断言失败即退出：结束时输出 JSON 摘要（迭代数、回答数、失败数、
首声/整轮时延、RSS 峰值、证据行数），供阶段 8 报告引用。

用法（目标板项目根，venv 内）：
    python Tools/soak_test.py --config Config/target.json --minutes 30
    python Tools/soak_test.py --config Config/target.json --minutes 2 --out evidence/soak/quick.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Linux 区分大小写：仓库/交付根里实际叫 ``src/``，历史写法 ``Src/`` 只在 Windows
# （不区分大小写）能命中。两个都探，命中即用，避免板上 import aieyes 直接失败。
for _pkg_parent in (ROOT / "src", ROOT / "Src"):
    if _pkg_parent.is_dir():
        if str(_pkg_parent) not in sys.path:
            sys.path.insert(0, str(_pkg_parent))
        break

import numpy as np  # noqa: E402

# Windows 本机没有 POSIX 的 fcntl/termios/tty，而 aieyes.vision 在**导入期**就要
# fcntl（camera.py 顶层 import）。这里只在真正缺失时补最小占位模块，让
# ``--help``、峰值核算等本机入口可用；真实采样仍在目标板上做，不改变板上行为。
# 顺序：tty 依赖 termios（真 tty 会读 TCSAFLUSH 等常量），先占位 tty 才不会踩到
# 半成品 termios；termios/fcntl 各自独立。
for _posix_name, _posix_attrs in (
    ("tty", {"setraw": lambda *a, **k: None, "setcbreak": lambda *a, **k: None}),
    ("termios", {"tcgetattr": lambda *a, **k: [], "tcsetattr": lambda *a, **k: 0,
                 "TCSANOW": 0, "TCSAFLUSH": 0, "TCSADRAIN": 0,
                 "ICANON": 0, "ECHO": 0, "VMIN": 0, "VTIME": 0}),
    ("fcntl", {"ioctl": lambda *a, **k: 0, "flock": lambda *a, **k: None,
               "fcntl": lambda *a, **k: 0, "LOCK_EX": 0, "LOCK_SH": 0,
               "LOCK_UN": 0, "F_SETFL": 0, "F_GETFL": 0}),
):
    try:
        __import__(_posix_name)
    except Exception:  # noqa: BLE001 — ImportError 或半成品模块的 NameError 都要兜
        import types as _types

        _stub = _types.ModuleType(_posix_name)
        for _k, _v in _posix_attrs.items():
            setattr(_stub, _k, _v)
        sys.modules[_posix_name] = _stub

from aieyes.app import _make_scene  # noqa: E402
from aieyes.audio.output import AudioSink  # noqa: E402
from aieyes.audio.replay_source import ReplaySource, build_replay_playlist  # noqa: E402
from aieyes.audio.vad import EnergyVad  # noqa: E402
from aieyes.audio.wav import read_wav  # noqa: E402
from aieyes.clock import CLOCK_ID, mono_ms, wall_iso  # noqa: E402
from aieyes.config import AppConfig  # noqa: E402
from aieyes.errors import AudioDeviceError  # noqa: E402
from aieyes.events import EventBus  # noqa: E402
from aieyes.interfaces import PlaybackHandle  # noqa: E402
from aieyes.interaction.engine import InteractionEngine  # noqa: E402
from aieyes.logging_setup import JsonlEvidenceLogger  # noqa: E402
from aieyes.speech.asr import StreamingAsr  # noqa: E402
from aieyes.speech.frontend import SpeechFrontend  # noqa: E402
from aieyes.speech.kws import SherpaKws  # noqa: E402
from aieyes.speech.tts import PiperTts  # noqa: E402
from aieyes.vision.vlm import VlmClient  # noqa: E402
from aieyes.chat import ChatSession  # noqa: E402

SAMPLE_RATE = 16000


def _pick_existing(candidates):
    """返回第一个存在的候选路径；都不存在时返回首个候选（保留原有报错形态）。

    目标板目录全小写（config/、tests/fixtures/），本机历史上用 Config/、Tests/Fixtures/；
    两种布局都要能跑，所以默认值不接受硬编码的单一拼写。顺序 = 旧拼写优先，保持向后兼容。
    """
    for cand in candidates:
        if Path(cand).exists():
            return Path(cand)
    return Path(candidates[0])


WAKE = _pick_existing(
    [
        ROOT / "Tests/Fixtures/audio/wake/tts_xiaobing_nihao.wav",
        ROOT / "tests/fixtures/audio/wake/tts_xiaobing_nihao.wav",
    ]
)
QUESTION_DIR = _pick_existing(
    [ROOT / "Tests/Fixtures/audio/questions", ROOT / "tests/fixtures/audio/questions"]
)
# 只认 argv[0] 的程序名，避免把命令行里带 llama-server 字样的无关进程算进来
_SERVER_MARKER = b"llama-server"


def _silence(ms: int) -> np.ndarray:
    return np.zeros(int(SAMPLE_RATE * ms / 1000), dtype=np.float32)


def _cycle_audio(wake: np.ndarray, question: np.ndarray) -> np.ndarray:
    """一次完整循环：唤醒(约1.1s) + 2s 静音 + 问题 + 24s 静音 ≈ 30s。

    24s 静音覆盖「回答问题（流式，几秒）+ T17 倾听 10s + 待机」，保证下一轮唤醒词
    落在 KWS 重新开启后；22s 版本在长回答时会与 T17 窗口重叠，导致唤醒被当成普通语音。
    """
    return np.concatenate([wake, _silence(2000), question, _silence(24000)])


def _proc_kb(field: str) -> int | None:
    """读 /proc/self/status 的某个 kB 字段；缺字段/不可读/非法值返回 None。

    不返回 0：0 kB 与“没读到”是两件事，把后者当 0 会让峰值看起来更小。
    """
    try:
        text = Path("/proc/self/status").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if not line.startswith(field + ":"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            value = int(parts[1])
        except ValueError:
            return None
        return value if value >= 0 else None
    return None


def _iter_proc_pids(proc_root: str | Path = "/proc") -> list[int]:
    try:
        names = list(Path(proc_root).iterdir())
    except OSError:
        return []
    return sorted(int(p.name) for p in names if p.name.isdigit())


def _server_pids(proc_root: str | Path = "/proc") -> list[int]:
    """按 /proc/<pid>/cmdline 的 argv[0] 归属 llama-server 进程。

    不用 ``pgrep -f llama-server``：它匹配整条命令行，会把编辑器、grep、日志
    查看这类进程算成模型进程；这里只认程序名（argv[0] 的 basename）。
    """
    pids: list[int] = []
    for pid in _iter_proc_pids(proc_root):
        try:
            cmdline = (Path(proc_root) / str(pid) / "cmdline").read_bytes()
        except OSError:
            continue
        argv0 = cmdline.split(b"\0", 1)[0].replace(b"\\", b"/")
        if not argv0:
            continue
        if _SERVER_MARKER in argv0.rsplit(b"/", 1)[-1]:
            pids.append(pid)
    return sorted(pids)


def _server_peak_kb(proc_root: str | Path = "/proc") -> int | None:
    """所有 llama-server 进程 VmHWM 之和；一个都没找到或不可读 → None。

    旧实现用 ``pgrep -f`` 只取第一个 pid，既可能命中无关进程，也会漏掉多进程；
    失败时还静默返回 0，让峰值看起来达标。
    """
    pids = _server_pids(proc_root)
    if not pids:
        return None
    total = 0
    for pid in pids:
        try:
            text = (Path(proc_root) / str(pid) / "status").read_text()
        except OSError:
            return None
        for line in text.splitlines():
            if line.startswith("VmHWM:"):
                parts = line.split()
                if len(parts) < 2:
                    return None
                try:
                    total += int(parts[1])
                except ValueError:
                    return None
                break
        else:
            return None
    return total


class NullSink:
    """无声卡输出时的播放替身：按音频时长实时“播完”，供持续运行检查。

    只验证软件链路（TTS→播放→打断→T12 计时），不代表真实扬声器出声；
    阶段 8 报告中据此标注“回放输入 + null 输出”。
    """

    def __init__(self, *, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self._lock = threading.Lock()
        self._started = False
        self._playing = False
        self._finished: threading.Event | None = None
        self._piece_id = ""
        self._text = ""
        self._total_ms = 0
        self._t0_ms = 0
        self._played_ms = 0
        self.play_calls = 0
        self.cancel_calls = 0

    def start(self) -> None:
        self._started = True

    def stop(self) -> None:
        with self._lock:
            self._playing = False
            if self._finished is not None:
                self._finished.set()

    def close(self) -> None:
        self.stop()

    def play(
        self,
        pcm: np.ndarray,
        sample_rate: int = SAMPLE_RATE,
        *,
        piece_id: str = "",
        text: str = "",
    ) -> PlaybackHandle:
        samples = np.asarray(pcm, dtype=np.float32).reshape(-1)
        total_ms = int(round(samples.size * 1000 / sample_rate))
        finished = threading.Event()
        with self._lock:
            if self._finished is not None:
                self._finished.set()
            self.play_calls += 1
            pid = piece_id or f"null-{self.play_calls}"
            self._finished = finished
            self._piece_id = pid
            self._text = text
            self._total_ms = total_ms
            self._t0_ms = mono_ms()
            self._played_ms = 0
            self._playing = total_ms > 0
        if total_ms <= 0:
            finished.set()
        else:
            def _wait_done() -> None:
                if finished.wait(total_ms / 1000.0):
                    return
                with self._lock:
                    if self._finished is finished:
                        self._played_ms = self._total_ms
                        self._playing = False
                        self._finished = None
                        finished.set()

            threading.Thread(target=_wait_done, daemon=True).start()
        return PlaybackHandle(
            piece_id=pid,
            total_ms=total_ms,
            text=text,
            _cancel=lambda: self.cancel(pid),
            _wait=finished.wait,
        )

    def cancel(self, piece_id: str | None = None) -> bool:
        with self._lock:
            if piece_id is not None and piece_id != self._piece_id:
                return False
            if not self._playing:
                return False
            self.cancel_calls += 1
            self._played_ms = min(self._total_ms, mono_ms() - self._t0_ms)
            self._playing = False
            if self._finished is not None:
                self._finished.set()
                self._finished = None
            return True

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._playing

    def played_ms(self) -> int:
        with self._lock:
            if self._playing and self._finished is not None:
                return min(self._total_ms, mono_ms() - self._t0_ms)
            return self._played_ms

    def reference_frame(self) -> np.ndarray | None:
        return None


def _read_records(path: Path) -> list[dict]:
    records = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _naive_sum_gib(*kb_values) -> float | None:
    """RSS 朴素和换算成 GiB；任一分量未知即 None（不把缺测当 0 加起来）。"""
    if any(value is None for value in kb_values):
        return None
    return round(sum(kb_values) / 1024 / 1024, 3)


def _summarize(records: list[dict], elapsed_s: float, rss: dict, llama_peak_kb: int) -> dict:
    def find(category: str, event: str) -> list[dict]:
        return [r for r in records if r.get("category") == category and r.get("event") == event]

    wakes = find("wake", "detected")
    answers = [r for r in find("answer", "spoken") if r.get("played")]
    # 流式回答每个短语都会记 first_audio；只取每个回答的第一条（其余是累计播放时间）
    first_audio: list[int] = []
    pending = False
    for record in records:
        if record.get("category") == "speech" and record.get("event") == "worker_start":
            # 流式回答只有首短语带 ref_ms（worker_start.ms>0），其余短语 ref=None 会累计
            if record.get("speech_kind") == "answer" and float(record.get("ms", 0)) > 0:
                pending = True
        elif record.get("category") == "latency" and record.get("event") == "first_audio":
            if record.get("speech_kind") == "answer" and pending:
                first_audio.append(int(record.get("ms", 0)))
                pending = False
    capacity_refusals = [r for r in find("compare", "error") if "CapacityError" in str(r.get("error", ""))]
    failures = (
        find("t11", "timeout")
        + find("scene", "error")
        + find("llm", "error")
        + find("device", "fault")
        + [r for r in find("compare", "error") if r not in capacity_refusals]
    )
    return {
        "clock_id": CLOCK_ID,
        "started_iso": wall_iso(),
        "elapsed_s": round(elapsed_s, 1),
        "wake_count": len(wakes),
        "answer_count": len(answers),
        "failure_count": len(failures),
        "capacity_refusals": len(capacity_refusals),
        "first_audio_ms": {
            "n": len(first_audio),
            "p50": _pct(first_audio, 50),
            "p95": _pct(first_audio, 95),
            "max": max(first_audio) if first_audio else None,
        },
        "proc_rss_peak_mib": None if rss.get("peak_kb") is None else round(rss["peak_kb"] / 1024, 1),
        "server_peak_mib": None if llama_peak_kb is None else round(llama_peak_kb / 1024, 1),
        # RSS 朴素和：只是筛查量，既不含共享页去重，也不含 NvMap 与子进程，
        # 单独不能支撑“峰值 ≤ 4 GiB”的结论（见 stack_peak_gib_* 标签）。
        "stack_peak_gib": _naive_sum_gib(rss.get("peak_kb"), llama_peak_kb),
        "stack_peak_gib_method": "rss_naive_sum_screening_only",
        "stack_peak_gib_is_conclusion": False,
        "stack_peak_gib_scope": "probe_process_plus_llama_server_rss_only",
        "stack_peak_gib_caveats": [
            "no_shared_page_dedup_pss_not_used",
            "nvmap_gpu_buffers_not_included",
            "child_processes_of_probe_not_included",
            "swap_and_zram_policy_not_verified_here",
        ],
        "rss_sample_counts": {
            "samples": int(rss.get("samples", 0)),
            "missing_samples": int(rss.get("missing_samples", 0)),
        },
        "evidence_lines": len(records),
    }


def _pct(values: list[int], p: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((p / 100) * (len(ordered) - 1)))))
    return round(float(ordered[idx]), 1)


def _make_sink(mode: str, cfg: AppConfig):
    """auto：优先真实声卡，打不开则退回 NullSink（无声卡环境的持续检查）。"""
    if mode != "null":
        try:
            sink = AudioSink(sample_rate=cfg.audio.sample_rate)
            sink.start()
            sink.stop()
            print("sink=alsa(ok)")
            return sink, "alsa"
        except AudioDeviceError as exc:
            if mode == "alsa":
                raise
            print(f"warn: 输出设备不可用（{exc}）；改用 null sink（不出声）")
    return NullSink(sample_rate=cfg.audio.sample_rate), "null"


async def _run(args: argparse.Namespace) -> int:
    cfg = AppConfig.from_file(args.config)
    # 回放环境：场景问答走文件图，避免占用真实相机
    cfg.vision.backend = "file"
    cfg.vision.file_source = str(
        _pick_existing(
            [
                ROOT / "Tests/Fixtures/images/text_cn_price.png",
                ROOT / "tests/fixtures/images/text_cn_price.png",
            ]
        )
    )
    base = Path(cfg.evidence_dir)
    out = Path(args.out) if args.out else base / "soak" / f"soak_{int(time.time())}.json"
    evidence_path = base / "soak" / "soak.jsonl"
    evidence = JsonlEvidenceLogger(evidence_path)

    # 单图意图 + 聊天意图混用：文件图保证 scene 路径也走真实 OCR/VLM。
    qdir = Path(args.questions_dir) if args.questions_dir else QUESTION_DIR
    questions = sorted(qdir.glob("*.wav"))
    if not questions:
        print(f"缺少问题素材目录：{qdir}", file=sys.stderr)
        return 2
    wake = read_wav(WAKE, target_rate=SAMPLE_RATE)[0]
    question_arrays = [read_wav(p, target_rate=SAMPLE_RATE)[0] for p in questions]

    cycles = int(args.minutes * 60 / 30) + 1
    total = np.concatenate(
        [_cycle_audio(wake, question_arrays[i % len(question_arrays)]) for i in range(cycles)]
    )
    print(f"soak: {cycles} 轮，音频 {len(total) / SAMPLE_RATE / 60:.1f} 分钟，证据 {evidence_path}")

    source = ReplaySource([total], realtime=True)
    tts = PiperTts(cfg.tts)
    sink, sink_kind = _make_sink(args.sink, cfg)
    frontend = SpeechFrontend(
        source,
        kws=SherpaKws(cfg.kws),
        vad=EnergyVad(cfg.vad),
        asr=StreamingAsr(cfg.asr),
    )
    scene, scene_provider = _make_scene(cfg)
    chat = ChatSession(VlmClient(cfg.llm, cfg.vlm))
    engine = InteractionEngine(
        cfg=cfg,
        bus=EventBus(),
        frontend=frontend,
        sink=sink,
        tts=tts,
        chat=chat,
        scene=scene,
        identity=None,
        evidence=evidence,
    )

    rss = {"peak_kb": None, "samples": 0, "missing_samples": 0}
    stop_at = mono_ms() + int(args.minutes * 60 * 1000)
    started = mono_ms()

    sink.start()
    await engine.start()
    frontend.start()

    while mono_ms() < stop_at:
        await asyncio.sleep(15)
        cur = _proc_kb("VmRSS")
        if cur is None:
            rss["missing_samples"] += 1
        else:
            rss["samples"] += 1
            if rss["peak_kb"] is None or cur > rss["peak_kb"]:
                rss["peak_kb"] = cur
        # 让出事件循环，保持心跳可观测
        rss_text = "?" if cur is None else f"{cur / 1024:7.1f}"
        print(
            f"  t={int((mono_ms() - started) / 1000):4d}s rss={rss_text}MiB "
            f"wakes={len([r for r in _read_records(evidence_path) if r.get('category') == 'wake'])}"
        )

    frontend.stop()
    await engine.stop()
    sink.close()
    tts.close()
    scene.close()
    if scene_provider is not None:
        scene_provider.close()

    records = _read_records(evidence_path)
    summary = _summarize(records, (mono_ms() - started) / 1000, rss, _server_peak_kb())
    summary["config"] = {
        "minutes": args.minutes,
        "cycles": cycles,
        "backend": cfg.audio.backend,
        "sink": sink_kind,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="30 分钟持续交互检查（回放输入）")
    parser.add_argument(
        "--config",
        default=str(
            _pick_existing(
                [ROOT / "Config/target.json", ROOT / "config/target.json"]
            )
        ),
    )
    parser.add_argument("--minutes", type=float, default=30.0)
    parser.add_argument(
        "--sink",
        choices=["auto", "alsa", "null"],
        default="auto",
        help="auto=优先真实声卡，失败退回 null；null=不出声（无声卡环境）",
    )
    parser.add_argument("--questions-dir", default="", help="问题素材目录（默认 Tests/Fixtures/audio/questions，缺失时回退 tests/fixtures/audio/questions）")
    parser.add_argument("--out", default="")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
