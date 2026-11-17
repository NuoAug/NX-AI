"""阶段 7 性能探针：整栈常驻内存 + 单次真实调用时延（目标板运行）。

用法：
    trun "python Tools/mem_probe.py --config Config/target.json --image Tests/Fixtures/images/text_cn_price.png"

输出一行 JSON：各阶段 RSS/VmHWM、llama-server RSS、总占用、各阶段耗时。
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path

# Linux 区分大小写：交付根里包目录实际叫 ``src/``，历史写法 ``Src/`` 只在 Windows
# （不区分大小写）能命中。两个都探，命中即用，这样板上 ``python tools/mem_probe.py``
# 不必额外设 PYTHONPATH 才能 import aieyes（与 Tools/soak_test.py 同一约定）。
_ROOT = Path(__file__).resolve().parents[1]
for _pkg_parent in (_ROOT / "src", _ROOT / "Src"):
    if _pkg_parent.is_dir():
        if str(_pkg_parent) not in sys.path:
            sys.path.insert(0, str(_pkg_parent))
        break

import numpy as np
from PIL import Image

from aieyes.clock import mono_ms
from aieyes.config import AppConfig
from aieyes.interfaces import AudioFrame, FRAME_SAMPLES, SAMPLE_RATE


DEFAULT_CAP_GIB = 4.0  # 验收口径 §八.1：首轮峰值 ≤ 4 GiB = 4 x 1024**3 B
_SERVER_MARKER = b"llama-server"


def _status_path(pid: int, proc_root: str | Path = "/proc") -> Path:
    return Path(proc_root) / str(pid) / "status"


def _read_status_kb(
    field: str, status_path: str | Path = "/proc/self/status"
) -> int | None:
    """读 /proc/<pid>/status 的某个 kB 字段。

    字段缺失、文件不可读或值非法一律返回 None —— 不返回 0，避免把“没测到”
    伪装成“占用 0”。
    """
    try:
        with open(status_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith(field + ":"):
                    parts = line.split()
                    if len(parts) < 2:
                        return None
                    try:
                        return int(parts[1])
                    except ValueError:
                        return None
    except OSError:
        return None
    return None


def _rss_kb(status_path: str | Path = "/proc/self/status") -> int | None:
    return _read_status_kb("VmRSS", status_path)


def _peak_kb(status_path: str | Path = "/proc/self/status") -> int | None:
    return _read_status_kb("VmHWM", status_path)


def _server_pids(
    proc_root: str | Path = "/proc", marker: bytes = _SERVER_MARKER
) -> list[int]:
    """按 /proc/<pid>/cmdline 的 argv[0] 归属 llama-server 进程。

    只认 argv[0]（去掉目录后的程序名），不再解析 ``ps -eo rss,cmd`` 的文本行：
    那种做法会把命令行里带 llama-server 字样的无关进程（编辑器、grep、日志查看）
    算进来；进程树内的子进程由 collect_memory_accounting 的 BFS 归属覆盖。
    """
    pids: list[int] = []
    try:
        entries = list(Path(proc_root).iterdir())
    except OSError:
        return pids
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        argv0 = cmdline.split(b"\0", 1)[0].replace(b"\\", b"/")
        if not argv0:
            continue
        if marker in argv0.rsplit(b"/", 1)[-1]:
            pids.append(int(entry.name))
    return sorted(pids)


def _sum_status_kb(
    pids: list[int], field: str, proc_root: str | Path = "/proc"
) -> int | None:
    """多个 pid 的同一 status 字段求和；任一进程不可读则整体不可判定（None）。"""
    if not pids:
        return None
    total = 0
    for pid in pids:
        value = _read_status_kb(field, _status_path(pid, proc_root))
        if value is None:
            return None
        total += value
    return total


def _server_rss_kb(proc_root: str | Path = "/proc") -> int | None:
    return _sum_status_kb(_server_pids(proc_root), "VmRSS", proc_root)


def _server_peak_kb(proc_root: str | Path = "/proc") -> int | None:
    """llama-server 的 VmHWM 之和（/proc/<pid>/status，多进程累加）。"""
    return _sum_status_kb(_server_pids(proc_root), "VmHWM", proc_root)


def _load_metrics_module():
    """复用 Testbench/metrics.py 的分层内存口径。

    返回 (module, 失败原因)。导入失败时调用方按 undeterminable 处理，
    不要退回“只加 RSS”的旧算法。
    """
    try:
        return importlib.import_module("metrics"), None
    except Exception as exc:  # noqa: BLE001 - 记录原因后重试一次
        first = f"import metrics: {type(exc).__name__}: {exc}"
    bench_dir = Path(__file__).resolve().parents[1] / "Testbench"
    if bench_dir.is_dir() and str(bench_dir) not in sys.path:
        sys.path.insert(0, str(bench_dir))
    try:
        return importlib.import_module("metrics"), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{first}; retry after sys.path insert: {type(exc).__name__}: {exc}"


def _scope_reasons(server_pids, proc_root: str | Path = "/proc") -> list[dict]:
    """范围不全的原因：模型进程没找到时，采样树只覆盖探针自己。"""
    if server_pids:
        return []
    return [
        {
            "component": "model",
            "pid": None,
            "path": str(proc_root),
            "reason": "llama_server_not_found_scope_incomplete",
        }
    ]


def _screening_flag(
    verdict_state: str | None, scope_reasons: list[dict] | tuple = ()
):
    """筛选项：True/False 只在范围完整且口径明确时给出，其余一律 None。"""
    if scope_reasons:
        return None
    if verdict_state == "within_cap_upper_bound_only":
        return True
    if verdict_state == "over_cap":
        return False
    return None


def _memory_accounting(
    roots: dict[str, int],
    *,
    proc_root: str | Path = "/proc",
    sys_root: str | Path = "/sys",
    cap_gib: float = DEFAULT_CAP_GIB,
):
    """调用分层口径，返回 (accounting | None, verdict, module, reason)。"""
    module, reason = _load_metrics_module()
    if module is None:
        return None, {"state": "undeterminable", "reasons": [{"component": "metrics_module", "reason": reason}]}, None, reason
    cap_bytes = int(cap_gib * 1024 ** 3)
    try:
        account = module.collect_memory_accounting(
            roots, proc_root=str(proc_root), sys_root=str(sys_root), cap_bytes=cap_bytes
        )
    except Exception as exc:  # noqa: BLE001
        reason = f"collect_memory_accounting: {type(exc).__name__}: {exc}"
        return None, {"state": "undeterminable", "reasons": [{"component": "accounting", "reason": reason}]}, module, reason
    return account, account.get("verdict", {}), module, None


def _mib(kb: int | None) -> float | None:
    return None if kb is None else round(kb / 1024.0, 1)


def _gib(value: int | None) -> float | None:
    return None if value is None else round(value / float(1024 ** 3), 3)


def _pick_existing(candidates: list[str]) -> str:
    """按顺序返回第一个存在的候选路径；都不存在时返回首个候选（保留原有报错形态）。

    目标板目录全小写（config/、tests/fixtures/），本机历史上用 Config/、Tests/Fixtures/；
    两种布局都要能跑，所以默认值不接受硬编码的单一拼写。顺序 = 旧拼写优先，保持向后兼容。
    """
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=_pick_existing(["Config/target.json", "config/target.json"]),
    )
    parser.add_argument(
        "--image",
        default=_pick_existing(
            [
                "Tests/Fixtures/images/text_cn_price.png",
                "tests/fixtures/images/text_cn_price.png",
            ]
        ),
    )
    parser.add_argument(
        "--image2",
        default=_pick_existing(
            [
                "Tests/Fixtures/images/scene_red_square.png",
                "tests/fixtures/images/scene_red_square.png",
            ]
        ),
    )
    args = parser.parse_args(argv)

    cfg = AppConfig.from_file(args.config)
    image_path = Path(args.image)
    marks: dict[str, float | None] = {}
    timings: dict[str, float] = {}
    proc_root = os.environ.get("AIEYES_PROC_ROOT", "/proc")
    sys_root = os.environ.get("AIEYES_SYS_ROOT", "/sys")
    probe_pid = os.getpid()

    def mark(name: str) -> None:
        rss_kb = _rss_kb(_status_path(probe_pid, proc_root))
        # 读不到就记 None（不可判定），不写 0
        marks[name] = None if rss_kb is None else rss_kb / 1024.0

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

    server_pids = _server_pids(proc_root)
    process_kb = _rss_kb(_status_path(probe_pid, proc_root))
    process_peak_kb = _peak_kb(_status_path(probe_pid, proc_root))
    server_kb = _server_rss_kb(proc_root)
    server_peak_kb = _server_peak_kb(proc_root)

    roots: dict[str, int] = {"probe": probe_pid}
    for pid in server_pids:
        roots[f"llama_server:{pid}"] = pid
    account, verdict, metrics_mod, account_reason = _memory_accounting(
        roots, proc_root=proc_root, sys_root=sys_root
    )
    components = (account or {}).get("components", {})
    upper_bytes = (account or {}).get("conservative_upper_bound", {}).get("bytes")
    dedup_bytes = (account or {}).get("dedup_estimate", {}).get("bytes")
    nvmap_bytes = components.get("nvmap_bytes")
    cap_bytes = int(DEFAULT_CAP_GIB * 1024 ** 3)

    # 峰值上界：树内各进程 VmHWM 之和 + NvMap。任一分量缺失即不可判定。
    tree_pids = [p["pid"] for p in (account or {}).get("processes", [])]
    peak_tree_kb = _sum_status_kb(tree_pids, "VmHWM", proc_root) if tree_pids else None
    peak_upper_bytes = (
        None
        if peak_tree_kb is None or nvmap_bytes is None
        else peak_tree_kb * 1024 + nvmap_bytes
    )

    verdict_state = verdict.get("state")
    scope_reasons = _scope_reasons(server_pids, proc_root)
    # within_cap 是“上界筛查”结果，不是达标结论：树外进程、nvmap 之外的内核
    # 分配、页面缓存与 swap/zram 策略都不在这里的观测范围内；模型进程没纳管时
    # 连筛查也不给（None），避免拿半棵树的数据去比 cap。
    within_cap = _screening_flag(verdict_state, scope_reasons)
    within_cap_peak = (
        None
        if (peak_upper_bytes is None or scope_reasons)
        else peak_upper_bytes <= cap_bytes
    )

    result = {
        "process_rss_mib": _mib(process_kb),
        "process_peak_mib": _mib(process_peak_kb),
        "llama_server_rss_mib": _mib(server_kb),
        "llama_server_peak_mib": _mib(server_peak_kb),
        "stack_total_gib": _gib(upper_bytes),
        "stack_peak_gib": _gib(peak_upper_bytes),
        "cap_gib": DEFAULT_CAP_GIB,
        "within_cap": within_cap,
        "within_cap_peak": within_cap_peak,
        "marks_mib": {k: (None if v is None else round(v, 1)) for k, v in marks.items()},
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "answer": getattr(answer, "text", ""),
        "answer_sources": [str(t) for t in (getattr(answer, "source_tags", ()) or ())],
        "answer_intent": getattr(answer, "intent", ""),
        # --- 以下为新增字段（只增不改）：口径标签、去重估计与不可判定原因 ---
        "stack_upper_gib": _gib(upper_bytes),
        "stack_dedup_gib": _gib(dedup_bytes),
        "cap_bytes": cap_bytes,
        "llama_server_pids": server_pids,
        "gpu_mem_bytes": nvmap_bytes,
        "accounting_method": (account or {}).get("method", "unavailable"),
        "accounting_complete": (account or {}).get("complete"),
        "accounting_reason": account_reason,
        "memory_verdict": verdict,
        "memory_accounting": account,
        "within_cap_scope": "conservative_upper_bound_sampled_tree_only",
        "within_cap_is_conclusion": False,
        "within_cap_peak_scope": "rss_hwm_upper_bound_sampled_tree_only",
        "verdict_state": verdict_state,
        "scope_incomplete_reasons": scope_reasons,
        "pass_requires": list((verdict or {}).get("requires_for_pass", ())),
        "memory_labels": {
            "stack_total_gib": "conservative_upper_bound",
            "stack_upper_gib": "conservative_upper_bound",
            "stack_peak_gib": "rss_hwm_upper_bound",
            "stack_dedup_gib": "dedup_estimate",
            "within_cap": "upper_bound_inside_cap_screening_only_not_a_pass_conclusion",
            "within_cap_peak": "peak_upper_bound_inside_cap_screening_only_not_a_pass_conclusion",
            "null_semantics": "undeterminable_not_zero",
        },
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
