"""阶段 7 性能探针：整栈常驻内存（分层口径）+ 单次真实调用时延（目标板运行）。

用法：
    trun "python Tools/mem_probe.py --config Config/target.json --image Tests/Fixtures/images/text_cn_price.png"

输出一行 JSON：各阶段 RSS、llama-server 与自身的内存账目、各阶段耗时。

内存口径（与仓库 ``Testbench/metrics.py`` 同一套语义，逐字段对齐）：

- ``conservative_upper_bound`` = ``sum(Rss) + sum(Swap) + NvMap``，**仅可筛查**，
  ``<= cap`` 不能宣布达标（共享页重复计、NvMap 可能与 PSS 重叠）。
- ``dedup_estimate`` = ``sum(Pss) + sum(SwapPss) + NvMap``，**仅可报告**，不是精确值；
  它 ``> cap`` 时超限结论成立（缺测只会让它更小）。
- 缺字段 / 进程退出 / NvMap 不可读 → ``undeterminable`` 并给出可定位的 reasons，
  **绝不返回 0，也绝不产出"通过"结论**（``verdict.is_pass_conclusion`` 恒为 False）。

本文件自足实现账目采集（不依赖 ``Testbench/``）：目标板交付清单里没有 Testbench，
探针必须在只有 ``src/`` + ``tools/`` 的板上也能给出可判读的口径。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 与 Tools/soak_test.py 同一约定：从仓库/交付根跑时把包目录挂上，这样本机
# Windows 与目标板都不必依赖外部 PYTHONPATH 才能 import aieyes。
# Linux 区分大小写：交付根里实际叫 ``src/``，历史写法 ``Src/`` 只在 Windows 命中，
# 所以两个都探，命中即用。
_ROOT = Path(__file__).resolve().parents[1]
for _pkg_parent in (_ROOT / "src", _ROOT / "Src"):
    if _pkg_parent.is_dir():
        if str(_pkg_parent) not in sys.path:
            sys.path.insert(0, str(_pkg_parent))
        break

DEFAULT_CAP_GIB = 4.0  # 验收口径 §八.1：首轮峰值 ≤ 4 GiB = 4 x 1024**3 B
_CAP_PEAK_BYTES = 4 * 1024 ** 3
_CAP_NORMAL_LOW_BYTES = 1610612736  # 1.5 GiB
_CAP_NORMAL_HIGH_BYTES = 2 * 1024 ** 3  # 2 GiB
_SERVER_MARKER = b"llama-server"
_READ_LIMIT = 1 << 20
_MEMORY_METHOD = "procfs_smaps_rollup"

_ROLES = ("worker", "model", "bench")

_SMAPS_ROLLUP_FIELDS = {
    "Rss": "rss_bytes",
    "Pss": "pss_bytes",
    "Pss_Anon": "pss_anon_bytes",
    "Pss_File": "pss_file_bytes",
    "Pss_Shmem": "pss_shmem_bytes",
    "Shared_Clean": "shared_clean_bytes",
    "Shared_Dirty": "shared_dirty_bytes",
    "Private_Clean": "private_clean_bytes",
    "Private_Dirty": "private_dirty_bytes",
    "Swap": "swap_bytes",
    "SwapPss": "swap_pss_bytes",
}
_SMAPS_REQUIRED = ("rss_bytes", "pss_bytes")
_SMAPS_OPTIONAL = ("swap_bytes", "swap_pss_bytes")

_GPU_MEM_CLIENT_DIRS = ("kernel/debug/nvmap/iovmm/clients",)
_GPU_MEM_TOTAL_FILES = (
    "kernel/debug/nvmap/iovmm/allocated",
    "kernel/debug/nvmap/allocated",
)

# 为什么 within_cap_upper_bound_only 只是筛查而不是达标
_VERDICT_CAVEATS = (
    "scope_is_sampled_process_tree_plus_nvmap_only",
    "processes_outside_the_sampled_tree_are_not_counted",
    "kernel_and_driver_allocations_outside_nvmap_are_not_counted",
    "page_cache_tmpfs_and_swap_zram_policy_are_not_verified_here",
    "nvmap_buffers_may_overlap_pss_anon_so_dedup_may_be_inflated",
)
_PASS_REQUIREMENTS = (
    "full_scope_process_inventory_on_target",
    "target_board_measurement_of_normal_and_peak",
    "swap_zram_policy_checked_not_relied_upon",
)


# --------------------------------------------------------------- 基础读取
def _now_ms() -> float:
    return time.monotonic() * 1000.0


def _read_text(path, limit=_READ_LIMIT):
    """Text content of ``path`` or None; unreadable files are never guessed."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(limit)
    except OSError:
        return None
    return raw.decode("utf-8", "replace")


def _read_number(path):
    text = _read_text(path)
    if text is None:
        return None
    parts = text.split()
    if not parts:
        return None
    try:
        value = int(parts[0], 10)
    except ValueError:
        return None
    return value if value >= 0 else None


def _coerce_pid(value):
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _read_status_kb(field, status_path="/proc/self/status"):
    """读 /proc/<pid>/status 的某个 kB 字段；缺失/不可读/非法一律 None（不编造 0）。"""
    text = _read_text(status_path)
    if text is None:
        return None
    for line in text.splitlines():
        if not line.startswith(field + ":"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            kib = int(parts[1])
        except ValueError:
            return None
        return kib if kib >= 0 else None
    return None


def _rss_kb(status_path="/proc/self/status"):
    return _read_status_kb("VmRSS", status_path)


def _peak_kb(status_path="/proc/self/status"):
    return _read_status_kb("VmHWM", status_path)


def _status_path(pid, proc_root="/proc"):
    return os.path.join(os.fspath(proc_root), str(pid), "status")


def _read_ppid(pid, proc_root="/proc"):
    """Field 4 (ppid) of ``/proc/<pid>/stat``, or None."""
    path = os.path.join(os.fspath(proc_root), str(pid), "stat")
    text = _read_text(path)
    if text is None:
        return None
    closing = text.rfind(")")
    if closing < 0:
        return None
    fields = text[closing + 1:].split()
    if len(fields) < 2:  # state, ppid
        return None
    try:
        ppid = int(fields[1])
    except ValueError:
        return None
    return ppid if ppid >= 0 else None


def _read_smaps_rollup_fields(pid, proc_root="/proc"):
    """``(values, missing_optional)`` from ``/proc/<pid>/smaps_rollup``.

    ``Rss``/``Pss`` 缺失 → 整体视为不可读（``None``），不补 0；``Swap``/``SwapPss``
    缺失记入 ``missing_optional``，由调用方把受影响的口径判成 undeterminable。
    """
    path = os.path.join(os.fspath(proc_root), str(pid), "smaps_rollup")
    text = _read_text(path)
    if text is None:
        return None, ()
    values = {}
    for line in text.splitlines():
        name, separator, rest = line.partition(":")
        if not separator:
            continue
        key = _SMAPS_ROLLUP_FIELDS.get(name.strip())
        if key is None:
            continue
        parts = rest.split()
        if not parts:
            continue
        try:
            kib = int(parts[0])
        except ValueError:
            return None, ()
        if kib < 0:
            return None, ()
        values[key] = kib * 1024
    for key in _SMAPS_REQUIRED:
        if key not in values:
            return None, ()
    missing = tuple(key for key in _SMAPS_OPTIONAL if key not in values)
    return values, missing


def _read_vmrss_bytes(pid, proc_root="/proc"):
    kib = _read_status_kb("VmRSS", _status_path(pid, proc_root))
    return None if kib is None else kib * 1024


def _read_vmswap_bytes(pid, proc_root="/proc"):
    """``VmSwap`` in bytes; 只有显式 ``VmSwap: 0 kB`` 才是 0，缺字段返回 None。"""
    kib = _read_status_kb("VmSwap", _status_path(pid, proc_root))
    return None if kib is None else kib * 1024


def _proc_table(proc_root):
    """``{pid: ppid}`` for every numeric entry of procfs; None if unreadable."""
    try:
        names = os.listdir(os.fspath(proc_root))
    except OSError:
        return None
    table = {}
    for name in names:
        if not name.isdigit():
            continue
        ppid = _read_ppid(name, proc_root)
        if ppid is None:
            continue
        table[int(name)] = ppid
    return table


def _normalize_roots(roots):
    """Map ``role -> pid`` (or pid list) to an ordered ``[(role, pid)]`` list."""
    if roots is None:
        return []
    if isinstance(roots, dict):
        ordered = [(role, roots[role]) for role in _ROLES if role in roots]
        ordered += [(k, v) for k, v in roots.items() if k not in _ROLES]
    else:
        ordered = [("root", roots)]
    pairs = []
    for role, value in ordered:
        if value is None:
            continue
        if isinstance(value, (str, bytes)) or not hasattr(value, "__iter__"):
            values = [value]
        else:
            values = list(value)
        for pid in values:
            coerced = _coerce_pid(pid)
            if coerced is not None:
                pairs.append((str(role), coerced))
    return pairs


def _tree_members(table, root_pairs, *, include_descendants=True):
    """Roots plus their whole descendant subtree, each pid claimed once."""
    claimed = {}
    for role, pid in root_pairs:
        claimed.setdefault(pid, (role, pid, 0))
    if include_descendants and table:
        children = {}
        for pid, ppid in table.items():
            children.setdefault(ppid, []).append(pid)
        for role, pid in root_pairs:
            pending = [pid]
            while pending:
                current = pending.pop(0)
                for child in sorted(children.get(current, ())):
                    if child in claimed:
                        continue
                    depth = claimed[current][2] + 1
                    claimed[child] = (role, pid, depth)
                    pending.append(child)
    members = []
    for pid, (role, role_root, depth) in claimed.items():
        members.append(
            {"pid": pid, "role": role, "role_root": role_root, "depth": depth}
        )
    members.sort(key=lambda item: (item["role"], item["depth"], item["pid"]))
    return members


def _read_gpu_mem_bytes(sys_root):
    """NvMap GPU buffer bytes: ``(bytes, backend, reason)``; ``allocated`` 按字节。"""
    for relative in _GPU_MEM_CLIENT_DIRS:
        root = os.path.join(os.fspath(sys_root), relative)
        try:
            names = os.listdir(root)
        except OSError:
            continue
        total = 0
        found = 0
        for name in sorted(names):
            raw = _read_number(os.path.join(root, name, "allocated"))
            if raw is None:
                continue
            total += int(raw)
            found += 1
        if found:
            return total, relative, None
    for relative in _GPU_MEM_TOTAL_FILES:
        raw = _read_number(os.path.join(os.fspath(sys_root), relative))
        if raw is not None:
            return int(raw), relative, None
    return None, None, "nvmap_unreadable_or_absent"


# --------------------------------------------------------------- 进程归属
def _server_pids(proc_root="/proc", marker=_SERVER_MARKER):
    """按 ``/proc/<pid>/cmdline`` 的 argv[0] 归属 llama-server。

    只认程序名，不解析 ``ps -eo rss,cmd`` 的文本行：那样会把命令行里带
    llama-server 字样的无关进程（编辑器、grep、日志查看）算进来。
    """
    pids = []
    try:
        entries = list(os.listdir(os.fspath(proc_root)))
    except OSError:
        return pids
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(os.path.join(os.fspath(proc_root), name, "cmdline"), "rb") as fh:
                cmdline = fh.read(_READ_LIMIT)
        except OSError:
            continue
        argv0 = cmdline.split(b"\0", 1)[0].replace(b"\\", b"/")
        if not argv0:
            continue
        if marker in argv0.rsplit(b"/", 1)[-1]:
            pids.append(int(name))
    return sorted(pids)


def _sum_status_kb(pids, field, proc_root="/proc"):
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


def _server_rss_kb(proc_root="/proc"):
    return _sum_status_kb(_server_pids(proc_root), "VmRSS", proc_root)


def _server_peak_kb(proc_root="/proc"):
    """llama-server 的 VmHWM 之和（/proc/<pid>/status，多进程累加）。"""
    return _sum_status_kb(_server_pids(proc_root), "VmHWM", proc_root)


def _scope_reasons(server_pids, proc_root="/proc"):
    """范围不全的原因：模型进程没找到时，采样树只覆盖探针自己。"""
    if server_pids:
        return []
    return [
        {
            "component": "model",
            "pid": None,
            "path": os.fspath(proc_root),
            "reason": "llama_server_not_found_scope_incomplete",
        }
    ]


def _screening_flag(verdict_state, scope_reasons=()):
    """筛选项：只在范围完整且口径明确时给 True/False，其余一律 None。"""
    if scope_reasons:
        return None
    if verdict_state == "within_cap_upper_bound_only":
        return True
    if verdict_state == "over_cap":
        return False
    return None


# --------------------------------------------------------------- 分层账目
def unavailable_memory(reason, *, extra_reasons=()):
    """Memory block that carries *no* numbers, only the reason it has none."""
    reasons = [{"component": "accounting", "pid": None, "path": None, "reason": reason}]
    for item in extra_reasons:
        reasons.append(dict(item))
    return {
        "method": _MEMORY_METHOD,
        "accounting": "undeterminable",
        "complete": False,
        "roots": {},
        "process_count": 0,
        "processes": [],
        "sampling_gaps": [],
        "notes": [],
        "components": {
            "rss_sum_bytes": None,
            "pss_sum_bytes": None,
            "swap_sum_bytes": None,
            "swap_pss_sum_bytes": None,
            "nvmap_bytes": None,
            "nvmap_backend": None,
        },
        "conservative_upper_bound": {
            "label": "conservative_upper_bound",
            "state": "undeterminable",
            "bytes": None,
            "formula": "sum(Rss) + sum(Swap) + NvMap",
            "overlap": True,
            "usable_for": "screening_only_not_for_pass_or_exceed_conclusion",
        },
        "dedup_estimate": {
            "label": "dedup_estimate",
            "state": "undeterminable",
            "bytes": None,
            "formula": "sum(Pss) + sum(SwapPss) + NvMap",
            "error_sources": [],
            "usable_for": "report_only_not_a_pass_or_exceed_conclusion",
        },
        "undeterminable": {
            "label": "undeterminable",
            "applicable": True,
            "reasons": reasons,
            "usable_for": "must_be_resolved_on_target_before_any_memory_claim",
        },
        "verdict": {
            "state": "undeterminable",
            "reason": reason,
            "cap_bytes": _CAP_PEAK_BYTES,
            "is_pass_conclusion": False,
            "is_exceed_conclusion": False,
            "scope": "sampled_process_tree_plus_nvmap_only",
            "caveats": _VERDICT_CAVEATS,
            "requires_for_pass": list(_PASS_REQUIREMENTS),
            "normal_range": {
                "low_bytes": _CAP_NORMAL_LOW_BYTES,
                "high_bytes": _CAP_NORMAL_HIGH_BYTES,
                "dedup_position": "unknown",
                "dedup_position_meaning": "estimate_not_exact_value",
            },
        },
        "sampling_window_ms": 0,
    }


def collect_memory_accounting(
    roots=None,
    *,
    proc_root="/proc",
    sys_root="/sys",
    include_descendants=True,
    gpu_mem_required=True,
    cap_bytes=_CAP_PEAK_BYTES,
):
    """Tiered memory accounting for a process tree.  Never raises.

    ``roots`` 是 ``{role: pid}``（值也可以是列表）。只统计这些根及其全部后代，
    共享 pid 只计一次，无关进程一律不计入。
    """
    window_start = _now_ms()
    try:
        proc_root = os.fspath(proc_root)
        sys_root = os.fspath(sys_root)
    except TypeError:
        return unavailable_memory("bad_proc_or_sys_root")
    root_pairs = _normalize_roots(roots)
    if not root_pairs:
        return unavailable_memory("no_roots")

    reasons = []
    gaps = []
    try:
        table = _proc_table(proc_root) if include_descendants else {}
    except Exception:  # noqa: BLE001 - 观测工具，宁可不可判定也不崩
        table = None
    if table is None:
        reasons.append(
            {
                "component": "procfs",
                "pid": None,
                "path": proc_root,
                "reason": "procfs_unreadable",
            }
        )
        table = {}
    members = _tree_members(table, root_pairs, include_descendants=include_descendants)

    processes = []
    rss_sum = 0
    pss_sum = 0
    swap_sum = 0
    swap_pss_sum = 0
    rss_complete = True
    pss_complete = True
    swap_complete = True
    swap_pss_complete = True

    for member in members:
        pid = member["pid"]
        entry = {
            "pid": pid,
            "role": member["role"],
            "role_root": member["role_root"],
            "depth": member["depth"],
            "rss_bytes": None,
            "pss_bytes": None,
            "swap_bytes": None,
            "swap_pss_bytes": None,
            "state": "exited",
        }
        rollup, missing = _read_smaps_rollup_fields(pid, proc_root)
        if rollup is not None:
            entry["state"] = "measured"
            entry["rss_bytes"] = rollup.get("rss_bytes")
            entry["pss_bytes"] = rollup.get("pss_bytes")
            entry["swap_bytes"] = rollup.get("swap_bytes")
            entry["swap_pss_bytes"] = rollup.get("swap_pss_bytes")
            entry["missing_optional_fields"] = list(missing)
            rss_sum += rollup.get("rss_bytes") or 0
            pss_sum += rollup.get("pss_bytes") or 0
            if "swap_bytes" in missing:
                swap_complete = False
                gaps.append(
                    {
                        "pid": pid,
                        "component": "swap",
                        "path": os.path.join(proc_root, str(pid), "smaps_rollup"),
                        "reason": "smaps_rollup_missing_swap_field",
                    }
                )
            else:
                swap_sum += rollup.get("swap_bytes") or 0
            if "swap_pss_bytes" in missing:
                swap_pss_complete = False
                gaps.append(
                    {
                        "pid": pid,
                        "component": "swap_pss",
                        "path": os.path.join(proc_root, str(pid), "smaps_rollup"),
                        "reason": "smaps_rollup_missing_swap_pss_field",
                    }
                )
            else:
                swap_pss_sum += rollup.get("swap_pss_bytes") or 0
        else:
            rss_bytes = _read_vmrss_bytes(pid, proc_root)
            if rss_bytes is not None:
                swap_bytes = _read_vmswap_bytes(pid, proc_root)
                entry["state"] = "rss_only"
                entry["rss_bytes"] = rss_bytes
                entry["swap_bytes"] = swap_bytes
                pss_complete = False
                swap_pss_complete = False
                rss_sum += rss_bytes
                if swap_bytes is None:
                    swap_complete = False
                    gaps.append(
                        {
                            "pid": pid,
                            "component": "swap",
                            "path": os.path.join(proc_root, str(pid), "status"),
                            "reason": "vmswap_field_missing_or_unreadable",
                        }
                    )
                else:
                    swap_sum += swap_bytes
                gaps.append(
                    {
                        "pid": pid,
                        "component": "pss",
                        "path": os.path.join(proc_root, str(pid), "smaps_rollup"),
                        "reason": "smaps_rollup_unreadable",
                    }
                )
            else:
                entry["state"] = "exited"
                rss_complete = False
                pss_complete = False
                swap_pss_complete = False
                swap_complete = False
                gaps.append(
                    {
                        "pid": pid,
                        "component": "process",
                        "path": os.path.join(proc_root, str(pid)),
                        "reason": "exited_during_sample",
                    }
                )
        processes.append(entry)

    nvmap_bytes, nvmap_backend, nvmap_reason = _read_gpu_mem_bytes(sys_root)
    notes = []
    if nvmap_bytes is None:
        if gpu_mem_required:
            reasons.append(
                {
                    "component": "gpu_nvmap",
                    "pid": None,
                    "path": os.path.join(sys_root, _GPU_MEM_CLIENT_DIRS[0]),
                    "reason": nvmap_reason,
                }
            )
        else:
            nvmap_bytes = 0
            nvmap_backend = "not_required_cpu_only_host"
            notes.append("gpu_mem_not_required_nvmap_counted_as_zero")
    for gap in gaps:
        reasons.append(gap)

    components = {
        "rss_sum_bytes": rss_sum if rss_complete else None,
        "pss_sum_bytes": pss_sum if pss_complete else None,
        "swap_sum_bytes": swap_sum if swap_complete else None,
        "swap_pss_sum_bytes": swap_pss_sum if swap_pss_complete else None,
        "nvmap_bytes": nvmap_bytes,
        "nvmap_backend": nvmap_backend,
    }

    upper_state = "computed"
    upper_bytes = None
    if rss_complete and swap_complete and nvmap_bytes is not None:
        upper_bytes = rss_sum + swap_sum + nvmap_bytes
    else:
        upper_state = "undeterminable"

    dedup_state = "computed"
    dedup_bytes = None
    if pss_complete and swap_pss_complete and nvmap_bytes is not None:
        dedup_bytes = pss_sum + swap_pss_sum + nvmap_bytes
    else:
        dedup_state = "undeterminable"

    error_sources = [
        "pss_shared_with_processes_outside_sampled_tree_is_under_counted",
        "nvmap_buffers_may_also_be_mapped_in_pss_anon",
        "kernel_and_driver_allocations_outside_nvmap_not_covered",
    ]
    if include_descendants:
        error_sources.append("descendants_of_roots_only_other_processes_excluded")

    normal_position = "unknown"
    if dedup_bytes is not None:
        if dedup_bytes < _CAP_NORMAL_LOW_BYTES:
            normal_position = "below_normal_range"
        elif dedup_bytes <= _CAP_NORMAL_HIGH_BYTES:
            normal_position = "within_normal_range"
        else:
            normal_position = "above_normal_range"

    cap_bytes = int(cap_bytes)
    # 判定顺序（诚实性优先）：去重超限 > 任何缺测 > 上界≤cap（仅筛查）> 重叠带 > 不可判定
    if dedup_state == "computed" and dedup_bytes > cap_bytes:
        verdict_state = "over_cap"
        verdict_reason = "dedup_estimate_exceeds_cap"
    elif reasons:
        verdict_state = "undeterminable"
        verdict_reason = reasons[0]["reason"]
    elif upper_state == "computed" and upper_bytes <= cap_bytes:
        verdict_state = "within_cap_upper_bound_only"
        verdict_reason = "conservative_upper_bound_inside_cap_screening_only"
    elif (
        upper_state == "computed"
        and dedup_state == "computed"
        and upper_bytes > cap_bytes >= dedup_bytes
    ):
        verdict_state = "in_indeterminate_band"
        verdict_reason = "overlap_band_upper_above_cap_dedup_below_cap"
    else:
        verdict_state = "undeterminable"
        verdict_reason = "aggregate_state_undeterminable"

    return {
        "method": _MEMORY_METHOD,
        "accounting": "tiered",
        "complete": not reasons,
        "roots": {role: pid for role, pid in root_pairs},
        "process_count": len(processes),
        "processes": processes,
        "sampling_gaps": gaps,
        "notes": notes,
        "components": components,
        "conservative_upper_bound": {
            "label": "conservative_upper_bound",
            "state": upper_state,
            "bytes": upper_bytes,
            "formula": "sum(Rss) + sum(Swap) + NvMap",
            "overlap": True,
            "usable_for": "screening_only_not_for_pass_or_exceed_conclusion",
        },
        "dedup_estimate": {
            "label": "dedup_estimate",
            "state": dedup_state,
            "bytes": dedup_bytes,
            "formula": "sum(Pss) + sum(SwapPss) + NvMap",
            "error_sources": error_sources,
            "usable_for": "report_only_not_a_pass_or_exceed_conclusion",
        },
        "undeterminable": {
            "label": "undeterminable",
            "applicable": bool(reasons),
            "reasons": reasons,
            "usable_for": "must_be_resolved_on_target_before_any_memory_claim",
        },
        "verdict": {
            "state": verdict_state,
            "reason": verdict_reason,
            "cap_bytes": cap_bytes,
            "is_pass_conclusion": False,
            "is_exceed_conclusion": verdict_state == "over_cap",
            "scope": "sampled_process_tree_plus_nvmap_only",
            "caveats": _VERDICT_CAVEATS,
            "requires_for_pass": list(_PASS_REQUIREMENTS),
            "normal_range": {
                "low_bytes": _CAP_NORMAL_LOW_BYTES,
                "high_bytes": _CAP_NORMAL_HIGH_BYTES,
                "dedup_position": normal_position,
                "dedup_position_meaning": "estimate_not_exact_value",
            },
        },
        "sampling_window_ms": max(0, _now_ms() - window_start),
    }


def _mib(value):
    return None if value is None else round(value / 1024.0, 1)


def _gib(value):
    return None if value is None else round(value / float(1024 ** 3), 3)


def _pick_existing(candidates):
    """按顺序返回第一个存在的候选路径；都不存在时返回首个候选（保留原有报错形态）。

    目标板目录全小写（config/、tests/fixtures/），本机历史上用 Config/、Tests/Fixtures/；
    两种布局都要能跑，所以默认值不接受硬编码的单一拼写。顺序 = 旧拼写优先，保持向后兼容。
    """
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return candidates[0]


def main(argv=None) -> int:
    import numpy as np
    from PIL import Image

    from aieyes.clock import mono_ms
    from aieyes.config import AppConfig
    from aieyes.interfaces import AudioFrame, FRAME_SAMPLES, SAMPLE_RATE

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
    parser.add_argument("--cap-gib", type=float, default=DEFAULT_CAP_GIB)
    args = parser.parse_args(argv)

    proc_root = os.environ.get("AIEYES_PROC_ROOT") or "/proc"
    sys_root = os.environ.get("AIEYES_SYS_ROOT") or "/sys"
    # 纯 CPU 宿主（无 nvmap debugfs）显式声明不需要 GPU 内存；目标板必须保持 True
    gpu_mem_required = os.environ.get("AIEYES_GPU_MEM_REQUIRED", "1") not in ("0", "false", "False")

    cfg = AppConfig.from_file(args.config)
    image_path = Path(args.image)
    marks: dict[str, float | None] = {}
    timings: dict[str, float] = {}

    def mark(name: str) -> None:
        marks[name] = _mib(_rss_kb(_status_path(os.getpid(), proc_root)))

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
    timed(
        "speaker_embed",
        lambda: speaker.embed(np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE),
    )

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

    # ---- 内存账目：分层口径（见模块 docstring）
    server_pids = _server_pids(proc_root)
    roots = {"worker": os.getpid()}
    if server_pids:
        roots["model"] = server_pids
    accounting = collect_memory_accounting(
        roots,
        proc_root=proc_root,
        sys_root=sys_root,
        gpu_mem_required=gpu_mem_required,
        cap_bytes=int(round(args.cap_gib * 1024 ** 3)),
    )
    verdict = accounting["verdict"]
    scope_reasons = _scope_reasons(server_pids, proc_root)
    screening = _screening_flag(verdict["state"], scope_reasons)

    server_rss = _server_rss_kb(proc_root)
    server_peak = _server_peak_kb(proc_root)
    process_kb = _rss_kb(_status_path(os.getpid(), proc_root))
    process_peak_kb = _peak_kb(_status_path(os.getpid(), proc_root))

    # RSS 朴素和：仅筛查用，缺项即 None（不补 0，也不当结论）
    stack_rss_gib = None
    if process_kb is not None and server_rss is not None:
        stack_rss_gib = _gib((process_kb + server_rss) * 1024)
    stack_peak_gib = None
    if process_peak_kb is not None and server_peak is not None:
        stack_peak_gib = _gib((process_peak_kb + server_peak) * 1024)

    result = {
        "process_rss_mib": _mib(process_kb),
        "process_peak_mib": _mib(process_peak_kb),
        "llama_server_rss_mib": _mib(server_rss),
        "llama_server_peak_mib": _mib(server_peak),
        # 旧键名保留，但都带口径标签：RSS 朴素和只是筛查量
        "stack_total_gib": stack_rss_gib,
        "stack_peak_gib": stack_peak_gib,
        "stack_total_gib_method": "rss_naive_sum_screening_only",
        "stack_peak_gib_method": "rss_naive_sum_screening_only",
        "stack_total_gib_is_conclusion": False,
        "stack_peak_gib_is_conclusion": False,
        "stack_upper_gib": _gib(accounting["conservative_upper_bound"]["bytes"]),
        "stack_dedup_gib": _gib(accounting["dedup_estimate"]["bytes"]),
        "gpu_mem_bytes": accounting["components"]["nvmap_bytes"],
        "gpu_mem_backend": accounting["components"]["nvmap_backend"],
        "cap_gib": args.cap_gib,
        "cap_bytes": verdict["cap_bytes"],
        "accounting_method": accounting["method"],
        "accounting_complete": accounting["complete"],
        "process_count": accounting["process_count"],
        "llama_server_pids": server_pids,
        "verdict_state": verdict["state"],
        "within_cap_screening": screening,
        "within_cap_is_conclusion": False,
        "within_cap_scope": verdict["scope"],
        "scope_incomplete_reasons": scope_reasons,
        "pass_requires": list(verdict["requires_for_pass"]),
        "memory_verdict": verdict,
        "memory_accounting": accounting,
        "memory_labels": {
            "conservative_upper_bound": accounting["conservative_upper_bound"]["usable_for"],
            "dedup_estimate": accounting["dedup_estimate"]["usable_for"],
            "undeterminable": accounting["undeterminable"]["usable_for"],
            "within_cap_screening": "screening_only_not_a_pass_conclusion",
            "stack_total_gib": "rss_naive_sum_screening_only",
            "stack_peak_gib": "rss_naive_sum_screening_only",
        },
        "marks_mib": {k: v for k, v in marks.items()},
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "answer": getattr(answer, "text", ""),
        "answer_sources": [str(t) for t in (getattr(answer, "source_tags", ()) or ())],
        "answer_intent": getattr(answer, "intent", ""),
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
