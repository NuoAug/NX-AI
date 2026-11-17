"""Read-only resource metrics sampling for the AI Eyes test bench.

Dependency-free (Python 3.12 standard library only): the module can be imported
standalone from ``testbench/`` or through a package import.  Nothing is
installed, no subprocess is started, nothing is written.

All probes are read-only, per the test-bench rules:

* RSS and CPU time come from ``<proc_root>/<pid>/status`` and
  ``<proc_root>/<pid>/stat``;
* GPU load and temperature come from readable sysfs nodes below ``<sys_root>``;
* :meth:`MetricsSampler.find_model_pid` only scans
  ``<proc_root>/<pid>/cmdline``.  It never signals, stops or otherwise touches a
  process, and no ``sudo`` is ever used.

CPU semantics: each role's ``*_cpu_percent`` is the fraction of *total machine*
capacity used by that process over the interval between two samples, expressed
as 0-100 (100 % means every core is busy; a single fully busy core shows
``100 / os.cpu_count()`` on an N-core machine).  It is computed from the
``utime + stime`` delta of ``/proc/<pid>/stat``, divided by the real clock-tick
rate (``SC_CLK_TCK``) and the *measured* monotonic interval, then normalized by
``os.cpu_count()``.  The first sample for a pid has no delta baseline, so it
reports ``None`` - a value is never invented.  ``cpu_percent`` is the sum over
the worker, model and test-bench processes; a pid appearing in more than one
role is counted only once for CPU.

RSS is reported per role (each field is the RSS of that role's process, even in
the unlikely case that two roles resolve to the same pid).  GPU and temperature
are ``None`` when no readable/parseable sysfs node exists; CPU values are never
substituted for GPU.

Memory accounting semantics (``detail["memory"]``, additive - existing keys are
never renamed): a single number is not a valid answer for the "<= 4 GiB" cap, so
the snapshot carries three explicitly labelled states.

* ``conservative_upper_bound`` - ``sum(Rss) + sum(Swap) + NvMap`` over the whole
  sampled process tree.  Shared pages and overlapping mappings (PSS / NvMap /
  SwapPss counted twice) are deliberately *not* removed, so this is an upper
  bound only: it may prove the stack is inside the cap, and it must never be
  used on its own to declare a pass *or* an overrun.
* ``dedup_estimate`` - ``sum(Pss) + sum(SwapPss) + NvMap``.  ``Pss`` already
  divides every shared page by its number of sharers, so pages shared inside
  the sampled set are counted once.  Error sources are listed in
  ``error_sources`` (pages shared with processes outside the sampled set are
  under-counted; ``NvMap`` may overlap ``Pss`` for mapped GPU buffers).
* ``undeterminable`` - when ``smaps_rollup`` / ``VmRSS`` / ``VmSwap`` / NvMap
  cannot be read, when a pid disappears inside the sampling window, or when the
  GPU memory interface is absent, the affected aggregate is reported as
  ``undeterminable`` with a machine-readable reason instead of ``0`` or a
  silently dropped process.

``verdict`` combines the two states and never upgrades a screening result into
a pass.  Order of decision: a dedup estimate above the cap is ``over_cap``
(gaps can only make that number smaller, so the exceed still stands); any
sampling gap makes the run ``undeterminable`` because an unmeasured process
cannot support a "within cap" claim; a complete upper bound inside the cap is
``within_cap_upper_bound_only``; the historical ~4.016 GiB
PSS+NvMap+SwapPss region between the two is ``in_indeterminate_band``.
``verdict.is_pass_conclusion`` is always ``False`` because processes
outside the sampled tree, kernel/driver allocations outside NvMap, page cache,
tmpfs workspace and the swap/zram policy are not observed here; a memory pass
has to be concluded on the target with the requirements listed in
``verdict.requires_for_pass``.

The caller controls the sampling cadence (the HTTP server samples every 2 s);
this module never sleeps or loops on its own.  Every probe catches its own I/O
and parse errors, so :meth:`MetricsSampler.sample` never raises: unmeasurable
fields are ``None`` and, if a catastrophic error occurs, the previous snapshot
is returned with a fresh ``sampled_at``.
"""

from __future__ import annotations

import copy
import os
import time

__all__ = ["MetricsSampler", "collect_memory_accounting", "unavailable_memory"]

_DEFAULT_CLOCK_TICKS = 100
_READ_LIMIT = 8192

try:
    _CLOCK_TICKS = int(os.sysconf("SC_CLK_TCK"))
except (AttributeError, ValueError, OSError):  # Windows / unusual platforms
    _CLOCK_TICKS = _DEFAULT_CLOCK_TICKS
if _CLOCK_TICKS <= 0:
    _CLOCK_TICKS = _DEFAULT_CLOCK_TICKS

_ROLES = ("worker", "model", "bench")

# Acceptance caps (Docs/MCPT/AI_Eyes项目背景与主线验收标准.md §八.1):
# normal 1.5-2 GiB, peak <= 4 GiB expressed as 4x1024**3 bytes.
_CAP_PEAK_BYTES = 4 * 1024 ** 3
_CAP_NORMAL_LOW_BYTES = 1610612736  # 1.5 GiB
_CAP_NORMAL_HIGH_BYTES = 2 * 1024 ** 3

# /proc/<pid>/smaps_rollup fields used by the memory accounting (kB values).
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

# Jetson NvMap (GPU buffer) accounting nodes, relative to sysfs.  debugfs has to
# be mounted and readable; when it is not, the GPU part is undeterminable.
_GPU_MEM_CLIENT_DIRS = ("kernel/debug/nvmap/iovmm/clients",)
_GPU_MEM_TOTAL_FILES = (
    "kernel/debug/nvmap/iovmm/allocated",
    "kernel/debug/nvmap/allocated",
)

_MEMORY_METHOD = "procfs_smaps_rollup"

# Why ``within_cap_upper_bound_only`` is a screening result and never a pass
# conclusion.  Kept as module constants so the report and the JSON carry the
# identical wording.
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

# Orin exposes GPU load as an integer 0-1000; divide by 10 to get percent.
_GPU_STATIC_LOADS = (
    ("devices/gpu.0/load", 10.0),
    ("devices/platform/gpu.0/load", 10.0),
)

# Thermal zone priority: GPU/CPU zones first, then any generic thermal zone,
# then any other readable zone (e.g. "soc0").
_THERMAL_STRONG_KEYWORDS = ("gpu", "cpu")
_THERMAL_GENERIC_KEYWORD = "thermal"


def _now_ms():
    """Epoch milliseconds; a clock failure must not break sampling."""
    try:
        return int(time.time() * 1000)
    except Exception:
        return 0


def _coerce_pid(value):
    if value is None:
        return None
    try:
        pid = int(value)
    except (TypeError, ValueError):
        return None
    return pid if pid > 0 else None


def _call_safely(func, default=None):
    try:
        return func()
    except Exception:
        return default


def _read_text(path):
    """Read a small sysfs/procfs text file, or return None if unreadable."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return handle.read(_READ_LIMIT)
    except OSError:
        return None


def _read_number(path):
    """Read the first whitespace-separated number in a small file."""
    text = _read_text(path)
    if text is None:
        return None
    parts = text.strip().split()
    if not parts:
        return None
    try:
        return float(parts[0])
    except ValueError:
        return None


def _clamp_percent(value):
    return round(min(max(float(value), 0.0), 100.0), 1)


def _empty_snapshot(sampled_at):
    return {
        "sampled_at": sampled_at,
        "worker_rss_bytes": None,
        "model_rss_bytes": None,
        "bench_rss_bytes": None,
        "cpu_percent": None,
        "gpu_percent": None,
        "temperature_c": None,
        "detail": {
            "worker_cpu_percent": None,
            "model_cpu_percent": None,
            "bench_cpu_percent": None,
            "memory": unavailable_memory("sample_failed"),
        },
    }


# ------------------------------------------------------- 内存口径（三态）
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


def _read_ppid(pid, proc_root="/proc"):
    """Field 4 (ppid) of ``/proc/<pid>/stat``, or None."""
    path = os.path.join(proc_root, str(pid), "stat")
    try:
        with open(path, "rb") as handle:
            raw = handle.read(_READ_LIMIT)
    except OSError:
        return None
    try:
        text = raw.decode("utf-8", "replace")
        closing = text.rfind(")")
        if closing < 0:
            return None
        fields = text[closing + 1:].split()
        if len(fields) < 2:  # state, ppid
            return None
        ppid = int(fields[1])
    except (IndexError, ValueError):
        return None
    return ppid if ppid >= 0 else None


def _read_smaps_rollup_fields(pid, proc_root="/proc"):
    """``(values, missing_optional)`` from ``/proc/<pid>/smaps_rollup``.

    ``Rss`` and ``Pss`` are mandatory: a rolled-up file that lost either of
    them is reported as unreadable (``None``) rather than silently defaulted to
    0, because defaulting would understate the dedup estimate.

    ``Swap`` / ``SwapPss`` are *not* defaulted either.  Their absence is
    reported in ``missing_optional`` so the caller can turn the affected
    aggregate ``undeterminable`` with a locatable reason instead of adding an
    invented zero into a budget figure.
    """
    path = os.path.join(proc_root, str(pid), "smaps_rollup")
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


def _read_smaps_rollup(pid, proc_root="/proc"):
    """PSS/RSS/Swap figures from ``/proc/<pid>/smaps_rollup``, or None."""
    values, _missing = _read_smaps_rollup_fields(pid, proc_root)
    return values


def _read_vmrss_bytes(pid, proc_root="/proc"):
    """``VmRSS`` from ``/proc/<pid>/status`` in bytes, or None."""
    text = _read_text(os.path.join(proc_root, str(pid), "status"))
    if text is None:
        return None
    for line in text.splitlines():
        if not line.startswith("VmRSS:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            kib = int(parts[1])
        except ValueError:
            return None
        return kib * 1024 if kib >= 0 else None
    return None


def _read_vmswap_bytes(pid, proc_root="/proc"):
    """``VmSwap`` from ``/proc/<pid>/status`` in bytes, or None when unknown.

    A status file that has no ``VmSwap`` line is *not* the same as a process
    with nothing swapped out: only an explicit ``VmSwap: 0 kB`` means zero.
    Missing fields therefore return ``None`` so the caller reports the upper
    bound ``undeterminable`` instead of quietly dropping swap from it.
    """
    text = _read_text(os.path.join(proc_root, str(pid), "status"))
    if text is None:
        return None
    for line in text.splitlines():
        if not line.startswith("VmSwap:"):
            continue
        parts = line.split()
        if len(parts) < 2:
            return None
        try:
            kib = int(parts[1])
        except ValueError:
            return None
        return kib * 1024 if kib >= 0 else None
    return None


def _proc_table(proc_root):
    """``{pid: ppid}`` for every numeric entry of procfs; None if unreadable."""
    try:
        names = os.listdir(proc_root)
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
    """NvMap GPU buffer bytes: ``(bytes, backend, reason)``.

    ``allocated`` nodes are treated as bytes, which is what the Jetson nvmap
    debugfs interface reports; a different unit on some kernel must be recorded
    in the run report rather than guessed at.
    """
    for relative in _GPU_MEM_CLIENT_DIRS:
        root = os.path.join(sys_root, relative)
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
        raw = _read_number(os.path.join(sys_root, relative))
        if raw is not None:
            return int(raw), relative, None
    return None, None, "nvmap_unreadable_or_absent"


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

    ``roots`` is ``{role: pid}`` (a pid may also be a list).  Every descendant
    of those roots is included, pids shared between roots are counted once, and
    unrelated processes are never added.
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
    except Exception:
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
        entry = dict(member)
        entry.update(
            {
                "rss_bytes": None,
                "pss_bytes": None,
                "swap_bytes": None,
                "swap_pss_bytes": None,
                "state": "unknown",
            }
        )
        rollup = None
        missing = ()
        try:
            rollup, missing = _read_smaps_rollup_fields(pid, proc_root)
        except Exception:
            rollup, missing = None, ()
        if rollup is not None:
            entry.update(
                {
                    "rss_bytes": rollup.get("rss_bytes"),
                    "pss_bytes": rollup.get("pss_bytes"),
                    "swap_bytes": rollup.get("swap_bytes"),
                    "swap_pss_bytes": rollup.get("swap_pss_bytes"),
                }
            )
            entry["missing_optional_fields"] = list(missing)
            entry["state"] = "measured"
            rss_sum += rollup["rss_bytes"]
            pss_sum += rollup["pss_bytes"]
            # Swap 与 SwapPss 分别记账：一个缺字段只让依赖它的口径不可判定，
            # 不能悄悄补 0（补 0 会把上界/去重估值压低）。
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
                swap_sum += rollup["swap_bytes"]
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
                swap_pss_sum += rollup["swap_pss_bytes"]
        else:
            rss_bytes = None
            try:
                rss_bytes = _read_vmrss_bytes(pid, proc_root)
            except Exception:
                rss_bytes = None
            if rss_bytes is not None:
                swap_bytes = None
                try:
                    swap_bytes = _read_vmswap_bytes(pid, proc_root)
                except Exception:
                    swap_bytes = None
                entry["rss_bytes"] = rss_bytes
                entry["swap_bytes"] = swap_bytes
                entry["state"] = "rss_only"
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

    # 判定顺序（诚实性优先）：
    # 1) 去重估计已超 cap —— 缺测只会让去重值更小，超限结论仍然成立；
    # 2) 存在任何缺测（进程退出 / smaps 不可读 / Swap 或 SwapPss 缺字段 /
    #    NvMap 不可读）—— 本次采样不能支撑任何"在 cap 内"的结论；
    # 3) 上界完整且 ≤ cap —— 只能给"筛查级"结论，见 within_cap_upper_bound_only；
    # 4) 上界 > cap ≥ 去重 —— 重叠带，既不能宣布通过也不能宣布超限；
    # 5) 其余 —— 不可判定。
    if dedup_state == "computed" and dedup_bytes > cap_bytes:
        verdict_state = "over_cap"
        verdict_reason = "dedup_estimate_exceeds_cap"
    elif reasons:
        verdict_state = "undeterminable"
        verdict_reason = reasons[0]["reason"]
    elif upper_state == "computed" and upper_bytes <= cap_bytes:
        # 上界低于 cap 只能证明“本工具采样范围内”不超；树外进程、内核/驱动
        # 非 nvmap 分配、页面缓存与临时工作区都不在上界里，所以这里不能叫
        # provably/pass，只能作为筛查结论（is_pass_conclusion 恒为 False）。
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


class MetricsSampler:
    """Sample worker / model / test-bench process and sysfs metrics.

    :param proc_root: procfs mount point used for RSS/CPU/cmdline reads.
    :param sys_root: sysfs mount point used for GPU load and temperatures.
    :param worker_pid_provider: optional zero-argument callable returning the
        business worker pid (or ``None``).
    :param model_pid_provider: optional zero-argument callable returning the
        model server pid.  When omitted, :meth:`find_model_pid` is used to
        discover ``llama-server`` on each sample.
    :param bench_pid: optional pid of this test bench's own server process.
        ``bench_pid``, ``worker_pid_provider`` and ``model_pid_provider`` are
        public attributes and may also be set after construction.
    :param include_descendants: when true (default) the memory accounting of
        ``detail["memory"]`` covers the whole process tree below the worker,
        model and bench pids, so worker/runner children are never dropped.
    :param gpu_mem_required: when true (default, target-board semantics) an
        unreadable NvMap node keeps the aggregates ``undeterminable``.  Set it
        to false only for a CPU-only dev host, where it is recorded as
        ``gpu_mem_not_required`` instead.
    """

    def __init__(
        self,
        *,
        proc_root="/proc",
        sys_root="/sys",
        worker_pid_provider=None,
        model_pid_provider=None,
        bench_pid=None,
        include_descendants=True,
        gpu_mem_required=True,
    ):
        self.proc_root = os.fspath(proc_root)
        self.sys_root = os.fspath(sys_root)
        self.worker_pid_provider = worker_pid_provider
        self.model_pid_provider = model_pid_provider
        self.bench_pid = bench_pid
        self.include_descendants = bool(include_descendants)
        self.gpu_mem_required = bool(gpu_mem_required)
        # pid -> (utime+stime ticks, monotonic timestamp of that reading)
        self._cpu_baseline = {}
        self._last_snapshot = None

    # ------------------------------------------------------------------ API
    def sample(self):
        """Return one metrics snapshot; never raises."""
        sampled_at = _now_ms()
        try:
            snapshot = self._collect(sampled_at)
        except Exception:
            previous = self._last_snapshot
            if previous is None:
                snapshot = _empty_snapshot(sampled_at)
            else:
                try:
                    snapshot = copy.deepcopy(previous)
                except Exception:
                    snapshot = _empty_snapshot(sampled_at)
                snapshot["sampled_at"] = sampled_at
        self._last_snapshot = snapshot
        return snapshot

    def find_model_pid(self):
        """Return the pid whose executable argument is ``llama-server``.

        Read-only: only ``<proc_root>/<pid>/cmdline`` is read.  Returns ``None``
        when no match is found or procfs is unreadable.
        """
        try:
            names = os.listdir(self.proc_root)
        except OSError:
            return None
        for name in names:
            if not name.isdigit():
                continue
            command = self._read_cmdline(os.path.join(self.proc_root, name))
            if not command:
                continue
            executable = command[0].replace("\\", "/").rsplit("/", 1)[-1]
            if executable.startswith("llama-server"):
                return int(name)
        return None

    # ------------------------------------------------------------- snapshot
    def _collect(self, sampled_at):
        interval_start = time.monotonic()

        if self.model_pid_provider is not None:
            model_pid = self._resolve_pid(self.model_pid_provider)
        else:
            model_pid = _call_safely(self.find_model_pid)

        roles = {
            "worker": self._resolve_pid(self.worker_pid_provider),
            "model": model_pid,
            "bench": self._resolve_pid(self.bench_pid),
        }

        # RSS: per-role field, one read per distinct pid.
        rss_cache = {}
        rss = {}
        for role in _ROLES:
            pid = roles[role]
            if pid is None:
                rss[role] = None
                continue
            if pid not in rss_cache:
                rss_cache[pid] = self._read_rss_bytes(pid)
            rss[role] = rss_cache[pid]

        # CPU: delta per distinct pid; a duplicated pid is counted once.
        cpu = {role: None for role in _ROLES}
        seen = set()
        cpu_count = os.cpu_count() or 1
        for role in _ROLES:
            pid = roles[role]
            if pid is None or pid in seen:
                continue
            seen.add(pid)
            ticks = self._read_cpu_ticks(pid)
            if ticks is None:
                self._cpu_baseline.pop(pid, None)
                continue
            previous = self._cpu_baseline.get(pid)
            self._cpu_baseline[pid] = (ticks, interval_start)
            if previous is None:
                continue
            previous_ticks, previous_start = previous
            elapsed = interval_start - previous_start
            delta = ticks - previous_ticks
            if elapsed <= 0.0 or delta < 0:
                continue
            cores = delta / _CLOCK_TICKS / elapsed
            cpu[role] = round(min(cores / cpu_count * 100.0, 100.0), 2)

        # Forget baselines of processes that are no longer sampled.
        for stale_pid in [pid for pid in self._cpu_baseline if pid not in seen]:
            self._cpu_baseline.pop(stale_pid, None)

        known = [value for value in cpu.values() if value is not None]
        cpu_percent = round(min(sum(known), 100.0), 2) if known else None

        memory = _call_safely(
            lambda: collect_memory_accounting(
                {
                    "worker": roles["worker"],
                    "model": roles["model"],
                    "bench": roles["bench"],
                },
                proc_root=self.proc_root,
                sys_root=self.sys_root,
                include_descendants=self.include_descendants,
                gpu_mem_required=self.gpu_mem_required,
            )
        )
        if not isinstance(memory, dict):
            memory = unavailable_memory("accounting_failed")

        return {
            "sampled_at": sampled_at,
            "worker_rss_bytes": rss["worker"],
            "model_rss_bytes": rss["model"],
            "bench_rss_bytes": rss["bench"],
            "cpu_percent": cpu_percent,
            "gpu_percent": self._read_gpu_percent(),
            "temperature_c": self._read_temperature_c(),
            "detail": {
                "worker_cpu_percent": cpu["worker"],
                "model_cpu_percent": cpu["model"],
                "bench_cpu_percent": cpu["bench"],
                "memory": memory,
            },
        }

    def _resolve_pid(self, source):
        if source is None:
            return None
        try:
            value = source() if callable(source) else source
        except Exception:
            return None
        return _coerce_pid(value)

    # ------------------------------------------------------------ /proc
    def _read_rss_bytes(self, pid):
        """``VmRSS`` from ``/proc/<pid>/status`` in bytes, or None."""
        return _read_vmrss_bytes(pid, self.proc_root)

    def _read_cpu_ticks(self, pid):
        """``utime + stime`` ticks from ``/proc/<pid>/stat``, or None."""
        try:
            with open(
                os.path.join(self.proc_root, str(pid), "stat"), "rb"
            ) as handle:
                raw = handle.read(_READ_LIMIT)
        except OSError:
            return None
        try:
            text = raw.decode("utf-8", "replace")
            closing = text.rfind(")")
            if closing < 0:
                return None
            # Field 3 (state) is fields[0], so utime(14)/stime(15) are 11/12.
            fields = text[closing + 1:].split()
            if len(fields) < 13:
                return None
            return int(fields[11]) + int(fields[12])
        except (IndexError, ValueError):
            return None

    def _read_cmdline(self, pid):
        path = os.path.join(self.proc_root, str(pid), "cmdline")
        try:
            with open(path, "rb") as handle:
                raw = handle.read(_READ_LIMIT)
        except OSError:
            return []
        if not raw:
            return []
        return [
            part.decode("utf-8", "replace")
            for part in raw.split(b"\x00")
            if part
        ]

    # ------------------------------------------------------------- /sys
    def _read_gpu_percent(self):
        """Read a known GPU-load node; None when unavailable (no CPU fallback)."""
        for relative, divisor in _GPU_STATIC_LOADS:
            raw = _read_number(os.path.join(self.sys_root, relative))
            if raw is not None:
                return _clamp_percent(raw / divisor)

        # Generic devfreq load; some kernels report 0-1000, some 0-100.
        devfreq_root = os.path.join(self.sys_root, "class", "devfreq")
        for device in self._child_dirs(devfreq_root):
            raw = _read_number(os.path.join(device, "load"))
            if raw is None:
                continue
            divisor = 10.0 if raw > 100.0 else 1.0
            return _clamp_percent(raw / divisor)

        # DRM / KGSL expose a ready-made busy percentage.
        drm_root = os.path.join(self.sys_root, "class", "drm")
        for card in self._child_dirs(drm_root):
            raw = _read_number(
                os.path.join(card, "device", "gpu_busy_percent")
            )
            if raw is not None:
                return _clamp_percent(raw)
        raw = _read_number(
            os.path.join(
                self.sys_root, "class", "kgsl", "kgsl-3d0",
                "gpu_busy_percentage",
            )
        )
        if raw is not None:
            return _clamp_percent(raw)
        return None

    def _read_temperature_c(self):
        """Read a thermal zone in degrees Celsius; None when unavailable."""
        thermal_root = os.path.join(self.sys_root, "class", "thermal")
        try:
            names = os.listdir(thermal_root)
        except OSError:
            return None
        zones = []
        for name in names:
            if not name.startswith("thermal_zone"):
                continue
            suffix = name[len("thermal_zone"):]
            if not suffix.isdigit():
                continue
            zones.append((int(suffix), os.path.join(thermal_root, name)))
        if not zones:
            return None
        zones.sort(key=lambda item: item[0])

        tiers = ([], [], [])
        for _, zone in zones:
            raw = _read_number(os.path.join(zone, "temp"))
            if raw is None:
                continue
            celsius = raw / 1000.0
            if not -40.0 <= celsius <= 150.0:
                continue
            label = (
                _read_text(os.path.join(zone, "type")) or ""
            ).strip().lower()
            if any(keyword in label for keyword in _THERMAL_STRONG_KEYWORDS):
                tiers[0].append(celsius)
            elif _THERMAL_GENERIC_KEYWORD in label:
                tiers[1].append(celsius)
            else:
                tiers[2].append(celsius)
        for values in tiers:
            if values:
                return round(values[0], 1)
        return None

    @staticmethod
    def _child_dirs(path):
        try:
            names = os.listdir(path)
        except OSError:
            return []
        return [os.path.join(path, name) for name in sorted(names)]
