"""内存口径分层测试（D 组：采样归因 / 三态口径 / 健壮性）。

三类缺陷用可注入的 procfs / sysfs 文本 fixture 复现：

1. 只有 RSS 三个字段、没有口径标签，调用方会把朴素和当成精确去重值；
2. 进程归属只覆盖 worker/model/bench 三个显式 pid，worker 子进程/运行器被漏算；
3. 进程退出、字段缺失、GPU(NvMap) 不可得时返回 0 或静默丢弃，给出错误数值。

形状与判定口径见 ``Testbench/metrics.py`` 模块 docstring。

在固定的 Testbench 目录导入；``AIEYES_METRICS_DIR`` 只用于把同一批用例指到
另一份 metrics.py（例如修复前的原始版本）做失败复现。
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_METRICS_DIR = os.environ.get("AIEYES_METRICS_DIR") or str(
    Path(__file__).resolve().parents[1]
)
sys.path.insert(0, _METRICS_DIR)

import metrics  # noqa: E402

GiB = 1024 ** 3


# --------------------------------------------------------------------------
# fixture 构造
# --------------------------------------------------------------------------
def stat_text(pid, ppid=0, utime=0, stime=0, comm="proc"):
    """52 字段 /proc/<pid>/stat；ppid 是字段 4（")" 之后 index 1）。"""
    tail = ["S"] + ["0"] * 49
    tail[1] = str(ppid)
    tail[11] = str(utime)
    tail[12] = str(stime)
    return f"{pid} ({comm}) " + " ".join(tail) + "\n"


def status_text(rss_kb=None, swap_kb=None):
    lines = ["Name:\tpython", "State:\tS (sleeping)", "Threads:\t1"]
    if rss_kb is not None:
        lines.append(f"VmRSS:\t{rss_kb:8d} kB")
    if swap_kb is not None:
        lines.append(f"VmSwap:\t{swap_kb:8d} kB")
    return "\n".join(lines) + "\n"


def smaps_rollup_text(rss_kb, pss_kb, swap_kb=0, swap_pss_kb=0, *, omit=()):
    """真实 /proc/<pid>/smaps_rollup 的字段形状（值为 kB）。"""
    fields = [
        ("Rss", rss_kb),
        ("Pss", pss_kb),
        ("Pss_Anon", pss_kb),
        ("Pss_File", 0),
        ("Pss_Shmem", 0),
        ("Shared_Clean", max(0, rss_kb - pss_kb)),
        ("Shared_Dirty", 0),
        ("Private_Clean", 0),
        ("Private_Dirty", pss_kb),
        ("Referenced", rss_kb),
        ("Anonymous", pss_kb),
        ("Swap", swap_kb),
        ("SwapPss", swap_pss_kb),
        ("Locked", 0),
    ]
    lines = ["55d0e5a1a000-55d0e5a2b000 ---p 00000000 00:00 0 \t[rollup]"]
    for name, value in fields:
        if name in omit:
            continue
        lines.append(f"{name}:{' ' * 16}{value} kB")
    return "\n".join(lines) + "\n"


class FakeTree:
    def __init__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.proc = self.root / "proc"
        self.sys = self.root / "sys"
        self.proc.mkdir()
        self.sys.mkdir()

    def cleanup(self):
        self._tmp.cleanup()

    def write(self, relative, content, *, binary=False):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if binary:
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        return path

    def add_process(
        self,
        pid,
        *,
        ppid=0,
        rss_kb=1024,
        pss_kb=None,
        swap_kb=0,
        swap_pss_kb=0,
        cmdline=None,
        with_status=True,
        with_smaps=True,
        smaps_kwargs=None,
    ):
        """一个进程；pss_kb 为 None 时不写 smaps_rollup。"""
        (self.proc / str(pid)).mkdir(parents=True, exist_ok=True)
        self.write(f"proc/{pid}/stat", stat_text(pid, ppid=ppid))
        if with_status:
            self.write(
                f"proc/{pid}/status",
                status_text(rss_kb=rss_kb, swap_kb=swap_kb),
            )
        if with_smaps and pss_kb is not None:
            self.write(
                f"proc/{pid}/smaps_rollup",
                smaps_rollup_text(
                    rss_kb, pss_kb, swap_kb, swap_pss_kb, **(smaps_kwargs or {})
                ),
            )
        if cmdline is not None:
            payload = b"".join(arg.encode("utf-8") + b"\x00" for arg in cmdline)
            self.write(f"proc/{pid}/cmdline", payload, binary=True)

    def remove_process(self, pid):
        shutil.rmtree(self.proc / str(pid), ignore_errors=True)

    def add_nvmap_client(self, name, allocated_bytes):
        self.write(
            f"sys/kernel/debug/nvmap/iovmm/clients/{name}/allocated",
            f"{allocated_bytes}\n",
        )


class MemoryCase(unittest.TestCase):
    def setUp(self):
        self.tree = FakeTree()
        self.addCleanup(self.tree.cleanup)

    def collect(self, roots, **kwargs):
        kwargs.setdefault("proc_root", str(self.tree.proc))
        kwargs.setdefault("sys_root", str(self.tree.sys))
        return metrics.collect_memory_accounting(roots, **kwargs)


# --------------------------------------------------------------------------
# 缺陷 1：没有口径标签，RSS 朴素和被当成去重值
# --------------------------------------------------------------------------
class TieringTests(MemoryCase):
    def test_snapshot_carries_three_labelled_states(self):
        self.tree.add_process(200, rss_kb=2048, pss_kb=1024)
        self.tree.add_nvmap_client("c0", 4096)
        sampler = metrics.MetricsSampler(
            proc_root=str(self.tree.proc),
            sys_root=str(self.tree.sys),
            worker_pid_provider=lambda: 200,
        )

        memory = sampler.sample()["detail"]["memory"]

        self.assertEqual(memory["accounting"], "tiered")
        upper = memory["conservative_upper_bound"]
        dedup = memory["dedup_estimate"]
        unknown = memory["undeterminable"]
        self.assertEqual(upper["label"], "conservative_upper_bound")
        self.assertEqual(dedup["label"], "dedup_estimate")
        self.assertEqual(unknown["label"], "undeterminable")
        self.assertIn("not_for_pass", upper["usable_for"])
        self.assertIn("not_a_pass", dedup["usable_for"])
        self.assertTrue(upper["overlap"])
        self.assertEqual(upper["state"], "computed")
        self.assertEqual(dedup["state"], "computed")

    def test_dedup_is_not_the_naive_rss_sum(self):
        # 两个进程各自 RSS 100 MiB、PSS 60 MiB（共享页），另有 10 MiB NvMap。
        self.tree.add_process(200, rss_kb=1024 * 100, pss_kb=1024 * 60)
        self.tree.add_process(201, ppid=200, rss_kb=1024 * 100, pss_kb=1024 * 60)
        self.tree.add_nvmap_client("c0", 10 * 1024 * 1024)

        memory = self.collect({"worker": 200})

        upper = memory["conservative_upper_bound"]["bytes"]
        dedup = memory["dedup_estimate"]["bytes"]
        self.assertEqual(upper, 200 * 1024 * 1024 + 10 * 1024 * 1024)
        self.assertEqual(dedup, 120 * 1024 * 1024 + 10 * 1024 * 1024)
        self.assertNotEqual(upper, dedup)
        self.assertEqual(memory["components"]["rss_sum_bytes"], 200 * 1024 * 1024)
        self.assertEqual(memory["components"]["pss_sum_bytes"], 120 * 1024 * 1024)

    def test_swap_is_counted_so_zram_cannot_hide_residency(self):
        self.tree.add_process(
            200, rss_kb=1024, pss_kb=512, swap_kb=256, swap_pss_kb=128
        )
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertEqual(
            memory["conservative_upper_bound"]["bytes"], (1024 + 256) * 1024
        )
        self.assertEqual(
            memory["dedup_estimate"]["bytes"], (512 + 128) * 1024
        )

    def test_memory_block_is_json_serializable(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)
        self.tree.add_nvmap_client("c0", 0)
        memory = self.collect({"worker": 200})
        json.dumps(memory)  # 不能抛：metrics.jsonl 原样落盘


# --------------------------------------------------------------------------
# 缺陷 2：进程归属只认三个 pid，漏算 worker 子进程/运行器
# --------------------------------------------------------------------------
class ProcessTreeTests(MemoryCase):
    def test_children_and_grandchildren_are_included(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)
        self.tree.add_process(201, ppid=200, rss_kb=2048, pss_kb=2048)
        self.tree.add_process(202, ppid=201, rss_kb=4096, pss_kb=4096)
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["process_count"], 3)
        self.assertEqual(
            sorted(item["pid"] for item in memory["processes"]),
            [200, 201, 202],
        )
        self.assertEqual(
            memory["conservative_upper_bound"]["bytes"], (1024 + 2048 + 4096) * 1024
        )
        depths = {item["pid"]: item["depth"] for item in memory["processes"]}
        self.assertEqual(depths, {200: 0, 201: 1, 202: 2})

    def test_unrelated_processes_are_not_counted(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)
        self.tree.add_process(300, ppid=1, rss_kb=8 * 1024 * 1024, pss_kb=1024)
        self.tree.add_process(301, ppid=300, rss_kb=1024, pss_kb=1024)
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["process_count"], 1)
        self.assertEqual(memory["conservative_upper_bound"]["bytes"], 1024 * 1024)

    def test_pid_shared_between_roles_is_counted_once(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)
        self.tree.add_process(210, rss_kb=1024, pss_kb=1024)
        self.tree.add_process(220, ppid=210, rss_kb=64 * 1024, pss_kb=64 * 1024)
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200, "model": 210, "bench": 210})

        self.assertEqual(memory["process_count"], 3)
        workers = [item for item in memory["processes"] if item["pid"] == 220]
        self.assertEqual(len(workers), 1)
        self.assertEqual(workers[0]["role"], "model")
        self.assertEqual(
            memory["conservative_upper_bound"]["bytes"], (1024 + 1024 + 64 * 1024) * 1024
        )

    def test_include_descendants_false_keeps_roots_only(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)
        self.tree.add_process(201, ppid=200, rss_kb=2048, pss_kb=2048)
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200}, include_descendants=False)

        self.assertEqual(memory["process_count"], 1)
        self.assertEqual(memory["conservative_upper_bound"]["bytes"], 1024 * 1024)

    def test_sampler_includes_worker_tree(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)
        self.tree.add_process(201, ppid=200, rss_kb=4096, pss_kb=4096)
        self.tree.add_nvmap_client("c0", 0)
        sampler = metrics.MetricsSampler(
            proc_root=str(self.tree.proc),
            sys_root=str(self.tree.sys),
            worker_pid_provider=lambda: 200,
        )

        memory = sampler.sample()["detail"]["memory"]

        self.assertEqual(memory["process_count"], 2)
        self.assertEqual(memory["roots"], {"worker": 200})


# --------------------------------------------------------------------------
# 缺陷 3：GPU / NvMap 不可得时不得编造 0
# --------------------------------------------------------------------------
class GpuMemoryTests(MemoryCase):
    def test_missing_nvmap_is_undeterminable_with_reason(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)

        memory = self.collect({"worker": 200})

        self.assertIsNone(memory["conservative_upper_bound"]["bytes"])
        self.assertEqual(memory["conservative_upper_bound"]["state"], "undeterminable")
        self.assertEqual(memory["dedup_estimate"]["state"], "undeterminable")
        self.assertIsNone(memory["components"]["nvmap_bytes"])
        reasons = [item["component"] for item in memory["undeterminable"]["reasons"]]
        self.assertIn("gpu_nvmap", reasons)
        self.assertEqual(memory["verdict"]["state"], "undeterminable")
        self.assertFalse(memory["complete"])

    def test_nvmap_clients_are_summed_and_backend_recorded(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)
        self.tree.add_nvmap_client("c0", 3 * 1024 * 1024)
        self.tree.add_nvmap_client("c1", 5 * 1024 * 1024)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["components"]["nvmap_bytes"], 8 * 1024 * 1024)
        self.assertEqual(
            memory["components"]["nvmap_backend"],
            "kernel/debug/nvmap/iovmm/clients",
        )
        self.assertEqual(
            memory["conservative_upper_bound"]["bytes"], 9 * 1024 * 1024
        )
        self.assertTrue(memory["complete"])

    def test_nvmap_total_file_is_the_fallback(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)
        self.tree.write("sys/kernel/debug/nvmap/iovmm/allocated", "2048\n")

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["components"]["nvmap_bytes"], 2048)
        self.assertEqual(
            memory["components"]["nvmap_backend"],
            "kernel/debug/nvmap/iovmm/allocated",
        )

    def test_cpu_only_host_opts_out_explicitly(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)

        memory = self.collect({"worker": 200}, gpu_mem_required=False)

        self.assertEqual(memory["components"]["nvmap_bytes"], 0)
        self.assertEqual(
            memory["components"]["nvmap_backend"], "not_required_cpu_only_host"
        )
        self.assertIn("gpu_mem_not_required_nvmap_counted_as_zero", memory["notes"])
        self.assertEqual(memory["conservative_upper_bound"]["state"], "computed")


# --------------------------------------------------------------------------
# 缺陷 4：进程退出 / 字段缺失 / 采样窗口漂移
# --------------------------------------------------------------------------
class RobustnessTests(MemoryCase):
    def test_process_exiting_mid_sample_is_a_gap_not_zero(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)
        self.tree.add_nvmap_client("c0", 0)
        sampler = metrics.MetricsSampler(
            proc_root=str(self.tree.proc),
            sys_root=str(self.tree.sys),
            worker_pid_provider=lambda: 200,
        )
        sampler.sample()
        self.tree.remove_process(200)

        memory = sampler.sample()["detail"]["memory"]

        gaps = [item["reason"] for item in memory["sampling_gaps"]]
        self.assertIn("exited_during_sample", gaps)
        self.assertIsNone(memory["conservative_upper_bound"]["bytes"])
        self.assertEqual(memory["verdict"]["state"], "undeterminable")
        self.assertFalse(memory["complete"])
        states = [item["state"] for item in memory["processes"]]
        self.assertEqual(states, ["exited"])

    def test_missing_smaps_falls_back_to_rss_and_flags_dedup(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=None, swap_kb=0)
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["processes"][0]["state"], "rss_only")
        # 上界仍可用（RSS+Swap+NvMap），去重估计必须不可判定。
        self.assertEqual(memory["conservative_upper_bound"]["state"], "computed")
        self.assertEqual(memory["conservative_upper_bound"]["bytes"], 1024 * 1024)
        self.assertEqual(memory["dedup_estimate"]["state"], "undeterminable")
        self.assertIsNone(memory["dedup_estimate"]["bytes"])
        reasons = [item["reason"] for item in memory["undeterminable"]["reasons"]]
        self.assertIn("smaps_rollup_unreadable", reasons)

    def test_smaps_without_pss_field_is_rejected(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=512)
        self.tree.write(
            "proc/200/smaps_rollup",
            smaps_rollup_text(1024, 512, omit=("Pss",)),
        )
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["processes"][0]["state"], "rss_only")
        self.assertEqual(memory["dedup_estimate"]["state"], "undeterminable")

    def test_corrupt_smaps_is_treated_as_unreadable(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=512)
        self.tree.write("proc/200/smaps_rollup", "garbage\n")
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["processes"][0]["state"], "rss_only")
        self.assertEqual(memory["conservative_upper_bound"]["bytes"], 1024 * 1024)

    def test_unreadable_procfs_returns_undeterminable(self):
        memory = self.collect(
            {"worker": 200}, proc_root=str(self.tree.root / "missing-proc")
        )

        self.assertIsInstance(memory, dict)
        self.assertEqual(memory["verdict"]["state"], "undeterminable")
        reasons = [item["reason"] for item in memory["undeterminable"]["reasons"]]
        self.assertIn("procfs_unreadable", reasons)
        self.assertIsNone(memory["conservative_upper_bound"]["bytes"])

    def test_no_roots_returns_undeterminable(self):
        memory = metrics.collect_memory_accounting(
            {"worker": None}, proc_root=str(self.tree.proc), sys_root=str(self.tree.sys)
        )
        self.assertEqual(memory["verdict"]["state"], "undeterminable")
        self.assertEqual(
            memory["undeterminable"]["reasons"][0]["reason"], "no_roots"
        )

    def test_sampling_window_is_reported(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=1024)
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertIsInstance(memory["sampling_window_ms"], int)
        self.assertGreaterEqual(memory["sampling_window_ms"], 0)


# --------------------------------------------------------------------------
# 缺陷 5：判定口径（清单第 39 项：4.016 GiB 那种重叠不可当精确超限）
# --------------------------------------------------------------------------
class VerdictTests(MemoryCase):
    def test_cap_is_exactly_4x1024_cubed(self):
        self.assertEqual(metrics._CAP_PEAK_BYTES, 4 * 1024 ** 3)
        self.assertEqual(metrics._CAP_PEAK_BYTES, 4294967296)
        self.assertEqual(metrics._CAP_NORMAL_LOW_BYTES, 1610612736)
        self.assertEqual(metrics._CAP_NORMAL_HIGH_BYTES, 2147483648)

    def _kv(self, name, value):
        return f"{name}:{' ' * 16}{value} kB"

    def _fixed_process(self, pid, rss_bytes, pss_bytes, nvmap_bytes=0):
        """直接用字节构造一条 smaps_rollup（绕过 kB 取整，便于卡边界）。"""
        (self.tree.proc / str(pid)).mkdir(parents=True, exist_ok=True)
        self.tree.write(f"proc/{pid}/stat", stat_text(pid, ppid=0))
        self.tree.write(f"proc/{pid}/status", status_text(rss_kb=rss_bytes // 1024))
        lines = [
            "55d0e5a1a000-55d0e5a2b000 ---p 00000000 00:00 0 \t[rollup]",
            self._kv("Rss", rss_bytes // 1024),
            self._kv("Pss", pss_bytes // 1024),
            self._kv("Swap", 0),
            self._kv("SwapPss", 0),
        ]
        self.tree.write(f"proc/{pid}/smaps_rollup", "\n".join(lines) + "\n")
        self.tree.add_nvmap_client("c0", nvmap_bytes)

    def test_upper_bound_inside_cap_is_screening_only_not_a_pass(self):
        self._fixed_process(200, 2 * GiB, 2 * GiB, nvmap_bytes=0)

        memory = self.collect({"worker": 200})

        verdict = memory["verdict"]
        # 上界在 cap 内只说明“采样树内”不超，绝不等于达标结论。
        self.assertEqual(verdict["state"], "within_cap_upper_bound_only")
        self.assertEqual(
            verdict["reason"], "conservative_upper_bound_inside_cap_screening_only"
        )
        self.assertFalse(verdict["is_pass_conclusion"])
        self.assertFalse(verdict["is_exceed_conclusion"])
        self.assertEqual(verdict["scope"], "sampled_process_tree_plus_nvmap_only")
        self.assertIn(
            "processes_outside_the_sampled_tree_are_not_counted", verdict["caveats"]
        )
        self.assertIn(
            "target_board_measurement_of_normal_and_peak", verdict["requires_for_pass"]
        )
        self.assertEqual(
            verdict["normal_range"]["dedup_position_meaning"],
            "estimate_not_exact_value",
        )
        self.assertEqual(
            verdict["normal_range"]["dedup_position"],
            "within_normal_range",
        )

    def test_over_cap_is_an_exceed_but_still_not_a_pass(self):
        self._fixed_process(200, 5 * GiB, 5 * GiB, nvmap_bytes=0)

        memory = self.collect({"worker": 200})

        verdict = memory["verdict"]
        self.assertEqual(verdict["state"], "over_cap")
        self.assertTrue(verdict["is_exceed_conclusion"])
        self.assertFalse(verdict["is_pass_conclusion"])

    def test_dedup_above_cap_is_over_cap(self):
        self._fixed_process(200, 5 * GiB, 5 * GiB, nvmap_bytes=0)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["verdict"]["state"], "over_cap")
        self.assertEqual(memory["verdict"]["reason"], "dedup_estimate_exceeds_cap")

    def test_overlap_band_is_indeterminate_neither_pass_nor_exceed(self):
        # 复刻清单第 39 项：朴素上界超 4 GiB，去重估计没超 → 落在重叠带。
        self._fixed_process(200, 4 * GiB + 200 * 1024 * 1024, 3 * GiB, nvmap_bytes=0)
        self.tree.add_process(201, ppid=200, rss_kb=1024 * 100, pss_kb=1024 * 90)
        self.tree.write("sys/kernel/debug/nvmap/iovmm/clients/c0/allocated", "0\n")

        memory = self.collect({"worker": 200})

        upper = memory["conservative_upper_bound"]["bytes"]
        dedup = memory["dedup_estimate"]["bytes"]
        self.assertGreater(upper, metrics._CAP_PEAK_BYTES)
        self.assertLessEqual(dedup, metrics._CAP_PEAK_BYTES)
        self.assertEqual(memory["verdict"]["state"], "in_indeterminate_band")
        self.assertEqual(
            memory["verdict"]["reason"],
            "overlap_band_upper_above_cap_dedup_below_cap",
        )

    def test_undeterminable_wins_over_band(self):
        self._fixed_process(200, 4 * GiB + 200 * 1024 * 1024, 3 * GiB)
        # 删掉 NvMap 节点 → 上界不可判定，即使 RSS 数值本身超 cap 也不能宣布超限。
        shutil.rmtree(self.tree.sys / "kernel", ignore_errors=True)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["verdict"]["state"], "undeterminable")
        self.assertIn(
            "gpu_nvmap",
            [item["component"] for item in memory["undeterminable"]["reasons"]],
        )


# --------------------------------------------------------------------------
# 第二轮复核 3：Swap / SwapPss / NvMap 的单位与缺字段口径
# --------------------------------------------------------------------------
class SwapAndNvMapSemanticsTests(MemoryCase):
    def test_swap_and_swap_pss_are_separate_components(self):
        """/proc 的 Swap 与 smaps_rollup 的 SwapPss 是两条独立账，不许互相顶替。"""
        self.tree.add_process(
            200, rss_kb=100 * 1024, pss_kb=60 * 1024, swap_kb=64 * 1024, swap_pss_kb=16 * 1024
        )
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        components = memory["components"]
        self.assertEqual(components["swap_sum_bytes"], 64 * 1024 * 1024)
        self.assertEqual(components["swap_pss_sum_bytes"], 16 * 1024 * 1024)
        # 上界吃 Swap，去重吃 SwapPss；两者不混用也不相加到同一个公式里。
        self.assertEqual(
            memory["conservative_upper_bound"]["bytes"],
            (100 + 64) * 1024 * 1024,
        )
        self.assertEqual(
            memory["dedup_estimate"]["bytes"], (60 + 16) * 1024 * 1024
        )

    def test_missing_swap_pss_breaks_only_the_dedup_estimate(self):
        self.tree.add_process(
            200,
            rss_kb=100 * 1024,
            pss_kb=60 * 1024,
            swap_kb=8 * 1024,
            swap_pss_kb=4 * 1024,
            smaps_kwargs={"omit": ("SwapPss",)},
        )
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["conservative_upper_bound"]["state"], "computed")
        self.assertEqual(
            memory["conservative_upper_bound"]["bytes"], (100 + 8) * 1024 * 1024
        )
        self.assertEqual(memory["dedup_estimate"]["state"], "undeterminable")
        self.assertIsNone(memory["dedup_estimate"]["bytes"])
        self.assertIsNone(memory["components"]["swap_pss_sum_bytes"])
        self.assertEqual(memory["processes"][0]["missing_optional_fields"], ["swap_pss_bytes"])
        self.assertIn(
            "smaps_rollup_missing_swap_pss_field",
            [gap["reason"] for gap in memory["sampling_gaps"]],
        )
        # 缺字段 → 不能给任何"在 cap 内"的筛查结论。
        self.assertEqual(memory["verdict"]["state"], "undeterminable")

    def test_missing_swap_field_breaks_the_upper_bound_too(self):
        self.tree.add_process(
            200,
            rss_kb=100 * 1024,
            pss_kb=60 * 1024,
            swap_kb=8 * 1024,
            smaps_kwargs={"omit": ("Swap",)},
        )
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["conservative_upper_bound"]["state"], "undeterminable")
        self.assertIsNone(memory["conservative_upper_bound"]["bytes"])
        self.assertIsNone(memory["components"]["swap_sum_bytes"])
        self.assertIn(
            "smaps_rollup_missing_swap_field",
            [gap["reason"] for gap in memory["sampling_gaps"]],
        )
        # 不能因为"没读到"就把换出页当 0 从而给出上界。
        self.assertEqual(memory["verdict"]["state"], "undeterminable")

    def test_vmswap_without_the_field_is_undeterminable_not_zero(self):
        """没有 smaps_rollup 时上界只能靠 status：VmSwap 缺失必须不可判定。"""
        self.tree.add_process(200, rss_kb=100 * 1024, with_smaps=False, swap_kb=None)
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        entry = memory["processes"][0]
        self.assertEqual(entry["state"], "rss_only")
        self.assertIsNone(entry["swap_bytes"])
        self.assertIsNone(memory["conservative_upper_bound"]["bytes"])
        self.assertIn(
            "vmswap_field_missing_or_unreadable",
            [gap["reason"] for gap in memory["sampling_gaps"]],
        )

    def test_vmswap_zero_is_a_real_zero(self):
        self.tree.add_process(200, rss_kb=100 * 1024, with_smaps=False, swap_kb=0)
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["processes"][0]["swap_bytes"], 0)
        self.assertEqual(
            memory["conservative_upper_bound"]["bytes"], 100 * 1024 * 1024
        )

    def test_nvmap_allocated_is_bytes_not_kib(self):
        """nvmap debugfs 的 allocated 是字节；乘 1024 会让 GPU 部分虚高 1024 倍。"""
        self.tree.add_process(200, rss_kb=1024, pss_kb=512)
        self.tree.add_nvmap_client("c0", 4096)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["components"]["nvmap_bytes"], 4096)
        self.assertEqual(
            memory["conservative_upper_bound"]["bytes"], 1024 * 1024 + 4096
        )

    def test_nvmap_clients_dir_unreadable_is_undeterminable(self):
        self.tree.add_process(200, rss_kb=1024, pss_kb=512)
        # 目录存在但不可读：写一个文件冒充目录名，listdir 会抛 OSError。
        self.tree.write("sys/kernel/debug/nvmap/iovmm/clients", "not-a-dir\n")

        memory = self.collect({"worker": 200})

        self.assertIsNone(memory["components"]["nvmap_bytes"])
        self.assertEqual(memory["verdict"]["state"], "undeterminable")
        self.assertIn(
            "gpu_nvmap",
            [item["component"] for item in memory["undeterminable"]["reasons"]],
        )


# --------------------------------------------------------------------------
# 第二轮复核 2：共享映射与重复计数（自造 procfs 树）
# --------------------------------------------------------------------------
class SharedMappingTests(MemoryCase):
    def test_parent_child_shared_pss_is_not_added_twice(self):
        """父子共享 40 MiB：Rss 各算一次（上界重叠），Pss 各按一半分摊（去重）。"""
        self.tree.add_process(
            200, rss_kb=100 * 1024, pss_kb=60 * 1024, swap_kb=0, swap_pss_kb=0
        )
        self.tree.add_process(
            201, ppid=200, rss_kb=100 * 1024, pss_kb=60 * 1024, swap_kb=0, swap_pss_kb=0
        )
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200})

        self.assertEqual(memory["process_count"], 2)
        # 上界：200 MiB（共享页被计两次，这正是它只能当上界的原因）
        self.assertEqual(memory["conservative_upper_bound"]["bytes"], 200 * 1024 * 1024)
        # 去重：Pss 分摊后 120 MiB，不是 200 MiB，也不是 100 MiB
        self.assertEqual(memory["dedup_estimate"]["bytes"], 120 * 1024 * 1024)
        self.assertEqual(memory["components"]["pss_sum_bytes"], 120 * 1024 * 1024)
        self.assertLess(
            memory["dedup_estimate"]["bytes"],
            memory["conservative_upper_bound"]["bytes"],
        )

    def test_same_pid_under_two_roots_is_counted_once(self):
        self.tree.add_process(200, rss_kb=100 * 1024, pss_kb=60 * 1024)
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200, "model": 200})

        self.assertEqual(memory["process_count"], 1)
        self.assertEqual(memory["processes"][0]["pid"], 200)
        self.assertEqual(memory["components"]["rss_sum_bytes"], 100 * 1024 * 1024)

    def test_three_level_tree_is_counted_once_per_pid(self):
        self.tree.add_process(200, rss_kb=10 * 1024, pss_kb=8 * 1024)
        self.tree.add_process(201, ppid=200, rss_kb=10 * 1024, pss_kb=8 * 1024)
        self.tree.add_process(202, ppid=201, rss_kb=10 * 1024, pss_kb=8 * 1024)
        self.tree.add_nvmap_client("c0", 0)

        memory = self.collect({"worker": 200, "bench": 201})

        self.assertEqual(memory["process_count"], 3)
        self.assertEqual(
            sorted(item["pid"] for item in memory["processes"]), [200, 201, 202]
        )
        self.assertEqual(memory["components"]["rss_sum_bytes"], 30 * 1024 * 1024)
        self.assertEqual(
            {item["pid"]: item["depth"] for item in memory["processes"]},
            {200: 0, 201: 0, 202: 1},
        )


# --------------------------------------------------------------------------
# 第二轮复核 2：真实父子进程夹具（pid/ppid 取运行时真值）
# --------------------------------------------------------------------------
_DEEP_TREE_SNIPPET = r'''
import json, os, subprocess, sys, time
depth = int(sys.argv[1])
if depth > 0:
    child = subprocess.Popen(
        [sys.executable, sys.argv[0], str(depth - 1)],
        stdout=subprocess.PIPE, text=True,
    )
    info = json.loads(child.stdout.readline())
    info["chain"].append({"pid": os.getpid(), "ppid": os.getppid(), "child": child.pid})
    print(json.dumps(info), flush=True)
    time.sleep(6)
else:
    print(json.dumps({"chain": [{"pid": os.getpid(), "ppid": os.getppid(), "child": None}]}), flush=True)
    time.sleep(6)
'''


class RealProcessTreeTests(unittest.TestCase):
    """真实父子进程给出 pid/ppid；内存数值仍来自自造 smaps（本机无 /proc）。

    Windows 上 ``Popen([sys.executable, ...])`` 会插入 launcher 中间进程
    （父的 ``child.pid`` 与子的 ``os.getpid()`` 不是同一个数），所以真实的
    ppid 关系同时取自子进程自报的 ``os.getppid()`` 和父进程的 ``child.pid``，
    两者取并集才是一条完整、可核对的祖先链。
    """

    def _spawn_chain(self, depth=2):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        script = Path(tmp.name) / "chain.py"
        script.write_text(_DEEP_TREE_SNIPPET, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(script), str(depth)],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.addCleanup(self._kill, proc)
        line = proc.stdout.readline()
        self.assertTrue(line, "子进程没有输出 pid 链")
        return json.loads(line)["chain"]

    @staticmethod
    def _kill(proc):
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        # POSIX fixture children exit on their own after six seconds; wait for
        # the owned parent so no unrelated process is signalled.
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

    @staticmethod
    def _real_table(chain):
        """并集所有观测到的 ppid 边，得到不含虚构节点的真实表的子集。"""
        table = {}
        for item in chain:
            table.setdefault(item["pid"], item["ppid"])
            if item.get("child"):
                table.setdefault(item["child"], item["pid"])
        return table

    @staticmethod
    def _ancestry(table, leaf, stop):
        path = [leaf]
        while path[-1] != stop and table.get(path[-1]) not in (None, 0):
            nxt = table[path[-1]]
            if nxt in path or len(path) > 12:
                break
            path.append(nxt)
        return path

    def test_chain_walks_back_to_the_spawned_process(self):
        chain = self._spawn_chain(2)
        table = self._real_table(chain)

        path = self._ancestry(table, chain[0]["pid"], chain[-1]["pid"])

        self.assertEqual(path[-1], chain[-1]["pid"], f"祖先链断开：{path} vs table={table}")
        # 中间若插入 launcher，链会比 3 个 python 进程更长；但每一跳都必须是
        # 真实的 ppid，且最深的进程一定是链尾进程的后代。
        self.assertGreaterEqual(len(path), 3)

    def test_members_cover_the_whole_real_ancestry_once(self):
        chain = self._spawn_chain(2)
        table = self._real_table(chain)
        path = self._ancestry(table, chain[0]["pid"], chain[-1]["pid"])

        members = metrics._tree_members(table, [("worker", chain[-1]["pid"])])

        pids = [m["pid"] for m in members]
        self.assertEqual(len(pids), len(set(pids)), "同一 pid 被计了两次")
        self.assertEqual(sorted(pids), sorted(path))

    def test_two_roots_over_the_real_chain_claim_each_pid_once(self):
        chain = self._spawn_chain(2)
        table = self._real_table(chain)
        path = self._ancestry(table, chain[0]["pid"], chain[-1]["pid"])
        middle = path[len(path) // 2]

        members = metrics._tree_members(
            table, [("worker", chain[-1]["pid"]), ("model", middle)]
        )

        pids = [m["pid"] for m in members]
        self.assertEqual(len(pids), len(set(pids)))
        self.assertEqual(sorted(pids), sorted(path))

    def test_real_pid_topology_with_synthetic_numbers_covers_whole_chain(self):
        chain = self._spawn_chain(1)
        table = self._real_table(chain)
        path = self._ancestry(table, chain[0]["pid"], chain[-1]["pid"])
        tree = FakeTree()
        self.addCleanup(tree.cleanup)
        for pid in path:
            tree.add_process(
                pid, ppid=table.get(pid, 0), rss_kb=1024 * 1024, pss_kb=1024 * 512
            )
        tree.add_nvmap_client("c0", 0)

        memory = metrics.collect_memory_accounting(
            {"worker": chain[-1]["pid"]},
            proc_root=str(tree.proc),
            sys_root=str(tree.sys),
        )

        self.assertEqual(memory["process_count"], len(path))
        self.assertEqual(
            sorted(p["pid"] for p in memory["processes"]), sorted(path)
        )
        self.assertEqual(
            memory["components"]["rss_sum_bytes"], len(path) * 1024 * 1024 * 1024
        )

    def test_real_process_exit_mid_sample_is_a_gap_not_zero(self):
        chain = self._spawn_chain(1)
        pids = [item["pid"] for item in chain]
        tree = FakeTree()
        self.addCleanup(tree.cleanup)
        for item in chain:
            tree.add_process(item["pid"], ppid=item["ppid"], rss_kb=1024, pss_kb=512)
        tree.add_nvmap_client("c0", 0)
        # 真实退出：杀掉孙进程，再从自造树里删掉它的目录，模拟采样窗口内消失。
        if os.name == "nt":
            subprocess.run(["taskkill", "/F", "/PID", str(pids[0])], capture_output=True, check=False)
        else:
            import signal
            os.kill(pids[0], signal.SIGTERM)
        tree.remove_process(pids[0])

        memory = metrics.collect_memory_accounting(
            {"worker": pids[1]}, proc_root=str(tree.proc), sys_root=str(tree.sys)
        )

        # 树表里已经没有它了 → 归属不到；但被显式列出的 root 消失必须报 gap。
        self.assertEqual(memory["process_count"], 1)

        memory2 = metrics.collect_memory_accounting(
            {"worker": pids[1], "gone": pids[0]},
            proc_root=str(tree.proc),
            sys_root=str(tree.sys),
        )
        gone = [p for p in memory2["processes"] if p["pid"] == pids[0]][0]
        self.assertEqual(gone["state"], "exited")
        self.assertIn(
            "exited_during_sample",
            [gap["reason"] for gap in memory2["sampling_gaps"]],
        )
        self.assertIsNone(memory2["conservative_upper_bound"]["bytes"])
        self.assertEqual(memory2["verdict"]["state"], "undeterminable")


# --------------------------------------------------------------------------
# 第二轮复核 4：五类失败场景必须进 undeterminable 且原因可定位
# --------------------------------------------------------------------------
class FailureHonestyMatrixTests(MemoryCase):
    def _assert_undeterminable(
        self, memory, reason, component=None, *, upper_may_be_computed=False
    ):
        self.assertEqual(memory["verdict"]["state"], "undeterminable")
        self.assertTrue(memory["undeterminable"]["applicable"])
        reasons = [item["reason"] for item in memory["undeterminable"]["reasons"]]
        self.assertIn(reason, reasons)
        if component is not None:
            self.assertIn(
                component,
                [item["component"] for item in memory["undeterminable"]["reasons"]],
            )
        # 去重口径永远不能给数值；上界只有在 VmRSS/VmSwap 仍可读时才允许保留，
        # 但两者都不允许静默变成 0。
        self.assertIsNone(memory["dedup_estimate"]["bytes"])
        if upper_may_be_computed:
            self.assertIsNotNone(memory["conservative_upper_bound"]["bytes"])
        else:
            self.assertIsNone(memory["conservative_upper_bound"]["bytes"])
        self.assertFalse(memory["verdict"]["is_pass_conclusion"])

    def test_scenario_1_procfs_unreadable(self):
        missing = Path(tempfile.gettempdir()) / "_no_such_procfs_"
        memory = metrics.collect_memory_accounting(
            {"worker": 1}, proc_root=str(missing), sys_root=str(self.tree.sys)
        )
        self._assert_undeterminable(memory, "procfs_unreadable", "procfs")

    def test_scenario_2_smaps_missing_mandatory_field(self):
        for omitted in ("Rss", "Pss"):
            with self.subTest(omitted=omitted):
                self.tree.add_process(
                    300,
                    rss_kb=100 * 1024,
                    pss_kb=60 * 1024,
                    smaps_kwargs={"omit": (omitted,)},
                )
                self.tree.add_nvmap_client("c0", 0)
                memory = self.collect({"worker": 300})
                # smaps 缺字段 → 该进程退化为 rss_only：上界还能由 VmRSS 给出，
                # 但去重口径不可判定，且整轮不能给"在 cap 内"的结论。
                self._assert_undeterminable(
                    memory,
                    "smaps_rollup_unreadable",
                    "pss",
                    upper_may_be_computed=True,
                )
                self.assertEqual(memory["processes"][0]["state"], "rss_only")

    def test_scenario_3_pid_exited_during_sample(self):
        self.tree.add_process(400, rss_kb=1024, pss_kb=512)
        self.tree.add_nvmap_client("c0", 0)
        self.tree.remove_process(400)

        memory = self.collect({"worker": 400})

        self._assert_undeterminable(memory, "exited_during_sample", "process")

    def test_scenario_4_nvmap_directory_absent(self):
        self.tree.add_process(500, rss_kb=1024, pss_kb=512)

        memory = self.collect({"worker": 500})

        self._assert_undeterminable(memory, "nvmap_unreadable_or_absent", "gpu_nvmap")

    def test_scenario_5_editor_holding_the_model_log_is_not_a_model(self):
        """argv[0] 是 vim、只是打开了 llama-server.log —— 不是模型进程。"""
        self.tree.add_process(
            600,
            rss_kb=9000,
            pss_kb=4000,
            cmdline=["vim", "llama-server.log"],
        )
        self.tree.add_process(601, rss_kb=1024, pss_kb=512)
        self.tree.add_nvmap_client("c0", 0)

        model_pid = metrics.MetricsSampler(
            proc_root=str(self.tree.proc), sys_root=str(self.tree.sys)
        ).find_model_pid()
        self.assertIsNone(model_pid)

        # 只把探针自己当 root 时，600/601 都不该被算进来；同时因为模型不在树内，
        # 上层（mem_probe）必须把这一轮标成范围不全。
        memory = self.collect({"worker": 601})
        self.assertEqual([p["pid"] for p in memory["processes"]], [601])
        self.assertEqual(memory["components"]["rss_sum_bytes"], 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
