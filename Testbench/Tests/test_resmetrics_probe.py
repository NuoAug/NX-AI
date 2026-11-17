"""Tools/mem_probe.py 内存口径回归测试（D 组，清单第 39 项）。

只测纯 helper 与口径复用，不跑 main()（main 需要目标板的模型资产与音频设备）：
- /proc/<pid>/status 字段缺失时返回 None，绝不编造 0；
- 进程归属按 /proc/<pid>/cmdline 的 argv[0] 判定，不再解析 `ps -eo` 文本行；
- 账目计算复用 Testbench/metrics.py 的分层口径，缺失即 undeterminable；
- 输出字段只增不改（旧键名保留）。
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _resolve(*parts: str) -> Path:
    """按实际大小写解析路径：本机有 Tools/Testbench，目标板是小写 tools/testbench。"""
    current = _REPO
    for part in parts:
        candidate = current / part
        if candidate.exists():
            current = candidate
            continue
        if current.is_dir():
            matches = [p for p in current.iterdir() if p.name.lower() == part.lower()]
            if matches:
                current = matches[0]
                continue
        return candidate
    return current


# 允许指向修复前的副本（复现用）：AIEYES_MEMPROBE_PATH=/tmp/d_prefix_probe/mem_probe.py
_PROBE_PATH = Path(
    os.environ.get("AIEYES_MEMPROBE_PATH") or _resolve("Tools", "mem_probe.py")
)


def _load_probe():
    if str(_resolve("Testbench")) not in sys.path:
        sys.path.insert(0, str(_resolve("Testbench")))
    if str(_resolve("Src")) not in sys.path:
        sys.path.insert(0, str(_resolve("Src")))
    spec = importlib.util.spec_from_file_location("mem_probe_under_test", _PROBE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load_probe()

# 发到目标板的副本（NX-AI/ 是模型工具源；根 Tools/ 是部署副本）
_NXAI_PROBE_PATH = Path(
    os.environ.get("AIEYES_NXAI_MEMPROBE_PATH")
    or _resolve("NX-AI", "Tools", "mem_probe.py")
)
# 峰值口径副本：AIEYES_SOAK_PATH 可指向修复前的拷贝（复现用）
_SOAK_PATH = Path(os.environ.get("AIEYES_SOAK_PATH") or _resolve("Tools", "soak_test.py"))

#: 本机开发树：Tools/mem_probe.py 是复用 Testbench/metrics.py 的包装入口。
#: 目标板：只部署 tools/mem_probe.py，且它就是自足实现（交付清单没有 Testbench）。
PROBE_IS_WRAPPER = hasattr(probe, "_load_metrics_module")
NXAI_COPY_AVAILABLE = _NXAI_PROBE_PATH.is_file()
#: 自足实现副本：本机取自 NX-AI 源树；目标板取自已部署的 tools/mem_probe.py。
_SELF_CONTAINED_PATH = _NXAI_PROBE_PATH if NXAI_COPY_AVAILABLE else _PROBE_PATH
SELF_CONTAINED_AVAILABLE = NXAI_COPY_AVAILABLE or not PROBE_IS_WRAPPER
nx_probe_path = _SELF_CONTAINED_PATH


def _load_nx_probe():
    spec = importlib.util.spec_from_file_location("nxai_mem_probe_under_test", nx_probe_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_soak():
    """导入 Tools/soak_test.py。

    Windows 上 ``aieyes.vision`` 会 import fcntl（POSIX 专有）；被导入的函数
    （_summarize/_proc_kb/_server_peak_kb）都不走那条路径，因此对 POSIX 专有模块
    注入空 stub 后再导入，不改变任何被测行为。
    """
    for name in ("fcntl", "termios", "tty"):
        if name in sys.modules:
            continue
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - POSIX 专有，仅用于导入占位
            sys.modules[name] = types.ModuleType(name)
    if str(_resolve("Testbench")) not in sys.path:
        sys.path.insert(0, str(_resolve("Testbench")))
    spec = importlib.util.spec_from_file_location("soak_test_under_test", _SOAK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


nx_probe = _load_nx_probe() if SELF_CONTAINED_AVAILABLE else None
soak = _load_soak()


def stat_text(pid: int, ppid: int = 0) -> str:
    tail = ["S"] + ["0"] * 49
    tail[1] = str(ppid)
    return f"{pid} (proc) " + " ".join(tail) + "\n"


def status_text(rss_kb: int | None, hwm_kb: int | None) -> str:
    lines = ["Name:\tprobe\n"]
    if rss_kb is not None:
        lines.append(f"VmRSS:\t{rss_kb:8d} kB\n")
    if hwm_kb is not None:
        lines.append(f"VmHWM:\t{hwm_kb:8d} kB\n")
    return "".join(lines)


def smaps_text(rss_kb: int, pss_kb: int, swap_kb: int = 0) -> str:
    return (
        f"Rss:  {rss_kb:8d} kB\n"
        f"Pss:  {pss_kb:8d} kB\n"
        f"Pss_Anon: {pss_kb:8d} kB\n"
        f"Swap: {swap_kb:8d} kB\n"
        f"SwapPss: {swap_kb:8d} kB\n"
    )


class FakeTree:
    """最小 /proc + /sys 假树。"""

    def __init__(self, root: Path):
        self.root = root
        self.proc = root / "proc"
        self.sys = root / "sys"
        self.proc.mkdir(parents=True, exist_ok=True)
        self.sys.mkdir(parents=True, exist_ok=True)

    def add_process(
        self,
        pid: int,
        *,
        ppid: int = 0,
        argv0: str | None = None,
        rss_kb: int | None = 1024,
        hwm_kb: int | None = None,
        pss_kb: int | None = None,
    ) -> None:
        d = self.proc / str(pid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "stat").write_text(stat_text(pid, ppid), encoding="utf-8")
        (d / "status").write_text(status_text(rss_kb, hwm_kb), encoding="utf-8")
        if argv0 is not None:
            (d / "cmdline").write_bytes(argv0.encode() + b"\x00--port\x00123\x00")
        if pss_kb is not None:
            (d / "smaps_rollup").write_text(
                smaps_text(rss_kb or 0, pss_kb), encoding="utf-8"
            )

    def add_nvmap(self, allocated_bytes: int) -> None:
        client = self.sys / "kernel/debug/nvmap/iovmm/clients/c0"
        client.mkdir(parents=True, exist_ok=True)
        (client / "allocated").write_text(f"{allocated_bytes}\n", encoding="utf-8")


class StatusFieldTests(unittest.TestCase):
    def test_missing_field_is_none_not_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status"
            path.write_text(status_text(rss_kb=2048, hwm_kb=None), encoding="utf-8")
            self.assertEqual(probe._read_status_kb("VmRSS", path), 2048)
            self.assertEqual(probe._rss_kb(path), 2048)
            self.assertIsNone(probe._read_status_kb("VmHWM", path))
            self.assertIsNone(probe._peak_kb(path))

    def test_unreadable_file_is_none_not_exception(self):
        missing = Path(tempfile.gettempdir()) / "_no_such_status_file_"
        self.assertIsNone(probe._read_status_kb("VmRSS", missing))
        self.assertIsNone(probe._rss_kb(missing))

    def test_garbage_value_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "status"
            path.write_text("VmRSS:\tgarbage kB\nVmHWM:\n", encoding="utf-8")
            self.assertIsNone(probe._read_status_kb("VmRSS", path))
            self.assertIsNone(probe._read_status_kb("VmHWM", path))


class ProcessAttributionTests(unittest.TestCase):
    def test_server_pids_match_argv0_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = FakeTree(Path(tmp))
            tree.add_process(10, argv0="llama-server")           # 真服务
            tree.add_process(11, argv0="/opt/bin/llama-server")   # 绝对路径启动
            tree.add_process(12, argv0="bash")                    # 无关
            tree.add_process(13, argv0="vim")                     # 编辑日志的无关进程
            (tree.proc / "13" / "cmdline").write_bytes(
                b"vim\x00llama-server.log\x00"
            )
            (tree.proc / "14").mkdir()
            (tree.proc / "14" / "cmdline").write_bytes(b"")       # 僵尸/内核线程
            (tree.proc / "self").mkdir()                          # 非数字项
            self.assertEqual(probe._server_pids(tree.proc), [10, 11])

    def test_missing_proc_root_yields_empty_list(self):
        self.assertEqual(probe._server_pids(Path(tempfile.gettempdir()) / "_noproc_"), [])

    def test_sum_status_kb_is_none_when_any_pid_unreadable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = FakeTree(Path(tmp))
            tree.add_process(20, rss_kb=1024)
            tree.add_process(21, rss_kb=2048)
            self.assertEqual(
                probe._sum_status_kb([20, 21], "VmRSS", tree.proc), 3072
            )
            (tree.proc / "22").mkdir()
            (tree.proc / "22" / "status").write_text("Name:\tx\n", encoding="utf-8")
            self.assertIsNone(probe._sum_status_kb([20, 22], "VmRSS", tree.proc))
            self.assertIsNone(probe._sum_status_kb([], "VmRSS", tree.proc))

    def test_server_rss_does_not_fabricate_zero(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = FakeTree(Path(tmp))
            tree.add_process(30, argv0="llama-server", rss_kb=None)
            self.assertIsNone(probe._server_rss_kb(tree.proc))
            tree.add_process(30, argv0="llama-server", rss_kb=4096, hwm_kb=8192)
            self.assertEqual(probe._server_rss_kb(tree.proc), 4096)
            self.assertEqual(probe._server_peak_kb(tree.proc), 8192)


@unittest.skipUnless(PROBE_IS_WRAPPER, "目标板部署的是自足实现，不复用 Testbench/metrics")
class AccountingReuseTests(unittest.TestCase):
    def test_metrics_module_is_reused(self):
        module, reason = probe._load_metrics_module()
        self.assertIsNone(reason)
        self.assertTrue(hasattr(module, "collect_memory_accounting"))
        self.assertEqual(
            Path(module.__file__).resolve(),
            _resolve("Testbench", "metrics.py").resolve(),
        )

    def test_accounting_covers_tree_and_labels_tiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = FakeTree(Path(tmp))
            tree.add_process(40, argv0="llama-server", rss_kb=1024 * 100, hwm_kb=1024 * 120, pss_kb=1024 * 60)
            tree.add_process(41, ppid=40, rss_kb=1024 * 50, hwm_kb=1024 * 60, pss_kb=1024 * 30)
            tree.add_nvmap(1024 * 1024 * 10)
            account, verdict, module, reason = probe._memory_accounting(
                {"probe": 41, "llama_server:40": 40},
                proc_root=tree.proc,
                sys_root=tree.sys,
            )
            self.assertIsNone(reason)
            self.assertIsNotNone(account)
            self.assertEqual(account["process_count"], 2)
            self.assertEqual(
                account["conservative_upper_bound"]["label"], "conservative_upper_bound"
            )
            self.assertEqual(account["dedup_estimate"]["label"], "dedup_estimate")
            self.assertEqual(verdict["state"], "within_cap_upper_bound_only")
            self.assertFalse(verdict["is_pass_conclusion"])
            self.assertEqual(
                account["conservative_upper_bound"]["bytes"],
                (1024 * 150) * 1024 + 1024 * 1024 * 10,
            )
            self.assertEqual(
                account["dedup_estimate"]["bytes"], (1024 * 90) * 1024 + 1024 * 1024 * 10
            )

    def test_missing_gpu_memory_makes_accounting_undeterminable(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = FakeTree(Path(tmp))
            tree.add_process(50, argv0="llama-server", rss_kb=1024, pss_kb=512)
            account, verdict, _module, _reason = probe._memory_accounting(
                {"llama_server:50": 50}, proc_root=tree.proc, sys_root=tree.sys
            )
            self.assertTrue(account["undeterminable"]["applicable"])
            self.assertEqual(verdict["state"], "undeterminable")
            self.assertNotEqual(account["conservative_upper_bound"]["state"], "computed")

    def test_unreadable_procfs_is_undeterminable_not_zero(self):
        missing = Path(tempfile.gettempdir()) / "_noproc2_"
        account, verdict, _module, _reason = probe._memory_accounting(
            {"probe": 1}, proc_root=missing, sys_root=missing
        )
        self.assertEqual(verdict["state"], "undeterminable")
        self.assertIsNone(account["conservative_upper_bound"]["bytes"])
        self.assertIsNone(account["dedup_estimate"]["bytes"])
        self.assertIsNone(account["components"]["nvmap_bytes"])
        reasons = {r["reason"] for r in account["undeterminable"]["reasons"]}
        self.assertIn("procfs_unreadable", reasons)

    def test_gib_and_mib_pass_through_none(self):
        self.assertIsNone(probe._mib(None))
        self.assertIsNone(probe._gib(None))
        self.assertEqual(probe._mib(2048), 2.0)
        self.assertEqual(probe._gib(1024 ** 3), 1.0)


class SourceGuardTests(unittest.TestCase):
    def test_no_ps_text_matching_left(self):
        source = _PROBE_PATH.read_text(encoding="utf-8")
        self.assertNotIn('"rss,cmd"', source)
        self.assertNotIn('["ps"', source)
        self.assertNotIn("import subprocess", source)

    @unittest.skipUnless(
        PROBE_IS_WRAPPER, "自足实现有意移除了误导性的 within_cap 结论键（见 NxaiOutputContractTests）"
    )
    def test_legacy_output_keys_preserved(self):
        source = _PROBE_PATH.read_text(encoding="utf-8")
        for key in (
            "process_rss_mib",
            "process_peak_mib",
            "llama_server_rss_mib",
            "llama_server_peak_mib",
            "stack_total_gib",
            "stack_peak_gib",
            "cap_gib",
            "within_cap",
            "within_cap_peak",
            "marks_mib",
            "timings_ms",
            "answer",
            "answer_sources",
            "answer_intent",
        ):
            self.assertIn(f'"{key}":', source, f"旧输出键 {key} 被删除")


class ScopeHonestyTests(unittest.TestCase):
    """模型进程没纳管时，不能拿半棵树的数字去比 cap。"""

    def test_missing_model_process_is_a_scope_reason(self):
        reasons = probe._scope_reasons([], "/proc")
        self.assertEqual(len(reasons), 1)
        self.assertEqual(reasons[0]["component"], "model")
        self.assertEqual(reasons[0]["reason"], "llama_server_not_found_scope_incomplete")
        self.assertEqual(reasons[0]["path"], "/proc")

    def test_found_model_process_has_no_scope_reason(self):
        self.assertEqual(probe._scope_reasons([4242], "/proc"), [])

    def test_scope_incomplete_suppresses_the_screening_flag(self):
        reasons = probe._scope_reasons([], "/proc")
        self.assertIsNone(
            probe._screening_flag("within_cap_upper_bound_only", reasons)
        )
        self.assertIsNone(probe._screening_flag("over_cap", reasons))

    def test_screening_flag_maps_only_the_screening_states(self):
        self.assertIs(probe._screening_flag("within_cap_upper_bound_only", []), True)
        self.assertIs(probe._screening_flag("over_cap", []), False)
        for state in (
            "in_indeterminate_band",
            "undeterminable",
            None,
            "provably_within_cap",  # 旧状态名不再被接受
        ):
            self.assertIsNone(probe._screening_flag(state, []), state)

    def test_source_declares_the_screening_limits(self):
        source = _PROBE_PATH.read_text(encoding="utf-8")
        for needle in (
            '"within_cap_is_conclusion": False',
            '"within_cap_scope":',
            '"scope_incomplete_reasons":',
            '"verdict_state":',
            '"pass_requires":',
        ):
            self.assertIn(needle, source, f"缺少口径声明 {needle}")


@unittest.skipUnless(SELF_CONTAINED_AVAILABLE, "没有自足实现副本（仅本机开发树的包装入口）")
class NxaiParityTests(unittest.TestCase):
    """NX-AI/Tools/mem_probe.py（发到目标板的副本）与 metrics 同口径。

    本机无 /proc：两侧都跑同一棵自造 procfs 树，比较的是**语义与数值算法**，
    不是 Orin 实测。副本自足实现（不 import Testbench），因此必须逐项对齐。
    """

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tree = FakeTree(Path(self._tmp.name))
        self.addCleanup(self._tmp.cleanup)

    def _both(self, roots, **kwargs):
        import metrics  # Testbench/metrics.py（PYTHONPATH 已含 Testbench）

        args = dict(
            proc_root=str(self.tree.proc), sys_root=str(self.tree.sys), **kwargs
        )
        return (
            metrics.collect_memory_accounting(roots, **args),
            nx_probe.collect_memory_accounting(roots, **args),
        )

    def _assert_same_semantics(self, left, right):
        self.assertEqual(
            left["conservative_upper_bound"]["bytes"],
            right["conservative_upper_bound"]["bytes"],
            "上界数值不一致",
        )
        self.assertEqual(
            left["dedup_estimate"]["bytes"], right["dedup_estimate"]["bytes"], "去重数值不一致"
        )
        self.assertEqual(left["components"], right["components"], "分量不一致")
        self.assertEqual(left["verdict"]["state"], right["verdict"]["state"])
        self.assertEqual(left["verdict"]["reason"], right["verdict"]["reason"])
        for block in ("conservative_upper_bound", "dedup_estimate"):
            self.assertEqual(left[block]["label"], right[block]["label"])
            self.assertEqual(left[block]["usable_for"], right[block]["usable_for"])
            self.assertEqual(left[block]["state"], right[block]["state"])
        self.assertEqual(left["undeterminable"]["usable_for"], right["undeterminable"]["usable_for"])
        self.assertEqual(left["verdict"]["caveats"], right["verdict"]["caveats"])
        self.assertEqual(
            left["verdict"]["requires_for_pass"], right["verdict"]["requires_for_pass"]
        )
        self.assertIs(right["verdict"]["is_pass_conclusion"], False)

    def test_measured_tree_matches_metrics(self):
        self.tree.add_process(200, ppid=0, rss_kb=100 * 1024, pss_kb=60 * 1024)
        self.tree.add_process(201, ppid=200, rss_kb=100 * 1024, pss_kb=60 * 1024)
        self.tree.add_nvmap(10485760)
        left, right = self._both({"worker": 200})
        self._assert_same_semantics(left, right)
        self.assertEqual(right["process_count"], 2)
        self.assertEqual(right["verdict"]["state"], "within_cap_upper_bound_only")

    def test_rss_only_without_vmswap_field_is_undeterminable(self):
        # 无 smaps_rollup 且 status 没有 VmSwap 行 → 连上界也不可判定（缺测不当 0）
        self.tree.add_process(300, ppid=0, rss_kb=200 * 1024, hwm_kb=200 * 1024)
        self.tree.add_nvmap(4096)
        left, right = self._both({"bench": 300})
        self._assert_same_semantics(left, right)
        self.assertIsNone(right["conservative_upper_bound"]["bytes"])
        self.assertIsNone(right["dedup_estimate"]["bytes"])
        self.assertEqual(right["verdict"]["state"], "undeterminable")
        reasons = [item["reason"] for item in right["undeterminable"]["reasons"]]
        self.assertIn("vmswap_field_missing_or_unreadable", reasons)

    def test_rss_only_with_explicit_vmswap_zero_matches_metrics(self):
        # 显式 VmSwap: 0 kB 才算 0：此时上界可算，去重仍不可得
        d = self.tree.proc / "301"
        d.mkdir(parents=True)
        (d / "stat").write_text(stat_text(301, 0), encoding="utf-8")
        (d / "status").write_text(
            "Name:\tprobe\nVmRSS:\t 204800 kB\nVmHWM:\t 204800 kB\nVmSwap:\t 0 kB\n",
            encoding="utf-8",
        )
        self.tree.add_nvmap(4096)
        left, right = self._both({"bench": 301})
        self._assert_same_semantics(left, right)
        self.assertEqual(right["conservative_upper_bound"]["state"], "computed")
        self.assertEqual(
            right["conservative_upper_bound"]["bytes"], 204800 * 1024 + 4096
        )
        self.assertIsNone(right["dedup_estimate"]["bytes"])
        self.assertEqual(right["verdict"]["state"], "undeterminable")

    def test_exited_process_matches_metrics(self):
        self.tree.add_process(400, ppid=0, rss_kb=1024, pss_kb=512)
        self.tree.add_nvmap(0)
        roots = {"worker": 400, "model": 999}  # 999 不存在
        left, right = self._both(roots)
        self._assert_same_semantics(left, right)
        self.assertIsNone(right["dedup_estimate"]["bytes"])

    def test_missing_nvmap_matches_metrics(self):
        self.tree.add_process(500, ppid=0, rss_kb=1024, pss_kb=512)
        left, right = self._both({"worker": 500})
        self._assert_same_semantics(left, right)
        self.assertEqual(right["verdict"]["state"], "undeterminable")
        self.assertEqual(right["undeterminable"]["reasons"][0]["reason"], "nvmap_unreadable_or_absent")

    def test_swapless_smaps_matches_metrics(self):
        # smaps 有 Rss/Pss 但没有 Swap/SwapPss → 两条口径各自不可判定
        d = self.tree.proc / "600"
        d.mkdir(parents=True)
        (d / "stat").write_text(stat_text(600, 0), encoding="utf-8")
        (d / "status").write_text(status_text(1024, 1024), encoding="utf-8")
        (d / "smaps_rollup").write_text("Rss:      1024 kB\nPss:       512 kB\n", encoding="utf-8")
        self.tree.add_nvmap(0)
        left, right = self._both({"worker": 600})
        self._assert_same_semantics(left, right)
        self.assertIsNone(right["conservative_upper_bound"]["bytes"])
        self.assertIsNone(right["dedup_estimate"]["bytes"])


@unittest.skipUnless(SELF_CONTAINED_AVAILABLE, "没有自足实现副本（仅本机开发树的包装入口）")
class NxaiOutputContractTests(unittest.TestCase):
    """发到目标板的副本不得再输出“RSS 和当达标”的结论位。"""

    def test_misleading_pass_keys_are_gone(self):
        source = nx_probe_path.read_text(encoding="utf-8")
        self.assertNotIn('"within_cap":', source)
        self.assertNotIn('"within_cap_peak":', source)
        self.assertNotIn("total <= 4.0", source)
        self.assertNotIn("total_peak <= 4.0", source)

    def test_replacement_keys_are_labelled(self):
        source = nx_probe_path.read_text(encoding="utf-8")
        for needle in (
            '"stack_total_gib_method": "rss_naive_sum_screening_only"',
            '"stack_peak_gib_method": "rss_naive_sum_screening_only"',
            '"stack_total_gib_is_conclusion": False',
            '"stack_peak_gib_is_conclusion": False',
            '"within_cap_screening":',
            '"within_cap_is_conclusion": False',
            '"verdict_state":',
            '"scope_incomplete_reasons":',
            '"pass_requires":',
            '"is_pass_conclusion": False',
        ):
            self.assertIn(needle, source, f"缺少口径声明 {needle}")

    def test_no_ps_text_matching_and_argv0_attribution(self):
        source = nx_probe_path.read_text(encoding="utf-8")
        self.assertNotIn("import subprocess", source)
        self.assertNotIn('"rss,cmd"', source)
        self.assertIn("cmdline", source)
        self.assertIn("argv0", source)

    def test_screening_flag_refuses_a_pass_conclusion(self):
        reasons = nx_probe._scope_reasons([], "/proc")
        self.assertIsNone(nx_probe._screening_flag("within_cap_upper_bound_only", reasons))
        self.assertIs(nx_probe._screening_flag("within_cap_upper_bound_only", []), True)
        self.assertIs(nx_probe._screening_flag("over_cap", []), False)
        for state in ("in_indeterminate_band", "undeterminable", None, "provably_within_cap"):
            self.assertIsNone(nx_probe._screening_flag(state, []), state)

    def test_server_pids_ignores_editors_mentioning_the_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = FakeTree(Path(tmp))
            tree.add_process(700, argv0="/usr/bin/llama-server")
            tree.add_process(701, argv0="/usr/bin/vim")
            # vim 打开 llama-server.log：cmdline 里含关键字，但 argv[0] 不是模型
            (tree.proc / "701" / "cmdline").write_bytes(
                b"/usr/bin/vim\x00llama-server.log\x00"
            )
            self.assertEqual(nx_probe._server_pids(str(tree.proc)), [700])


class SoakPeakAccountingTests(unittest.TestCase):
    """Tools/soak_test.py 的峰值口径：RSS 朴素和必须带标签，缺测不得当 0。"""

    def test_missing_status_field_is_none(self):
        self.assertIsNone(soak._proc_kb("NoSuchField"))

    def test_naive_sum_is_labelled_not_a_conclusion(self):
        summary = soak._summarize(
            [], 1.0, {"peak_kb": 200 * 1024, "samples": 3, "missing_samples": 0}, 300 * 1024
        )
        self.assertEqual(summary["stack_peak_gib"], 0.488)
        self.assertEqual(summary["stack_peak_gib_method"], "rss_naive_sum_screening_only")
        self.assertIs(summary["stack_peak_gib_is_conclusion"], False)
        self.assertEqual(
            summary["stack_peak_gib_scope"], "probe_process_plus_llama_server_rss_only"
        )
        self.assertIn("nvmap_gpu_buffers_not_included", summary["stack_peak_gib_caveats"])
        self.assertEqual(summary["rss_sample_counts"], {"samples": 3, "missing_samples": 0})

    def test_unknown_component_never_becomes_zero(self):
        summary = soak._summarize(
            [], 1.0, {"peak_kb": None, "samples": 0, "missing_samples": 4}, None
        )
        self.assertIsNone(summary["stack_peak_gib"])
        self.assertIsNone(summary["proc_rss_peak_mib"])
        self.assertIsNone(summary["server_peak_mib"])

    def test_server_peak_sums_all_model_processes(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = FakeTree(Path(tmp))
            tree.add_process(800, argv0="/usr/bin/llama-server", rss_kb=1024, hwm_kb=200 * 1024)
            tree.add_process(801, argv0="/opt/llama-server", rss_kb=1024, hwm_kb=100 * 1024)
            self.assertEqual(soak._server_peak_kb(str(tree.proc)), 300 * 1024)

    def test_server_peak_absent_is_none(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree = FakeTree(Path(tmp))
            tree.add_process(900, argv0="/usr/bin/vim")
            self.assertIsNone(soak._server_peak_kb(str(tree.proc)))
            self.assertIsNone(soak._server_peak_kb(str(Path(tmp) / "missing")))

    def test_source_has_no_pgrep_and_no_default_zero(self):
        source = _SOAK_PATH.read_text(encoding="utf-8")
        self.assertNotIn('"pgrep"', source)
        self.assertNotIn('[l]lama-server', source)
        self.assertNotIn('rss.get("peak_kb", 0)', source)


@unittest.skipUnless(NXAI_COPY_AVAILABLE, "目标板无 NX-AI 源树，仅校验部署副本")
class SoakCopiesIdenticalTests(unittest.TestCase):
    """根 Tools/ 与发到目标板的 NX-AI/Tools/ 副本必须字节一致（防止口径漂移）。"""

    def test_two_copies_are_byte_identical(self):
        root_copy = _resolve("Tools", "soak_test.py").read_bytes()
        shipped_copy = _resolve("NX-AI", "Tools", "soak_test.py").read_bytes()
        self.assertEqual(
            root_copy,
            shipped_copy,
            "Tools/soak_test.py 与 NX-AI/Tools/soak_test.py 已漂移，请同步",
        )


@unittest.skipUnless(
    NXAI_COPY_AVAILABLE and PROBE_IS_WRAPPER, "需要开发树的包装入口与 NX-AI 自足副本同时存在"
)
class EntryParityAcrossCopiesTests(unittest.TestCase):
    """两条 mem_probe 入口的判定语义必须逐项相等（不要求字节相同）。

    根 ``Tools/mem_probe.py`` 复用 ``Testbench/metrics.py``（本机有 Testbench），
    ``NX-AI/Tools/mem_probe.py`` 是板上自足实现（交付清单里没有 Testbench）。
    两者字节必然不同，但同一 fixture 下的 upper / dedup / verdict / reason /
    components 必须一致，否则本机测出来的口径不能代表板上。
    """

    _STAT = (
        "9001 (llama-server) S 1 9001 9001 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 1 0 100 0\n"
    )

    def _fake_tree(self, root: Path):
        proc = root / "proc"
        sysroot = root / "sys"
        pid_dir = proc / "9001"
        pid_dir.mkdir(parents=True, exist_ok=True)
        (pid_dir / "stat").write_text(self._STAT, encoding="utf-8")
        (pid_dir / "status").write_text(
            "Name:\tllama-server\nVmRSS:\t  102400 kB\nVmHWM:\t  122880 kB\n"
            "VmSwap:\t    8192 kB\n",
            encoding="utf-8",
        )
        (pid_dir / "smaps_rollup").write_text(
            "Rss:              102400 kB\nPss:               61440 kB\n"
            "Pss_Anon:          61440 kB\nSwap:               8192 kB\n"
            "SwapPss:            4096 kB\n",
            encoding="utf-8",
        )
        (pid_dir / "cmdline").write_text("llama-server\0--port\0 8080\0", encoding="utf-8")
        nvmap = sysroot / "kernel/debug/nvmap/iovmm"
        nvmap.mkdir(parents=True, exist_ok=True)
        (nvmap / "allocated").write_text("10485760\n", encoding="utf-8")
        return proc, sysroot

    def test_same_fixture_same_semantics(self):
        nx_probe = _load_nx_probe()
        with tempfile.TemporaryDirectory() as tmp:
            proc, sysroot = self._fake_tree(Path(tmp))
            roots = {"llama_server:9001": 9001}
            account, verdict, _module, reason = probe._memory_accounting(
                roots, proc_root=proc, sys_root=sysroot
            )
            nx_account = nx_probe.collect_memory_accounting(
                roots, proc_root=str(proc), sys_root=str(sysroot)
            )

        self.assertIsNone(reason)
        self.assertEqual(
            account["conservative_upper_bound"]["bytes"],
            nx_account["conservative_upper_bound"]["bytes"],
            "两条入口的保守上界不一致",
        )
        self.assertEqual(
            account["dedup_estimate"]["bytes"],
            nx_account["dedup_estimate"]["bytes"],
            "两条入口的去重估计不一致",
        )
        self.assertEqual(account["components"], nx_account["components"])
        self.assertEqual(verdict["state"], nx_account["verdict"]["state"])
        self.assertEqual(verdict.get("reason"), nx_account["verdict"].get("reason"))
        self.assertEqual(
            account["undeterminable"]["reasons"],
            nx_account["undeterminable"]["reasons"],
        )
        for block in (verdict, nx_account["verdict"]):
            self.assertFalse(block["is_pass_conclusion"])
            self.assertEqual(block["scope"], "sampled_process_tree_plus_nvmap_only")

    def test_two_entry_points_do_not_require_identical_bytes(self):
        """两条入口定位不同（自足 vs 复用），字节异同不作判据。

        判据只有三条：入口可用性、语义 parity、缺测诚实性；字节是否相同与本组无关，
        也不再断言"必须不同"（主控 R6 裁定）。
        """
        root_source = _resolve("Tools", "mem_probe.py").read_text(encoding="utf-8")
        nx_source = _resolve("NX-AI", "Tools", "mem_probe.py").read_text(encoding="utf-8")
        self.assertIn('"Testbench"', root_source)
        self.assertNotIn("from metrics import", nx_source)
        self.assertIn("def collect_memory_accounting", nx_source)


class MinimalEntryTests(unittest.TestCase):
    """追加任务 B：缺 Testbench / tempdir 不可写时，入口必须"明确不可判定"，不许静默 0。"""

    def _minimal_root(self, tmp: Path) -> Path:
        (tmp / "Tools").mkdir(parents=True, exist_ok=True)
        (tmp / "Tools" / "mem_probe.py").write_bytes(
            _resolve("Tools", "mem_probe.py").read_bytes()
        )
        return tmp

    def _fixture_tree(self, root: Path):
        proc = root / "proc"
        sysroot = root / "sys"
        pid_dir = proc / "9001"
        pid_dir.mkdir(parents=True, exist_ok=True)
        (pid_dir / "stat").write_text(
            "9001 (llama-server) S 1 9001 9001 0 -1 0 0 0 0 0 0 0 0 0 20 0 1 0 1 0 100 0\n",
            encoding="utf-8",
        )
        (pid_dir / "status").write_text(
            "Name:\tllama-server\nVmRSS:\t  102400 kB\nVmHWM:\t  122880 kB\n",
            encoding="utf-8",
        )
        (pid_dir / "smaps_rollup").write_text(
            "Rss:              102400 kB\nPss:               61440 kB\n", encoding="utf-8"
        )
        (pid_dir / "cmdline").write_text("llama-server\0", encoding="utf-8")
        (sysroot / "kernel/debug/nvmap/iovmm").mkdir(parents=True, exist_ok=True)
        (sysroot / "kernel/debug/nvmap/iovmm/allocated").write_text(
            "10485760\n", encoding="utf-8"
        )
        return proc, sysroot

    @unittest.skipUnless(PROBE_IS_WRAPPER, "自足实现本就不依赖 Testbench，无需该退化路径")
    def test_missing_testbench_is_explicit_undeterminable(self):
        import json
        import subprocess

        with tempfile.TemporaryDirectory() as tmp:
            root = self._minimal_root(Path(tmp))
            proc, sysroot = self._fixture_tree(root)
            script = root / "call.py"
            script.write_text(
                "import importlib.util, json, sys\n"
                "spec = importlib.util.spec_from_file_location('mp', 'Tools/mem_probe.py')\n"
                "m = importlib.util.module_from_spec(spec); sys.argv=['mp']\n"
                "spec.loader.exec_module(m)\n"
                f"acc, verdict, _mod, reason = m._memory_accounting({{'llama_server:9001': 9001}},"
                f" proc_root=r'{proc}', sys_root=r'{sysroot}')\n"
                "print(json.dumps({'account': acc, 'verdict': verdict, 'reason': reason}))\n",
                encoding="utf-8",
            )
            env = {
                **os.environ,
                "PYTHONPATH": str(_resolve("Src")),
                "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            done = subprocess.run(
                [sys.executable, "-B", str(script)],
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
                # 子进程按 PYTHONUTF8=1 输出 UTF-8；父进程 text=True 默认读 locale（Windows 为 GBK），
                # 不显式指定会 UnicodeDecodeError 让 stdout=None，断言变成 TypeError（与被测代码无关）。
                encoding="utf-8",
                errors="replace",
                timeout=120,
            )
            self.assertEqual(done.returncode, 0, done.stderr[-800:])
            payload = json.loads(done.stdout.strip().splitlines()[-1])

        self.assertIsNone(payload["account"], "缺 Testbench 时不该编造账目")
        self.assertEqual(payload["verdict"]["state"], "undeterminable")
        self.assertIn("No module named 'metrics'", payload["reason"])

    def test_tools_do_not_use_tempdir(self):
        for rel in (
            "Tools/mem_probe.py",
            "NX-AI/Tools/mem_probe.py",
            "Tools/soak_test.py",
            "NX-AI/Tools/soak_test.py",
        ):
            path = _resolve(*rel.split("/"))
            if not path.is_file():
                # 目标板没有 NX-AI 源树；部署副本由 Tools/ 两条用例覆盖
                continue
            source = path.read_text(encoding="utf-8")
            self.assertNotIn("tempfile", source, f"{rel} 不应依赖 tempfile")
            self.assertNotIn("mkdtemp", source, f"{rel} 不应自造临时目录")

    def test_soak_entry_runs_without_pythonpath(self):
        """Windows 本机没有 fcntl；soak 入口必须自带占位，否则 --help 都跑不起来。"""
        import subprocess

        env = {
            k: v
            for k, v in os.environ.items()
            if k.upper() not in ("PYTHONPATH", "PYTHONHOME")
        }
        env["PYTHONUTF8"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        done = subprocess.run(
            [sys.executable, "-B", str(_resolve("Tools", "soak_test.py")), "--help"],
            cwd=str(_REPO),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",   # 同上：子进程是 UTF-8，父进程默认 locale 会解码失败
            errors="replace",
            timeout=120,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        self.assertIn("--minutes", done.stdout)


if __name__ == "__main__":
    unittest.main()
