"""Unit tests for Testbench/metrics.py.

Standard-library ``unittest`` only.  Fake procfs / sysfs trees are built in
temporary directories so nothing outside the test directory is read for the
process metrics.

Run from the repository root (no bytecode written)::

    python -B -m unittest discover -s Testbench/Tests -p "test_metrics.py" -v
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import metrics


# --------------------------------------------------------------------------
# Fake tree helpers
# --------------------------------------------------------------------------
class FakeClock:
    """Replacement for the module-level ``time`` object."""

    def __init__(self, monotonic=1000.0, wall=1_700_000_000.0):
        self._monotonic = monotonic
        self._wall = wall

    def time(self):
        return self._wall

    def monotonic(self):
        return self._monotonic

    def advance(self, seconds):
        self._monotonic += seconds

    def advance_wall(self, seconds):
        self._wall += seconds


def stat_text(pid, utime, stime, comm="proc"):
    """Build a 52-field /proc/<pid>/stat line with the right utime/stime."""
    # After ") ": index 0 = field 3 (state), so utime (field 14) is index 11
    # and stime (field 15) is index 12.
    tail = ["S"] + ["0"] * 49
    tail[11] = str(utime)
    tail[12] = str(stime)
    return f"{pid} ({comm}) " + " ".join(tail) + "\n"


def status_text(rss_kb=None):
    lines = ["Name:\tpython", "State:\tS (sleeping)", "Threads:\t1"]
    if rss_kb is not None:
        lines.append(f"VmRSS:\t{rss_kb:8d} kB")
    return "\n".join(lines) + "\n"


class FakeTree:
    """Temporary directory shaped like procfs and sysfs."""

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
        utime=0,
        stime=0,
        rss_kb=1024,
        cmdline=None,
        comm="proc",
        with_stat=True,
        with_status=True,
    ):
        pid_dir = self.proc / str(pid)
        pid_dir.mkdir(parents=True, exist_ok=True)
        if with_stat:
            self.write(
                f"proc/{pid}/stat", stat_text(pid, utime, stime, comm)
            )
        if with_status:
            self.write(f"proc/{pid}/status", status_text(rss_kb))
        if cmdline is not None:
            payload = b"".join(
                argument.encode("utf-8") + b"\x00" for argument in cmdline
            )
            self.write(f"proc/{pid}/cmdline", payload, binary=True)
        return pid_dir

    def remove_process(self, pid):
        shutil.rmtree(self.proc / str(pid), ignore_errors=True)


class MetricsTestCase(unittest.TestCase):
    def setUp(self):
        self.tree = FakeTree()
        self.addCleanup(self.tree.cleanup)
        self.clock = FakeClock()

    def make_sampler(self, **kwargs):
        kwargs.setdefault("proc_root", str(self.tree.proc))
        kwargs.setdefault("sys_root", str(self.tree.sys))
        return metrics.MetricsSampler(**kwargs)

    def patched_clock(self):
        return mock.patch.object(metrics, "time", self.clock)

    @staticmethod
    def cpu_percent_for(delta_ticks, elapsed_seconds):
        """Same normalization the sampler promises: 100 % = all cores."""
        cpu_count = os.cpu_count() or 1
        value = delta_ticks / metrics._CLOCK_TICKS / elapsed_seconds
        return round(value / cpu_count * 100.0, 2)


# --------------------------------------------------------------------------
# Snapshot shape / robustness
# --------------------------------------------------------------------------
class SnapshotTests(MetricsTestCase):
    def test_snapshot_shape_and_sampled_at(self):
        sampler = self.make_sampler()
        with self.patched_clock():
            snapshot = sampler.sample()

        self.assertEqual(
            set(snapshot),
            {
                "sampled_at",
                "worker_rss_bytes",
                "model_rss_bytes",
                "bench_rss_bytes",
                "cpu_percent",
                "gpu_percent",
                "temperature_c",
                "detail",
            },
        )
        self.assertEqual(
            set(snapshot["detail"]),
            {
                "worker_cpu_percent",
                "model_cpu_percent",
                "bench_cpu_percent",
            },
        )
        self.assertEqual(snapshot["sampled_at"], int(1_700_000_000.0 * 1000))
        for key in (
            "worker_rss_bytes",
            "model_rss_bytes",
            "bench_rss_bytes",
            "cpu_percent",
            "gpu_percent",
            "temperature_c",
        ):
            self.assertIsNone(snapshot[key], key)

    def test_garbage_files_do_not_raise(self):
        self.tree.add_process(123)
        self.tree.write("proc/123/stat", "garbage")
        self.tree.write("proc/123/status", "garbage")
        self.tree.write("sys/devices/gpu.0/load", "not-a-number\n")
        self.tree.write("sys/class/thermal/thermal_zone0/temp", "oops\n")
        sampler = self.make_sampler(worker_pid_provider=lambda: 123)

        snapshot = sampler.sample()

        self.assertIsNone(snapshot["worker_rss_bytes"])
        self.assertIsNone(snapshot["detail"]["worker_cpu_percent"])
        self.assertIsNone(snapshot["gpu_percent"])
        self.assertIsNone(snapshot["temperature_c"])

    def test_internal_failure_falls_back_to_previous_structure(self):
        self.tree.add_process(77, rss_kb=512)
        sampler = self.make_sampler(worker_pid_provider=lambda: 77)
        with self.patched_clock():
            first = sampler.sample()
            self.clock.advance_wall(1.0)
            with mock.patch.object(
                sampler, "_collect", side_effect=RuntimeError("boom")
            ):
                second = sampler.sample()

        self.assertEqual(second["worker_rss_bytes"], 512 * 1024)
        self.assertEqual(
            second["sampled_at"], first["sampled_at"] + 1000
        )
        self.assertIsNone(second["cpu_percent"])

    def test_provider_exception_does_not_raise(self):
        def broken_provider():
            raise RuntimeError("provider failed")

        sampler = self.make_sampler(worker_pid_provider=broken_provider)
        snapshot = sampler.sample()

        self.assertIsNone(snapshot["worker_rss_bytes"])
        self.assertIsNone(snapshot["detail"]["worker_cpu_percent"])

    def test_invalid_provider_values_ignored(self):
        for bad_value in ("abc", -1, 0, None):
            sampler = self.make_sampler(
                worker_pid_provider=lambda value=bad_value: value
            )
            snapshot = sampler.sample()
            self.assertIsNone(snapshot["worker_rss_bytes"])


# --------------------------------------------------------------------------
# RSS
# --------------------------------------------------------------------------
class RssTests(MetricsTestCase):
    def test_vmrss_kb_converted_to_bytes(self):
        self.tree.add_process(4242, rss_kb=2048)
        sampler = self.make_sampler(worker_pid_provider=lambda: 4242)

        snapshot = sampler.sample()

        self.assertEqual(snapshot["worker_rss_bytes"], 2048 * 1024)
        self.assertIsNone(snapshot["model_rss_bytes"])
        self.assertIsNone(snapshot["bench_rss_bytes"])

    def test_missing_process_returns_none(self):
        sampler = self.make_sampler(worker_pid_provider=lambda: 9999)

        snapshot = sampler.sample()

        self.assertIsNone(snapshot["worker_rss_bytes"])
        self.assertIsNone(snapshot["detail"]["worker_cpu_percent"])

    def test_status_without_vmrss_returns_none(self):
        self.tree.add_process(4242, rss_kb=None)
        sampler = self.make_sampler(worker_pid_provider=lambda: 4242)

        self.assertIsNone(sampler.sample()["worker_rss_bytes"])

    def test_process_without_status_file_returns_none(self):
        self.tree.add_process(4242, with_status=False)
        sampler = self.make_sampler(worker_pid_provider=lambda: 4242)

        self.assertIsNone(sampler.sample()["worker_rss_bytes"])


# --------------------------------------------------------------------------
# CPU
# --------------------------------------------------------------------------
class CpuTests(MetricsTestCase):
    def test_first_sample_cpu_is_none(self):
        self.tree.add_process(101, utime=10, stime=5)
        sampler = self.make_sampler(worker_pid_provider=lambda: 101)

        with self.patched_clock():
            snapshot = sampler.sample()

        self.assertIsNone(snapshot["cpu_percent"])
        self.assertIsNone(snapshot["detail"]["worker_cpu_percent"])

    def test_second_sample_uses_delta_and_normalizes_by_cpu_count(self):
        # Two cores worth of ticks over 2 s == 1 % of a fully busy machine,
        # independent of the real core count.
        cpu_count = os.cpu_count() or 1
        delta_ticks = 2 * cpu_count
        self.tree.add_process(101, utime=100, stime=50)
        sampler = self.make_sampler(worker_pid_provider=lambda: 101)

        with self.patched_clock():
            sampler.sample()
            self.clock.advance(2.0)
            self.tree.add_process(
                101, utime=100 + delta_ticks, stime=50
            )
            snapshot = sampler.sample()

        expected = self.cpu_percent_for(delta_ticks, 2.0)
        self.assertEqual(expected, 1.0)
        self.assertEqual(snapshot["detail"]["worker_cpu_percent"], expected)
        self.assertEqual(snapshot["cpu_percent"], expected)

    def test_measured_interval_used_for_delta(self):
        cpu_count = os.cpu_count() or 1
        delta_ticks = 2 * cpu_count
        self.tree.add_process(101, utime=0, stime=0)
        sampler = self.make_sampler(worker_pid_provider=lambda: 101)

        with self.patched_clock():
            sampler.sample()
            self.clock.advance(4.0)
            self.tree.add_process(101, utime=delta_ticks, stime=0)
            snapshot = sampler.sample()

        self.assertEqual(
            snapshot["detail"]["worker_cpu_percent"],
            self.cpu_percent_for(delta_ticks, 4.0),
        )

    def test_duplicate_pid_is_counted_once(self):
        cpu_count = os.cpu_count() or 1
        delta_ticks = 2 * cpu_count
        self.tree.add_process(999, utime=0, stime=0)
        sampler = self.make_sampler(
            worker_pid_provider=lambda: 999,
            model_pid_provider=lambda: 999,
            bench_pid=999,
        )

        with self.patched_clock():
            sampler.sample()
            self.clock.advance(2.0)
            self.tree.add_process(999, utime=delta_ticks, stime=0)
            snapshot = sampler.sample()

        expected = self.cpu_percent_for(delta_ticks, 2.0)
        self.assertEqual(snapshot["detail"]["worker_cpu_percent"], expected)
        self.assertIsNone(snapshot["detail"]["model_cpu_percent"])
        self.assertIsNone(snapshot["detail"]["bench_cpu_percent"])
        self.assertEqual(snapshot["cpu_percent"], expected)
        # RSS is a per-role property: all three roles report the same process.
        self.assertEqual(snapshot["worker_rss_bytes"], 1024 * 1024)
        self.assertEqual(snapshot["model_rss_bytes"], 1024 * 1024)
        self.assertEqual(snapshot["bench_rss_bytes"], 1024 * 1024)

    def test_process_disappearing_returns_none(self):
        self.tree.add_process(101, utime=0, stime=0)
        sampler = self.make_sampler(worker_pid_provider=lambda: 101)

        with self.patched_clock():
            sampler.sample()
            self.clock.advance(2.0)
            self.tree.remove_process(101)
            snapshot = sampler.sample()

        self.assertIsNone(snapshot["worker_rss_bytes"])
        self.assertIsNone(snapshot["detail"]["worker_cpu_percent"])
        self.assertIsNone(snapshot["cpu_percent"])

    def test_pid_change_resets_cpu_baseline(self):
        current = {"pid": 101}
        self.tree.add_process(101, utime=0, stime=0)
        sampler = self.make_sampler(
            worker_pid_provider=lambda: current["pid"]
        )

        with self.patched_clock():
            sampler.sample()
            self.tree.add_process(202, utime=0, stime=0)
            current["pid"] = 202
            self.clock.advance(2.0)
            snapshot = sampler.sample()

        self.assertEqual(snapshot["worker_rss_bytes"], 1024 * 1024)
        self.assertIsNone(snapshot["detail"]["worker_cpu_percent"])
        self.assertIsNone(snapshot["cpu_percent"])

    def test_stat_comm_with_spaces_and_parens_parsed(self):
        cpu_count = os.cpu_count() or 1
        delta_ticks = 2 * cpu_count
        comm = "weird (proc) name"
        self.tree.add_process(101, utime=0, stime=0, comm=comm)
        sampler = self.make_sampler(worker_pid_provider=lambda: 101)

        with self.patched_clock():
            sampler.sample()
            self.clock.advance(2.0)
            self.tree.add_process(
                101, utime=delta_ticks, stime=0, comm=comm
            )
            snapshot = sampler.sample()

        self.assertEqual(
            snapshot["detail"]["worker_cpu_percent"],
            self.cpu_percent_for(delta_ticks, 2.0),
        )

    def test_worker_and_model_roles_are_summed(self):
        cpu_count = os.cpu_count() or 1
        delta_ticks = 2 * cpu_count
        self.tree.add_process(11, utime=0, stime=0)
        self.tree.add_process(22, utime=0, stime=0)
        sampler = self.make_sampler(
            worker_pid_provider=lambda: 11,
            model_pid_provider=lambda: 22,
            bench_pid=33,
        )

        with self.patched_clock():
            sampler.sample()
            self.clock.advance(2.0)
            self.tree.add_process(11, utime=delta_ticks, stime=0)
            self.tree.add_process(22, utime=delta_ticks, stime=0)
            snapshot = sampler.sample()

        each = self.cpu_percent_for(delta_ticks, 2.0)
        self.assertEqual(snapshot["detail"]["worker_cpu_percent"], each)
        self.assertEqual(snapshot["detail"]["model_cpu_percent"], each)
        self.assertIsNone(snapshot["detail"]["bench_cpu_percent"])
        self.assertEqual(snapshot["cpu_percent"], round(each * 2, 2))


# --------------------------------------------------------------------------
# GPU
# --------------------------------------------------------------------------
class GpuTests(MetricsTestCase):
    def test_missing_gpu_nodes_return_none(self):
        sampler = self.make_sampler()
        self.assertIsNone(sampler.sample()["gpu_percent"])

    def test_orin_gpu_load_scaled_from_0_1000(self):
        self.tree.write("sys/devices/gpu.0/load", "250\n")
        sampler = self.make_sampler()
        self.assertEqual(sampler.sample()["gpu_percent"], 25.0)

    def test_orin_gpu_load_clamped_to_100(self):
        self.tree.write("sys/devices/gpu.0/load", "1500\n")
        sampler = self.make_sampler()
        self.assertEqual(sampler.sample()["gpu_percent"], 100.0)

    def test_devfreq_load_supported(self):
        self.tree.write("sys/class/devfreq/gpu/load", "1000\n")
        sampler = self.make_sampler()
        self.assertEqual(sampler.sample()["gpu_percent"], 100.0)

    def test_drm_gpu_busy_percent_supported(self):
        self.tree.write(
            "sys/class/drm/card0/device/gpu_busy_percent", "37\n"
        )
        sampler = self.make_sampler()
        self.assertEqual(sampler.sample()["gpu_percent"], 37.0)

    def test_unparseable_gpu_node_returns_none(self):
        self.tree.write("sys/devices/gpu.0/load", "not-a-number\n")
        sampler = self.make_sampler()
        self.assertIsNone(sampler.sample()["gpu_percent"])


# --------------------------------------------------------------------------
# Temperature
# --------------------------------------------------------------------------
class TemperatureTests(MetricsTestCase):
    def test_missing_thermal_zones_return_none(self):
        sampler = self.make_sampler()
        self.assertIsNone(sampler.sample()["temperature_c"])

    def test_millidegrees_converted_and_gpu_zone_preferred(self):
        self.tree.write(
            "sys/class/thermal/thermal_zone0/type", "soc-thermal\n"
        )
        self.tree.write(
            "sys/class/thermal/thermal_zone0/temp", "50000\n"
        )
        self.tree.write(
            "sys/class/thermal/thermal_zone1/type", "gpu-thermal\n"
        )
        self.tree.write(
            "sys/class/thermal/thermal_zone1/temp", "45000\n"
        )
        sampler = self.make_sampler()

        self.assertEqual(sampler.sample()["temperature_c"], 45.0)

    def test_non_preferred_zone_used_as_fallback(self):
        self.tree.write(
            "sys/class/thermal/thermal_zone0/type", "soc0\n"
        )
        self.tree.write(
            "sys/class/thermal/thermal_zone0/temp", "41000\n"
        )
        sampler = self.make_sampler()

        self.assertEqual(sampler.sample()["temperature_c"], 41.0)

    def test_zone_without_type_used_as_fallback(self):
        self.tree.write(
            "sys/class/thermal/thermal_zone0/temp", "33000\n"
        )
        sampler = self.make_sampler()

        self.assertEqual(sampler.sample()["temperature_c"], 33.0)

    def test_unparseable_temperature_returns_none(self):
        self.tree.write(
            "sys/class/thermal/thermal_zone0/type", "gpu-thermal\n"
        )
        self.tree.write(
            "sys/class/thermal/thermal_zone0/temp", "oops\n"
        )
        sampler = self.make_sampler()

        self.assertIsNone(sampler.sample()["temperature_c"])


# --------------------------------------------------------------------------
# find_model_pid
# --------------------------------------------------------------------------
class FindModelPidTests(MetricsTestCase):
    def test_finds_llama_server_executable(self):
        self.tree.add_process(
            321,
            cmdline=[
                "/opt/llama.cpp/build/bin/llama-server",
                "-m",
                "/models/qwen.gguf",
            ],
        )
        self.assertEqual(self.make_sampler().find_model_pid(), 321)

    def test_ignores_other_processes(self):
        self.tree.add_process(
            321, cmdline=["/usr/bin/python3", "aieyes.main"]
        )
        self.assertIsNone(self.make_sampler().find_model_pid())

    def test_process_without_cmdline_is_skipped(self):
        pid_dir = self.tree.add_process(321, with_stat=False, with_status=False)
        self.assertTrue(pid_dir.is_dir())
        self.assertIsNone(self.make_sampler().find_model_pid())

    def test_empty_cmdline_is_skipped(self):
        self.tree.add_process(321, cmdline=[])
        self.assertIsNone(self.make_sampler().find_model_pid())

    def test_missing_proc_root_returns_none(self):
        sampler = self.make_sampler(
            proc_root=str(self.tree.root / "missing-proc")
        )
        self.assertIsNone(sampler.find_model_pid())

    def test_sample_discovers_model_pid_when_provider_absent(self):
        self.tree.add_process(
            555,
            rss_kb=4096,
            cmdline=["/usr/local/bin/llama-server", "-m", "qwen.gguf"],
        )
        sampler = self.make_sampler()

        snapshot = sampler.sample()

        self.assertEqual(snapshot["model_rss_bytes"], 4096 * 1024)
        self.assertIsNone(snapshot["detail"]["model_cpu_percent"])
        self.assertIsNone(snapshot["worker_rss_bytes"])

    def test_explicit_model_provider_wins_over_discovery(self):
        self.tree.add_process(555, rss_kb=4096, cmdline=["llama-server"])
        self.tree.add_process(777, rss_kb=1024)
        sampler = self.make_sampler(model_pid_provider=lambda: 777)

        snapshot = sampler.sample()

        self.assertEqual(snapshot["model_rss_bytes"], 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
