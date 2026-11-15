"""Read-only resource metrics sampling for the AI Eyes test bench.

Dependency-free (Python 3.12 standard library only): the module can be imported
standalone from ``Testbench/`` or through a package import.  Nothing is
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

__all__ = ["MetricsSampler"]

_DEFAULT_CLOCK_TICKS = 100
_READ_LIMIT = 8192

try:
    _CLOCK_TICKS = int(os.sysconf("SC_CLK_TCK"))
except (AttributeError, ValueError, OSError):  # Windows / unusual platforms
    _CLOCK_TICKS = _DEFAULT_CLOCK_TICKS
if _CLOCK_TICKS <= 0:
    _CLOCK_TICKS = _DEFAULT_CLOCK_TICKS

_ROLES = ("worker", "model", "bench")

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
        },
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
    """

    def __init__(
        self,
        *,
        proc_root="/proc",
        sys_root="/sys",
        worker_pid_provider=None,
        model_pid_provider=None,
        bench_pid=None,
    ):
        self.proc_root = os.fspath(proc_root)
        self.sys_root = os.fspath(sys_root)
        self.worker_pid_provider = worker_pid_provider
        self.model_pid_provider = model_pid_provider
        self.bench_pid = bench_pid
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
        text = _read_text(os.path.join(self.proc_root, str(pid), "status"))
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
            if kib < 0:
                return None
            return kib * 1024
        return None

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
