"""supervisor.py 归属校验与环境构造测试（仅 POSIX /proc 平台执行）。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent
SRC = BENCH.parent / "Src"
for path in (str(BENCH), str(SRC)):
    if path not in sys.path:
        sys.path.insert(0, path)

import supervisor as supervisor_mod  # noqa: E402

POSIX = sys.platform.startswith("linux")


@unittest.skipUnless(POSIX, "需要 Linux /proc")
class ProcHelpersTests(unittest.TestCase):
    def test_read_start_ticks_matches_proc(self) -> None:
        ticks = supervisor_mod._read_start_ticks(os.getpid())
        self.assertIsInstance(ticks, int)
        stat = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8", errors="replace")
        expected = int(stat.rsplit(")", 1)[1].split()[19])
        self.assertEqual(ticks, expected)

    def test_read_start_ticks_missing_pid(self) -> None:
        self.assertIsNone(supervisor_mod._read_start_ticks(99999999))

    def test_read_cmdline_contains_python(self) -> None:
        argv = supervisor_mod._read_cmdline(os.getpid())
        self.assertTrue(argv)
        self.assertIn("python", argv.lower())


class EnvTests(unittest.TestCase):
    def test_default_env_directs_caches_into_bench(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            project = Path(tmp) / "proj"
            bench = project / "Testbench"
            bench.mkdir(parents=True)
            env = supervisor_mod.default_env(project, bench)
            self.assertEqual(env["PYTHONPATH"], str(project / "Src"))
            self.assertEqual(env["PYTHONDONTWRITEBYTECODE"], "1")
            self.assertEqual(env["PYTHONUNBUFFERED"], "1")
            for key in ("XDG_CACHE_HOME", "TMPDIR", "CUDA_CACHE_PATH"):
                self.assertTrue(env[key].startswith(str(bench / ".runtime")))

    def test_worker_python_relative_path(self) -> None:
        resolved = supervisor_mod.worker_python("../.venv/bin/python")
        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(resolved.replace("\\", "/").endswith(".venv/bin/python"))


if __name__ == "__main__":
    unittest.main()
