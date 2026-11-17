import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server

@unittest.skipUnless(os.name == 'posix', 'Linux flock')
class SingletonTests(unittest.TestCase):
    def test_second_lock_rejected_and_released_lock_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'server.lock'
            first = server.acquire_server_lock(path)
            self.assertIsNotNone(first)
            try:
                self.assertIsNone(server.acquire_server_lock(path))
            finally:
                first.close()
            second = server.acquire_server_lock(path)
            self.assertIsNotNone(second)
            second.close()

    def test_duplicate_main_never_constructs_http_or_worker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = server.acquire_server_lock(root / 'server.lock')
            try:
                with patch.object(server, 'RUNTIME_DIR', root), patch.object(server, '_serve') as serve:
                    self.assertEqual(server.main(), 2)
                    serve.assert_not_called()
            finally:
                first.close()
