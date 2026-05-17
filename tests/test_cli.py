# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path


class TestCli(unittest.TestCase):
    def test_cli_help_exits_zero(self) -> None:
        root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "-m", "forestoptilm.cli", "analyze", "--help"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("query", proc.stdout.lower())


if __name__ == "__main__":
    unittest.main()
