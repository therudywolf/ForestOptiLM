# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from corpus_planner import file_relevance_heuristic, format_plan_ru, plan_corpus


class TestCorpusPlanner(unittest.TestCase):
    def test_file_heuristic_matches_query(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "auth_login.py"
            p.write_text("def login(): pass\n", encoding="utf-8")
            hi = file_relevance_heuristic(p, "find login security issues")
            lo = file_relevance_heuristic(p, "database migration schema")
            self.assertGreater(hi, lo)

    def test_plan_zip(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "src.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("a.py", "x" * 5000)
                zf.writestr("b.py", "y" * 5000)
            plan = plan_corpus(z, "audit python", 1000, scout_mode=True)
            self.assertTrue(plan.large_corpus)
            self.assertGreater(plan.files_total, 0)
            text = format_plan_ru(plan)
            self.assertIn("Файлов", text)


if __name__ == "__main__":
    unittest.main()
