# SPDX-License-Identifier: AGPL-3.0-or-later
"""Тесты scout-pass, адаптивного margin и fingerprint корпуса."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from cache import build_job_id, corpus_fingerprint_from_paths
from processor import (
    _context_safety_margin,
    _server_safe_context_limit,
    filter_indices_by_scout_scores,
    scout_relevance_score,
)


class TestContextMargin(unittest.TestCase):
    def test_adaptive_margin_small_context(self) -> None:
        old = os.environ.pop("NOCTURNE_CONTEXT_SAFETY_MARGIN", None)
        try:
            margin = _context_safety_margin(8192)
            self.assertGreaterEqual(margin, 512)
            self.assertLessEqual(margin, 8192)
            safe = _server_safe_context_limit(8192)
            self.assertGreater(safe, 4096)
        finally:
            if old is not None:
                os.environ["NOCTURNE_CONTEXT_SAFETY_MARGIN"] = old

    def test_explicit_margin_env(self) -> None:
        os.environ["NOCTURNE_CONTEXT_SAFETY_MARGIN"] = "1024"
        try:
            self.assertEqual(_context_safety_margin(32000), 1024)
        finally:
            os.environ.pop("NOCTURNE_CONTEXT_SAFETY_MARGIN", None)


class TestScoutFilter(unittest.TestCase):
    def test_filters_low_scores(self) -> None:
        indices = [0, 1, 2, 3]
        scores = {0: 0.9, 1: 0.1, 2: 0.35, 3: 0.34}
        deep, skipped = filter_indices_by_scout_scores(indices, scores, 0.35)
        self.assertEqual(deep, [0, 2])
        self.assertEqual(skipped, [1, 3])

    def test_missing_score_runs_deep_map(self) -> None:
        deep, skipped = filter_indices_by_scout_scores([5], {}, 0.5)
        self.assertEqual(deep, [5])
        self.assertEqual(skipped, [])

    def test_scout_score_parsing(self) -> None:
        self.assertLess(scout_relevance_score({"no_relevant_data": True}), 0.1)
        self.assertGreaterEqual(
            scout_relevance_score({"relevant": True, "relevance_score": 0.2}),
            0.45,
        )


class TestCorpusFingerprint(unittest.TestCase):
    def test_fingerprint_changes_when_file_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.txt"
            a.write_text("one", encoding="utf-8")
            fp1 = corpus_fingerprint_from_paths([a])
            a.write_text("two", encoding="utf-8")
            fp2 = corpus_fingerprint_from_paths([a])
            self.assertNotEqual(fp1, fp2)

    def test_job_id_includes_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "proj"
            folder.mkdir()
            f = folder / "x.txt"
            f.write_text("x", encoding="utf-8")
            fp = corpus_fingerprint_from_paths([f])
            j1 = build_job_id(folder, "q", corpus_fingerprint=fp)
            j2 = build_job_id(folder, "q")
            self.assertNotEqual(j1, j2)


if __name__ == "__main__":
    unittest.main()
