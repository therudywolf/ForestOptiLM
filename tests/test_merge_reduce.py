# SPDX-License-Identifier: AGPL-3.0-or-later
"""MAP merge / reduce helpers using fixtures (no live LLM)."""
from __future__ import annotations

import json
import unittest
from pathlib import Path

from merge_hierarchy import hierarchical_merge_map_results
from processor import (
    _fallback_merge_map_json,
    _merge_map_json_deterministic,
    _sanitize_map_json,
    _validate_final_report,
)


class TestMergeReduceFixtures(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixture = Path(__file__).parent / "fixtures" / "map_sample.json"
        cls.sample = fixture.read_text(encoding="utf-8")

    def test_deterministic_merge_single(self) -> None:
        out = _merge_map_json_deterministic([self.sample])
        parsed = json.loads(out)
        self.assertFalse(parsed.get("no_relevant_data"))
        self.assertGreaterEqual(len(parsed.get("findings") or []), 1)

    def test_fallback_merge_preserves_findings(self) -> None:
        merged = _fallback_merge_map_json([self.sample])
        parsed = json.loads(merged)
        self.assertGreaterEqual(len(parsed.get("findings") or []), 1)

    def test_hierarchical_with_fixture(self) -> None:
        corpus_json, tree = hierarchical_merge_map_results([self.sample])
        corpus = json.loads(corpus_json)
        self.assertIn("findings", corpus)
        self.assertIn("files", tree)

    def test_sanitize_downgrade_without_quote(self) -> None:
        obj = {
            "findings": [{
                "severity": "critical",
                "explanation": "x",
                "evidence_refs": [],
            }],
        }
        out = _sanitize_map_json(obj)
        self.assertEqual(out["findings"][0]["severity"], "medium")

    def test_validate_with_metrics(self) -> None:
        body = (
            "## Executive Summary\n\n" + "word " * 80 + "\n\n"
            "## Comprehensive Findings\n\n" + "word " * 80 + "\n\n"
            "## Evidence Matrix\n\n| f | 1 | quote text here |\n\n" + "word " * 80 + "\n\n"
            "## Action Plan\n\n" + "word " * 80 + "\n"
        )
        text, warnings = _validate_final_report(body, {"evidence_refs_count": 3, "findings_count": 5})
        self.assertEqual(len(warnings), 0, warnings)


if __name__ == "__main__":
    unittest.main()
