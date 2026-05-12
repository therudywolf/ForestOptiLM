# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
"""Регрессионные проверки качества MAP/REDUCE и валидации отчёта."""
from __future__ import annotations

import unittest

from processor import (
    _map_metrics_from_results,
    _sanitize_map_json,
    _validate_final_report,
)


class TestSanitizeMapJson(unittest.TestCase):
    def test_downgrades_critical_without_evidence(self) -> None:
        obj = {
            "findings": [
                {
                    "severity": "critical",
                    "explanation": "bug",
                    "evidence_refs": [],
                }
            ]
        }
        out = _sanitize_map_json(obj)
        self.assertEqual(out["findings"][0]["severity"], "medium")
        self.assertIn("downgraded", out["findings"][0]["explanation"])

    def test_keeps_critical_with_evidence(self) -> None:
        obj = {
            "findings": [
                {
                    "severity": "critical",
                    "explanation": "bug",
                    "evidence_refs": [
                        {"file": "a.py", "chunk": "1", "quote": "x = 1"},
                    ],
                }
            ]
        }
        out = _sanitize_map_json(obj)
        self.assertEqual(out["findings"][0]["severity"], "critical")


class TestMapMetrics(unittest.TestCase):
    def test_counts_relevant_and_evidence(self) -> None:
        j = (
            '{"no_relevant_data": false, "findings": ['
            '{"severity": "low", "evidence_refs": ['
            '{"file": "f.txt", "chunk": "1", "quote": "abc"}]}]}'
        )
        m = _map_metrics_from_results([j])
        self.assertEqual(m["relevant_chunks"], 1)
        self.assertEqual(m["findings_count"], 1)
        self.assertEqual(m["evidence_refs_count"], 1)


class TestValidateFinalReport(unittest.TestCase):
    def test_warns_short_report(self) -> None:
        text, w = _validate_final_report("hi", {"evidence_refs_count": 0})
        self.assertTrue(any("short" in x for x in w))

    def test_ok_full_report(self) -> None:
        pad = "Lorem ipsum dolor sit amet. " * 20  # длина > 400 символов
        body = (
            "## Executive Summary\n\n" + pad + "\n\n"
            "## Comprehensive Findings\n\n" + pad + "\n\n"
            "## Evidence Matrix\n\nfile `a` chunk 1 quote `z`\n\n" + pad + "\n\n"
            "## Action Plan\n\n" + pad + "\n"
        )
        text, w = _validate_final_report(body, {"evidence_refs_count": 5})
        self.assertEqual(len(w), 0)
        self.assertNotIn("Валидация", text)


if __name__ == "__main__":
    unittest.main()
