# SPDX-License-Identifier: AGPL-3.0-or-later
"""Deterministic finding aggregation & categorization (no LLM)."""
from __future__ import annotations

import json
import unittest

from findings_aggregate import (
    build_category_block,
    categorize,
    extract_facets,
    iter_findings_from_map,
    normalize_severity,
)


def _map_json(file: str, findings: list[dict]) -> str:
    return json.dumps(
        {"chunk_index": 1, "file": file, "no_relevant_data": False,
         "findings": findings, "recommendations": []},
        ensure_ascii=False,
    )


class TestNormalizeSeverity(unittest.TestCase):
    def test_variants(self) -> None:
        self.assertEqual(normalize_severity("CRITICAL"), "critical")
        self.assertEqual(normalize_severity("Высокий"), "high")
        self.assertEqual(normalize_severity("3"), "high")        # ZAP riskcode
        self.assertEqual(normalize_severity(9.8), "critical")    # CVSS
        self.assertEqual(normalize_severity(7.5), "high")
        self.assertEqual(normalize_severity(""), "unknown")


class TestFacets(unittest.TestCase):
    def test_extract_cve_cwe_asset(self) -> None:
        f = {
            "severity": "high",
            "type": "vuln",
            "explanation": "lodash@4.17.20 affected by CVE-2024-1111 (CWE-1321)",
            "evidence_refs": [{"file": "app/pkg.json", "chunk": "1", "quote": "lodash 4.17.20"}],
        }
        facets = extract_facets(f)
        self.assertIn("CVE-2024-1111", facets["cve"])
        self.assertIn("CWE-1321", facets["cwe"])
        self.assertEqual(facets["asset"], "app/pkg.json")
        self.assertEqual(facets["component"], "lodash")
        self.assertTrue(facets["has_evidence"])


class TestCategorize(unittest.TestCase):
    def _findings(self) -> list[dict]:
        m1 = _map_json("hostA.json", [
            {"severity": "CRITICAL", "type": "rce", "explanation": "CVE-2024-1111 rce",
             "evidence_refs": [{"file": "hostA.json", "quote": "x"}]},
            {"severity": "high", "type": "xss", "explanation": "CVE-2024-2222 xss",
             "evidence_refs": [{"file": "hostA.json", "quote": "y"}]},
        ])
        m2 = _map_json("hostB.json", [
            {"severity": "low", "type": "info", "explanation": "banner",
             "evidence_refs": [{"file": "hostB.json", "quote": "z"}]},
            # duplicate of CVE-2024-1111 on a different asset
            {"severity": "critical", "type": "rce", "explanation": "CVE-2024-1111 rce",
             "evidence_refs": [{"file": "hostB.json", "quote": "x"}]},
        ])
        return list(iter_findings_from_map([m1, m2]))

    def test_counts_by_severity_dedup_default(self) -> None:
        findings = self._findings()
        # default dedup (severity,type,explanation): the two CVE-2024-1111 rce
        # collapse (same sev/type/expl) -> 3 unique.
        summ = categorize(findings, dedup_keys=("severity", "type", "explanation"))
        self.assertEqual(summ.unique, 3)
        self.assertEqual(summ.by_severity.get("critical"), 1)
        self.assertEqual(summ.by_severity.get("high"), 1)
        self.assertEqual(summ.by_severity.get("low"), 1)

    def test_dedup_by_cve_asset_keeps_both_assets(self) -> None:
        findings = self._findings()
        # group by asset → dedup key (asset, cve): same CVE on 2 assets stays 2.
        summ = categorize(findings, dedup_keys=("asset", "cve"), axis="asset")
        # 4 distinct (asset,cve) combos
        self.assertEqual(summ.unique, 4)
        assets = dict(summ.by_axis)
        self.assertIn("hostA.json", assets)
        self.assertIn("hostB.json", assets)

    def test_top_cves(self) -> None:
        findings = self._findings()
        summ = categorize(findings, dedup_keys=("asset", "cve"))
        top = dict(summ.top_cves)
        self.assertEqual(top.get("CVE-2024-1111"), 2)

    def test_block_markdown_has_numbers(self) -> None:
        m = _map_json("h.json", [
            {"severity": "critical", "type": "rce", "explanation": "CVE-2024-9 boom",
             "evidence_refs": [{"file": "h.json", "quote": "q"}]},
        ])
        block = build_category_block([m], dedup_keys=("severity", "type", "explanation"), language="ru")
        self.assertIn("severity", block.lower())
        self.assertIn("critical: 1", block)


class TestNoFindings(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertEqual(build_category_block([]), "")
        nrd = json.dumps({"no_relevant_data": True, "findings": []})
        self.assertEqual(build_category_block([nrd]), "")


if __name__ == "__main__":
    unittest.main()
