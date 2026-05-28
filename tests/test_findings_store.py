# SPDX-License-Identifier: AGPL-3.0-or-later
"""Persistent findings memory & cross-scan diff (Pillar 5)."""
from __future__ import annotations

import os
import tempfile
import unittest

import findings_store as fs


def _f(severity: str, cve: str = "", asset: str = "h.json", type_: str = "vuln") -> dict:
    return {
        "_facets": {
            "severity": severity,
            "type": type_,
            "explanation": f"{type_} {cve}".strip(),
            "cve": [cve] if cve else [],
            "cwe": [],
            "asset": asset,
            "component": "",
            "has_evidence": True,
        }
    }


class TestFindingsStore(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        os.environ["NOCTURNE_CACHE_DIR"] = self._tmp.name
        fs.reset_connection()

    def tearDown(self) -> None:
        fs.reset_connection()
        os.environ.pop("NOCTURNE_CACHE_DIR", None)

    def test_canonical_key_cve_asset(self) -> None:
        k1 = fs.canonical_finding_key({"cve": ["CVE-2024-1"], "asset": "h1"})
        k2 = fs.canonical_finding_key({"cve": ["CVE-2024-1"], "asset": "h1", "type": "x"})
        k3 = fs.canonical_finding_key({"cve": ["CVE-2024-1"], "asset": "h2"})
        self.assertEqual(k1, k2)        # type ignored when CVE+asset present
        self.assertNotEqual(k1, k3)     # different asset → different finding

    def test_first_scan_has_no_diff(self) -> None:
        block = fs.record_and_diff(
            job_id="j1", source_path="/corpus", query="q",
            findings=[_f("critical", "CVE-2024-1"), _f("low")],
        )
        self.assertEqual(block, "")
        self.assertEqual(len(fs.list_scans("/corpus")), 1)

    def test_diff_new_and_fixed(self) -> None:
        s1 = fs.record_scan(
            job_id="j1", source_path="/c", query="q",
            findings=[_f("critical", "CVE-2024-1"), _f("high", "CVE-2024-2")],
        )
        s2 = fs.record_scan(
            job_id="j2", source_path="/c", query="q",
            findings=[_f("high", "CVE-2024-2"), _f("critical", "CVE-2024-3")],
        )
        assert s1 and s2
        d = fs.diff_scans(s1, s2)
        new_cves = {x.get("cve") for x in d["new"]}
        fixed_cves = {x.get("cve") for x in d["fixed"]}
        persistent_cves = {x.get("cve") for x in d["persistent"]}
        self.assertEqual(new_cves, {"CVE-2024-3"})
        self.assertEqual(fixed_cves, {"CVE-2024-1"})
        self.assertEqual(persistent_cves, {"CVE-2024-2"})

    def test_record_and_diff_second_run_block(self) -> None:
        fs.record_and_diff(
            job_id="j1", source_path="/c", query="q",
            findings=[_f("critical", "CVE-2024-1")],
        )
        block = fs.record_and_diff(
            job_id="j2", source_path="/c", query="q",
            findings=[_f("critical", "CVE-2024-9")],
            language="ru",
        )
        self.assertIn("Изменения с прошлого скана", block)
        self.assertIn("CVE-2024-9", block)

    def test_disabled_returns_empty(self) -> None:
        os.environ["NOCTURNE_FINDINGS_MEMORY"] = "0"
        try:
            block = fs.record_and_diff(
                job_id="j", source_path="/c", query="q", findings=[_f("high", "CVE-1")],
            )
            self.assertEqual(block, "")
        finally:
            os.environ.pop("NOCTURNE_FINDINGS_MEMORY", None)


if __name__ == "__main__":
    unittest.main()
