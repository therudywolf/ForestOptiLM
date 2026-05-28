# SPDX-License-Identifier: AGPL-3.0-or-later
"""Format-agnostic record extraction & chunking (Trivy/ZAP-shaped, no hardcoded schema)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from xml.etree import ElementTree as ET

from record_chunking import (
    build_record_chunks,
    extract_json_records,
    extract_records_from_file,
    extract_xml_records,
    records_to_chunks,
)

# Trivy-shaped: root scalars + Results[] (poor container) → Vulnerabilities[] (records)
TRIVY = {
    "SchemaVersion": 2,
    "ArtifactName": "myimage:1.0",
    "ArtifactType": "container_image",
    "Results": [
        {
            "Target": "app/package-lock.json",
            "Class": "lang-pkgs",
            "Type": "npm",
            "Vulnerabilities": [
                {
                    "VulnerabilityID": "CVE-2024-1111",
                    "PkgName": "lodash",
                    "InstalledVersion": "4.17.20",
                    "FixedVersion": "4.17.21",
                    "Severity": "HIGH",
                    "Title": "Prototype pollution",
                    "CVSS": {"nvd": {"V3Score": 7.5}},
                    "References": ["https://example/1", "https://example/2"],
                },
                {
                    "VulnerabilityID": "CVE-2024-2222",
                    "PkgName": "minimist",
                    "InstalledVersion": "1.2.0",
                    "FixedVersion": "1.2.6",
                    "Severity": "CRITICAL",
                    "Title": "Argument injection",
                },
            ],
        },
        {
            "Target": "usr/bin",
            "Class": "os-pkgs",
            "Type": "debian",
            "Vulnerabilities": [
                {
                    "VulnerabilityID": "CVE-2024-3333",
                    "PkgName": "openssl",
                    "Severity": "MEDIUM",
                    "Title": "Info leak",
                },
            ],
        },
    ],
}

# ZAP-JSON-shaped: site[] (poor container) → alerts[] (rich container → record),
# alert.instances[] summarized.
ZAP = {
    "@version": "2.14.0",
    "site": [
        {
            "@name": "https://app.example",
            "@host": "app.example",
            "@port": "443",
            "@ssl": "true",
            "alerts": [
                {
                    "pluginid": "40012",
                    "alertRef": "40012",
                    "alert": "Cross Site Scripting (Reflected)",
                    "name": "Cross Site Scripting (Reflected)",
                    "riskcode": "3",
                    "confidence": "2",
                    "riskdesc": "High (Medium)",
                    "desc": "XSS desc",
                    "solution": "Encode output",
                    "reference": "https://owasp",
                    "cweid": "79",
                    "wascid": "8",
                    "count": "5",
                    "instances": [
                        {"uri": "https://app.example/a", "method": "GET", "param": "q"},
                        {"uri": "https://app.example/b", "method": "GET", "param": "x"},
                    ],
                },
                {
                    "pluginid": "10202",
                    "alertRef": "10202",
                    "alert": "Absence of Anti-CSRF Tokens",
                    "name": "Absence of Anti-CSRF Tokens",
                    "riskcode": "1",
                    "confidence": "2",
                    "riskdesc": "Low (Medium)",
                    "desc": "csrf",
                    "solution": "Add token",
                    "reference": "https://owasp",
                    "cweid": "352",
                    "wascid": "9",
                    "count": "1",
                    "instances": [
                        {"uri": "https://app.example/form", "method": "POST"},
                    ],
                },
            ],
        }
    ],
}

ZAP_XML = """<?xml version="1.0"?>
<OWASPZAPReport version="2.14.0">
  <site name="https://app.example" host="app.example" port="443" ssl="true">
    <alerts>
      <alertitem>
        <pluginid>40012</pluginid>
        <alert>Cross Site Scripting (Reflected)</alert>
        <riskcode>3</riskcode>
        <riskdesc>High (Medium)</riskdesc>
        <cweid>79</cweid>
        <desc>xss</desc>
        <instances>
          <instance><uri>https://app.example/a</uri><method>GET</method></instance>
          <instance><uri>https://app.example/b</uri><method>GET</method></instance>
        </instances>
      </alertitem>
      <alertitem>
        <pluginid>10202</pluginid>
        <alert>Absence of Anti-CSRF Tokens</alert>
        <riskcode>1</riskcode>
        <riskdesc>Low (Medium)</riskdesc>
        <cweid>352</cweid>
        <desc>csrf</desc>
        <instances>
          <instance><uri>https://app.example/form</uri><method>POST</method></instance>
        </instances>
      </alertitem>
    </alerts>
  </site>
</OWASPZAPReport>
"""


class TestJsonRecordExtraction(unittest.TestCase):
    def test_trivy_emits_vulnerabilities_with_context(self) -> None:
        records = extract_json_records(TRIVY)
        # 3 vulnerabilities total (2 + 1), NOT the 2 Results.
        self.assertEqual(len(records), 3)
        ids = sorted(r.get("VulnerabilityID") for _, r in records)
        self.assertEqual(ids, ["CVE-2024-1111", "CVE-2024-2222", "CVE-2024-3333"])
        # Parent Result scalars carried as @-context.
        first = dict(records[0][1])
        self.assertEqual(first.get("@ArtifactName"), "myimage:1.0")
        self.assertIn(first.get("@Target"), ("app/package-lock.json", "usr/bin"))

    def test_zap_emits_alerts_not_instances(self) -> None:
        records = extract_json_records(ZAP)
        # Rich alert containers become records; instances summarized.
        self.assertEqual(len(records), 2)
        alerts = {r.get("alert") for _, r in records}
        self.assertIn("Cross Site Scripting (Reflected)", alerts)
        rec0 = dict(records[0][1])
        self.assertEqual(rec0.get("@@host") or rec0.get("@host"), "app.example")
        # instances collapsed to "[N items]"
        self.assertTrue(str(rec0.get("instances", "")).startswith("["))

    def test_flat_array(self) -> None:
        data = [{"id": 1, "sev": "high"}, {"id": 2, "sev": "low"}]
        records = extract_json_records(data)
        self.assertEqual(len(records), 2)

    def test_single_object_is_one_record(self) -> None:
        records = extract_json_records({"id": "X", "severity": "high", "title": "t"})
        self.assertEqual(len(records), 1)


class TestXmlRecordExtraction(unittest.TestCase):
    def test_zap_xml_emits_alertitems(self) -> None:
        root = ET.fromstring(ZAP_XML)
        records = extract_xml_records(root)
        self.assertEqual(len(records), 2)
        tags = {r.get("alert") for _, r in records}
        self.assertIn("Cross Site Scripting (Reflected)", tags)
        rec0 = dict(records[0][1])
        # instances (repeating child) summarized
        self.assertTrue(str(rec0.get("instances", "")).startswith("["))


class TestChunking(unittest.TestCase):
    def test_records_not_split_and_headers(self) -> None:
        records = extract_json_records(TRIVY)
        chunks = records_to_chunks(records, "scan.json", chunk_size_tokens=50)
        total = sum(c.count("- record @") for c in chunks)
        self.assertEqual(total, len(records))
        for c in chunks:
            self.assertIn("[FILE_PATH: scan.json]", c)
            self.assertIn("[CHUNK_INDEX:", c)

    def test_build_record_chunks_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trivy.json"
            p.write_text(json.dumps(TRIVY), encoding="utf-8")
            chunks = build_record_chunks(p, 4000, root_dir=Path(td))
            assert chunks is not None
            joined = "\n".join(chunks)
            self.assertIn("CVE-2024-2222", joined)
            self.assertIn("[FILE_PATH: trivy.json]", joined)

    def test_jsonl_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "findings.jsonl"
            p.write_text(
                '{"id":1,"sev":"high"}\n{"id":2,"sev":"low"}\n', encoding="utf-8",
            )
            records = extract_records_from_file(p)
            assert records is not None
            self.assertEqual(len(records), 2)

    def test_non_record_json_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "scalar.json"
            p.write_text("[1, 2, 3]", encoding="utf-8")  # list of scalars, no records
            self.assertIsNone(extract_records_from_file(p))


class TestParseFileRouting(unittest.TestCase):
    def test_parse_file_routes_json_to_records(self) -> None:
        from parser import parse_file

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "trivy.json"
            p.write_text(json.dumps(TRIVY), encoding="utf-8")
            kind, payload, df = parse_file(p, dynamic_chunk_size=4000, root_dir=Path(td))
            self.assertEqual(kind, "text")
            joined = "\n".join(payload)  # type: ignore[arg-type]
            self.assertIn("CVE-2024-1111", joined)
            self.assertIn("record @", joined)


if __name__ == "__main__":
    unittest.main()
