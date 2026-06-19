# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for sprint modules: profiles, manifest, metrics, chunking, errors, sessions."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chunking import build_document_chunks, chunks_to_map_strings
from conflict_resolve import pick_findings_from_dual_worker
from corpus_manifest import build_corpus_manifest
from errors import ErrorCode, ProcessingError, classify_exception
from first_run import is_first_run, mark_first_run_complete
from merge_hierarchy import top_evidence_from_tree
from metrics import list_recent_runs, record_run_finish, record_run_start
from run_profiles import get_profile, load_profiles
from sessions import create_session, list_sessions


class TestRunProfiles(unittest.TestCase):
    def test_large_corpus_profile(self) -> None:
        prof = get_profile("large_corpus")
        self.assertTrue(prof.get("scout_mode"))
        self.assertEqual(prof.get("max_chunk_tokens"), 4500)

    def test_load_profiles_has_keys(self) -> None:
        profiles = load_profiles()
        self.assertIn("large_corpus", profiles)


class TestConflictResolve(unittest.TestCase):
    def test_tie_prefers_first_result(self) -> None:
        empty = json.dumps({"findings": []})
        self.assertEqual(pick_findings_from_dual_worker(empty, empty), empty)

    def test_picks_richer_result(self) -> None:
        a = json.dumps({"findings": [{"evidence_refs": [{"file": "a"}]}]})
        b = json.dumps({"findings": []})
        self.assertEqual(pick_findings_from_dual_worker(a, b), a)
        rich_b = json.dumps({
            "findings": [
                {"evidence_refs": [{"file": "b"}]},
                {"evidence_refs": [{"file": "b2"}]},
            ],
        })
        self.assertEqual(pick_findings_from_dual_worker(a, rich_b), rich_b)


class TestCorpusManifest(unittest.TestCase):
    def test_manifest_counts_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.py").write_text("print(1)\n", encoding="utf-8")
            (root / "b.md").write_text("# hi", encoding="utf-8")
            m = build_corpus_manifest([root])
            self.assertEqual(m["files_total"], 2)
            self.assertIn("python", m.get("languages", {}))


class TestMetrics(unittest.TestCase):
    def test_record_start_and_finish(self) -> None:
        import uuid

        job = f"job-test-{uuid.uuid4().hex[:8]}"
        rid = record_run_start(job, "query preview", {"map": "m"})
        self.assertGreater(rid, 0)
        record_run_finish(
            rid,
            duration_s=1.5,
            chunks_total=10,
            chunks_ok=8,
            chunks_failed=2,
            scout_skipped=3,
        )
        runs = list_recent_runs(5)
        self.assertTrue(any(r.get("job_id") == job for r in runs))


class TestChunking(unittest.TestCase):
    def test_text_file_produces_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "note.txt"
            p.write_text("Hello world.\n\nSecond paragraph here.\n", encoding="utf-8")
            chunks = build_document_chunks(p, chunk_size_tokens=50, overlap_tokens=10)
            self.assertGreaterEqual(len(chunks), 1)
            strings = chunks_to_map_strings(chunks)
            self.assertIn("[FILE_PATH:", strings[0])


class TestErrors(unittest.TestCase):
    def test_classify_connect(self) -> None:
        import httpx

        err = classify_exception(httpx.ConnectError("failed"))
        self.assertEqual(err.code, ErrorCode.SERVER_UNAVAILABLE)

    def test_user_hint(self) -> None:
        pe = ProcessingError(ErrorCode.CONTEXT_OVERFLOW, "too long")
        self.assertIn("chunk", pe.user_hint().lower())


class TestFirstRun(unittest.TestCase):
    def test_first_run_marker_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "first_run_done.json"
            with patch("first_run._marker_path", return_value=marker):
                self.assertTrue(is_first_run())
                mark_first_run_complete({"base_url": "http://127.0.0.1:1234/v1"})
                self.assertFalse(is_first_run())


class TestSessions(unittest.TestCase):
    def test_create_session(self) -> None:
        path = create_session("audit-alpha")
        self.assertTrue((path / "session.json").is_file())
        names = [s.get("name") for s in list_sessions()]
        self.assertTrue(any("audit" in str(n) for n in names))


class TestTopEvidence(unittest.TestCase):
    def test_extracts_from_tree(self) -> None:
        tree = {
            "corpus": {
                "findings": [{
                    "evidence_refs": [
                        {"file": "f.txt", "chunk": "1", "quote": "x"},
                    ],
                }],
            },
        }
        ev = top_evidence_from_tree(tree, limit=5)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["file"], "f.txt")


if __name__ == "__main__":
    unittest.main()
