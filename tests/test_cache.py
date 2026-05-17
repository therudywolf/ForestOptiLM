# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import cache


class TestMapCache(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmpdir.cleanup)
        os.environ["NOCTURNE_CACHE_DIR"] = self._tmpdir.name
        cache.CACHE_DIR = Path(self._tmpdir.name)
        cache.DB_PATH = cache.CACHE_DIR / "cache.db"
        cache.reset_cache_connection()

    def tearDown(self) -> None:
        cache.reset_cache_connection()
        os.environ.pop("NOCTURNE_CACHE_DIR", None)

    def test_set_get_roundtrip(self) -> None:
        cache.set_cached_response("job1", 0, '{"ok": true}')
        got = cache.get_cached_response("job1", 0)
        self.assertEqual(got, '{"ok": true}')

    def test_miss_returns_none(self) -> None:
        self.assertIsNone(cache.get_cached_response("missing", 99))

    def test_job_id_stable(self) -> None:
        p = Path(self._tmpdir.name) / "doc.txt"
        p.write_text("x", encoding="utf-8")
        a = cache.build_job_id(p, "query")
        b = cache.build_job_id(p, "query")
        self.assertEqual(a, b)

    def test_job_state_and_resume_list(self) -> None:
        cache.save_job_state(
            "job_resume",
            chunks_total=10,
            query_preview="find bugs",
            source_path="/data/corpus",
            status="running",
        )
        cache.set_cached_response("job_resume", 0, "{}")
        cache.set_cached_response("job_resume", 1, "{}")
        jobs = cache.list_resumable_jobs(5)
        self.assertTrue(any(j["job_id"] == "job_resume" for j in jobs))
        self.assertEqual(cache.count_cached_chunks("job_resume"), 2)
        st = cache.get_job_state("job_resume")
        assert st is not None
        self.assertEqual(st["chunks_total"], 10)
        cache.mark_job_paused("job_resume")
        st2 = cache.get_job_state("job_resume")
        assert st2 is not None
        self.assertEqual(st2["status"], "paused")
        jobs2 = cache.list_resumable_jobs(5)
        self.assertTrue(any(j["job_id"] == "job_resume" for j in jobs2))
        cache.mark_job_complete("job_resume")
        st3 = cache.get_job_state("job_resume")
        assert st3 is not None
        self.assertEqual(st3["status"], "complete")


if __name__ == "__main__":
    unittest.main()
