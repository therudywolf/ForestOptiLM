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


if __name__ == "__main__":
    unittest.main()
