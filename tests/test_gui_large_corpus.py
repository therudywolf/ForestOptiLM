# SPDX-License-Identifier: AGPL-3.0-or-later
"""GUI worker paths for ZIP and huge plain files (no display required)."""
from __future__ import annotations

import asyncio
import os
import queue
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

from large_corpus_io import is_large_corpus_input, large_corpus_profile_kwargs
from gui import MSG_TRACE, _run_processing


class TestLargeCorpusDetection(unittest.TestCase):
    def test_zip_is_large_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "repo.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("a.py", "x = 1\n")
            ok, reason = is_large_corpus_input(z)
            self.assertTrue(ok)
            self.assertEqual(reason, "archive")

    def test_big_log_is_large_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "app.log"
            p.write_text("line\n" * 5000, encoding="utf-8")
            os.environ["NOCTURNE_AUTO_SCOUT_BYTES"] = "1000"
            ok, _ = is_large_corpus_input(p)
            self.assertTrue(ok)

    def test_profile_has_scout(self) -> None:
        prof = large_corpus_profile_kwargs()
        self.assertTrue(prof.get("scout_mode"))


class TestRunProcessingWorker(unittest.TestCase):
    @patch("gui._run_folder_batch")
    def test_zip_routes_to_folder_batch_with_query(self, folder_batch: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "code.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("src/main.py", "def main():\n    pass\n")
            q: queue.Queue = queue.Queue()
            user_query = "Find dead code and security issues"
            _run_processing(
                file_path=z,
                folder_path=None,
                query=user_query,
                model="test-model",
                base_url="http://127.0.0.1:1234",
                api_key="k",
                context_budget=8096,
                response_reserve=2048,
                workers=2,
                out_queue=q,
                scout_mode=True,
            )
            folder_batch.assert_called_once()
            self.assertEqual(folder_batch.call_args.kwargs["query"], user_query)
            fp = folder_batch.call_args.kwargs["folder_path"]
            self.assertIsInstance(fp, Path)

    @patch("gui.run_map_reduce", new_callable=AsyncMock)
    def test_huge_plain_file_streams_chunks_and_query(self, map_reduce: AsyncMock) -> None:
        map_reduce.return_value = "## Executive Summary\n\nok"

        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "huge.log"
            log.write_text("".join(f"ERROR line {i}\n" for i in range(8000)), encoding="utf-8")
            os.environ["NOCTURNE_STREAMING_FILE_BYTES"] = "1000"
            q: queue.Queue = queue.Queue()
            _run_processing(
                file_path=log,
                folder_path=None,
                query="Summarize errors in log",
                model="m",
                base_url="http://127.0.0.1:1234",
                api_key="k",
                context_budget=8096,
                response_reserve=2048,
                workers=2,
                out_queue=q,
                scout_mode=True,
            )
            map_reduce.assert_called_once()
            self.assertEqual(map_reduce.call_args.kwargs["user_query"], "Summarize errors in log")
            chunks = map_reduce.call_args.kwargs["chunks"]
            self.assertGreater(len(chunks), 0)
            self.assertTrue(any("ERROR line" in c for c in chunks))

    @patch("gui._run_folder_batch")
    def test_trace_contains_query_preview(self, folder_batch: MagicMock) -> None:
        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "x.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("f.txt", "data")
            q: queue.Queue = queue.Queue()
            _run_processing(
                file_path=z,
                folder_path=None,
                query="проверь утечки памяти",
                model="m",
                base_url="http://127.0.0.1:1234",
                api_key="k",
                context_budget=4096,
                response_reserve=1024,
                workers=1,
                out_queue=q,
            )
            traces = []
            while not q.empty():
                msg = q.get_nowait()
                if msg.get("type") == MSG_TRACE:
                    traces.append(msg.get("line", ""))
            self.assertTrue(any("QUERY" in t and "утечки" in t for t in traces), traces)


class TestAutoTuneHelpers(unittest.TestCase):
    def test_streaming_plain_file_via_parse_file(self) -> None:
        from parser import parse_file

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "big.txt"
            p.write_text("".join(f"row {i}\n" for i in range(3000)), encoding="utf-8")
            os.environ["NOCTURNE_STREAMING_FILE_BYTES"] = "100"
            kind, chunks, _ = parse_file(p, dynamic_chunk_size=100, root_dir=Path(td))
            self.assertEqual(kind, "text")
            self.assertGreater(len(chunks), 5)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
