# SPDX-License-Identifier: AGPL-3.0-or-later
"""MAP resume, job_state cache, and worker job_id signaling."""
from __future__ import annotations

import os
import queue
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import cache
from gui import MSG_JOB_ID, _run_folder_batch
from processor import compute_job_id  # used in archive root test
from run_config import RunConfig


class TestJobPointer(unittest.TestCase):
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
        ptr = cache._last_job_pointer_path()
        if ptr.is_file():
            ptr.unlink()

    def test_last_job_pointer_roundtrip(self) -> None:
        cache.save_job_state(
            "job_ptr",
            chunks_total=5,
            query_preview="audit security",
            source_path="/tmp/corpus",
            status="running",
        )
        loaded = cache.load_last_job_pointer()
        assert loaded is not None
        self.assertEqual(loaded["job_id"], "job_ptr")
        self.assertEqual(loaded["query_preview"], "audit security")
        self.assertEqual(loaded["source_path"], "/tmp/corpus")

    def test_list_resumable_excludes_complete(self) -> None:
        cache.save_job_state(
            "done_job",
            chunks_total=2,
            query_preview="q",
            source_path="/x",
            status="running",
        )
        cache.set_cached_response("done_job", 0, "{}")
        cache.set_cached_response("done_job", 1, "{}")
        cache.mark_job_complete("done_job")
        jobs = cache.list_resumable_jobs(10)
        self.assertFalse(any(j["job_id"] == "done_job" for j in jobs))

    def test_paused_job_listed_as_resumable(self) -> None:
        cache.save_job_state(
            "paused_job",
            chunks_total=4,
            query_preview="resume me",
            source_path="/data",
            status="running",
        )
        cache.set_cached_response("paused_job", 0, "{}")
        cache.mark_job_paused("paused_job")
        jobs = cache.list_resumable_jobs(5)
        self.assertTrue(any(j["job_id"] == "paused_job" for j in jobs))


class TestFolderBatchJobId(unittest.TestCase):
    @patch("gui.run_map_reduce", new_callable=AsyncMock)
    def test_emits_job_id_message(self, map_reduce: AsyncMock) -> None:
        map_reduce.return_value = "ok"

        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "proj"
            root.mkdir()
            (root / "a.py").write_text("print(1)\n", encoding="utf-8")
            q: queue.Queue = queue.Queue()

            rc = RunConfig.from_gui(
                base_url="http://127.0.0.1:1234",
                api_key="k",
                chat_model="m",
                vision_model=None,
                composer_model=None,
                scout_model=None,
                embedding_model="",
                api_mode="native",
                low_vram=True,
                workers=2,
                context_budget=8096,
                response_reserve=2048,
                max_chunk_tokens=6000,
                max_reduce_input_tokens=24000,
                scout_mode=False,
                scout_threshold=0.35,
            )
            _run_folder_batch(
                folder_path=root,
                query="find issues",
                rc=rc,
                dynamic_chunk_size=2000,
                out_queue=q,
                put_progress=lambda *_a, **_k: None,
                stop_flag=lambda: False,
                job_id_root=root,
                source_path=str(root),
            )

            job_msgs = []
            while not q.empty():
                msg = q.get_nowait()
                if msg.get("type") == MSG_JOB_ID:
                    job_msgs.append(msg["job_id"])

        self.assertEqual(len(job_msgs), 1)
        map_reduce.assert_called_once()
        self.assertEqual(map_reduce.call_args.kwargs["job_id"], job_msgs[0])
        self.assertEqual(map_reduce.call_args.kwargs["source_path"], str(root))


class TestArchiveJobIdRoot(unittest.TestCase):
    def test_archive_job_id_uses_original_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            archive = Path(td) / "repo.zip"
            archive.write_bytes(b"not a real zip for fingerprint")
            inner = Path(td) / "extracted"
            inner.mkdir()
            (inner / "f.py").write_text("x=1", encoding="utf-8")
            files = [inner / "f.py"]
            id_archive = compute_job_id(archive, "q", file_paths=files)
            id_inner = compute_job_id(inner, "q", file_paths=files)
            self.assertNotEqual(id_archive, id_inner)
