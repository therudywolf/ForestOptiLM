# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import os
import tempfile
import unittest
import zipfile
from pathlib import Path

from large_corpus_io import (
    corpus_input_root,
    is_archive,
    should_stream_plain_file,
    chunk_plain_file_streaming,
    streaming_threshold_bytes,
)
from pipeline import _iter_files


class TestLargeCorpusIo(unittest.TestCase):
    def test_should_stream_by_size(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "big.log"
            p.write_text("line\n" * 10, encoding="utf-8")
            os.environ["NOCTURNE_STREAMING_FILE_BYTES"] = "5"
            self.assertTrue(should_stream_plain_file(p))
            os.environ["NOCTURNE_STREAMING_FILE_BYTES"] = str(10**12)
            self.assertFalse(should_stream_plain_file(p))

    def test_streaming_chunks_lines(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "data.txt"
            p.write_text("".join(f"row {i}\n" for i in range(500)), encoding="utf-8")
            os.environ["NOCTURNE_STREAMING_FILE_BYTES"] = "1"
            chunks = chunk_plain_file_streaming(p, chunk_size_tokens=80, overlap_tokens=0)
            self.assertGreater(len(chunks), 3)
            self.assertIn("FILE_PATH", chunks[0])
            self.assertIn("row 0", chunks[0])

    def test_archive_expands_as_folder(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            z = root / "code.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("pkg/a.py", "print('hi')\n")
                zf.writestr("pkg/b.py", "x = 1\n")
            self.assertTrue(is_archive(z))
            with corpus_input_root(z) as corpus:
                self.assertTrue(corpus.is_dir())
                files = _iter_files([corpus])
                rels = sorted(str(f.relative_to(corpus)).replace("\\", "/") for f in files)
                self.assertEqual(rels, ["pkg/a.py", "pkg/b.py"])

    def test_streaming_threshold_default_sane(self) -> None:
        os.environ.pop("NOCTURNE_STREAMING_FILE_BYTES", None)
        self.assertGreater(streaming_threshold_bytes(), 1_000_000)

    def test_auto_scout_archive(self) -> None:
        from large_corpus_io import auto_scout_file_bytes, is_large_corpus_input

        with tempfile.TemporaryDirectory() as td:
            z = Path(td) / "a.zip"
            with zipfile.ZipFile(z, "w") as zf:
                zf.writestr("x.txt", "hi")
            self.assertTrue(is_large_corpus_input(z)[0])
            self.assertGreater(auto_scout_file_bytes(), 0)


if __name__ == "__main__":
    unittest.main()
