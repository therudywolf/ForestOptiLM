# SPDX-License-Identifier: AGPL-3.0-or-later
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from models import DocumentChunk, RetrievalHit
from parser import parse_file
from pipeline import _is_hidden_or_system, _iter_files
from retrieval import LocalFaissStore


class TestParser(unittest.TestCase):
    def test_parse_txt_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sample.txt"
            p.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
            kind, payload, _ = parse_file(p, dynamic_chunk_size=50, root_dir=Path(td))
            self.assertEqual(kind, "text")
            self.assertIsInstance(payload, list)
            self.assertGreaterEqual(len(payload), 1)
            joined = "\n".join(payload)
            self.assertIn("alpha", joined)


class TestPipelineHelpers(unittest.TestCase):
    def test_iter_files_skips_hidden(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "visible.txt").write_text("ok", encoding="utf-8")
            hidden = root / ".secret"
            hidden.mkdir()
            (hidden / "x.txt").write_text("no", encoding="utf-8")
            files = _iter_files([root])
            names = [f.name for f in files]
            self.assertIn("visible.txt", names)
            self.assertFalse(any(".secret" in str(f) for f in files))

    def test_is_hidden_or_system(self) -> None:
        self.assertTrue(_is_hidden_or_system(Path(".git")))
        self.assertTrue(_is_hidden_or_system(Path("node_modules")))
        self.assertFalse(_is_hidden_or_system(Path("src")))


class TestFaissStore(unittest.TestCase):
    def test_build_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = LocalFaissStore(Path(td))
            chunks = [
                DocumentChunk(
                    chunk_id="c0",
                    source_path="a.txt",
                    text="hello world",
                    tokens=2,
                    metadata={},
                ),
                DocumentChunk(
                    chunk_id="c1",
                    source_path="b.txt",
                    text="other topic",
                    tokens=2,
                    metadata={},
                ),
            ]
            vectors = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
            stats = store.build(chunks, vectors, "embed-test")
            self.assertEqual(stats.chunks_total, 2)
            hits = store.search(vectors[0], top_k=1)
            self.assertEqual(len(hits), 1)
            self.assertIsInstance(hits[0], RetrievalHit)
            self.assertIn("hello", hits[0].text)


if __name__ == "__main__":
    unittest.main()
