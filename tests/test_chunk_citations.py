# SPDX-License-Identifier: AGPL-3.0-or-later
"""Page/line metadata on chunks → citations."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from chunking import build_document_chunks
from notebook_chat import select_contexts


class _Hit:
    def __init__(self, text, source_path, metadata) -> None:
        self.text = text
        self.source_path = source_path
        self.metadata = metadata
        self.chunk_id = "c"
        self.score = 0.9


class TestChunkCitations(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.tmp = Path(self._tmp.name)

    def test_line_and_page_metadata(self) -> None:
        page1 = "Первый абзац на странице один, обычное содержимое отчёта.\n\n" * 4
        page2 = ("Совсем другой текст на странице два с уникальным маркером ZEBRA, "
                 "которого больше нигде нет.\n\n") * 4
        content = page1 + "\f" + page2  # \f = page break (as PDF extraction emits)
        p = self.tmp / "doc.txt"
        p.write_text(content, encoding="utf-8")

        chunks = build_document_chunks(p, chunk_size_tokens=60)
        self.assertTrue(chunks)
        # every chunk got a line_start
        self.assertTrue(all(c.metadata.get("line_start") for c in chunks))
        # the chunk with the page-2 marker is tagged page 2; no \f leaks into text
        zebra = [c for c in chunks if "ZEBRA" in c.text]
        self.assertTrue(zebra)
        self.assertEqual(zebra[0].metadata.get("page"), 2)
        self.assertNotIn("\f", zebra[0].text)
        # page-1 content stays page 1
        first = [c for c in chunks if "странице один" in c.text]
        self.assertTrue(first)
        self.assertEqual(first[0].metadata.get("page"), 1)

    def test_citation_exposes_locator(self) -> None:
        hits = [_Hit("важный фрагмент про релиз", "C:/x/report.pdf",
                     {"page": 7, "line_start": 142})]
        ctx = select_contexts(hits, max_tokens=1000)
        self.assertEqual(ctx[0].page, 7)
        self.assertEqual(ctx[0].locator(), "стр. 7")
        cit = ctx[0].to_citation()
        self.assertEqual(cit["page"], 7)
        self.assertEqual(cit["locator"], "стр. 7")

    def test_locator_falls_back_to_line(self) -> None:
        hits = [_Hit("text", "C:/x/a.txt", {"line_start": 30})]
        ctx = select_contexts(hits, max_tokens=1000)
        self.assertEqual(ctx[0].locator(), "строка 30")


if __name__ == "__main__":
    unittest.main()
