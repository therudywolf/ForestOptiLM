# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
"""Smoke/regression: mega-file chunking, FILE_PATH, vision extractors."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from parser import chunk_text_for_map_file


class TestMegaFileChunking(unittest.TestCase):
    def test_small_file_single_part(self) -> None:
        p = Path("dummy.txt")
        text = "Paragraph one.\n\nParagraph two.\n\n" * 5
        chunks = chunk_text_for_map_file(p, text, chunk_size_tokens=200, overlap_tokens=20)
        self.assertTrue(len(chunks) >= 1)
        for c in chunks:
            self.assertIn("[FILE_PATH:", c)
            self.assertIn("[FILE_PART: 1/1]", c)
            self.assertIn("[CHUNK_INDEX:", c)
            self.assertIn("[Файл:", c)

    def test_mega_threshold_splits_parts(self) -> None:
        p = Path("big.txt")
        # Длинный текст, чтобы превысить порог при низком mega_th
        text = ("word " * 500 + "\n\n") * 200
        with patch.dict(os.environ, {"NOCTURNE_MEGA_FILE_TOKEN_THRESHOLD": "200"}):
            chunks = chunk_text_for_map_file(p, text, chunk_size_tokens=80, overlap_tokens=10)
        parts = {c.split("[FILE_PART:")[1].split("]")[0].strip() for c in chunks if "[FILE_PART:" in c}
        self.assertTrue(len(parts) >= 1)


class TestSpecialTokenSafety(unittest.TestCase):
    """Реальный баг (Telegram-экспорт): сообщение с литералом `<|endoftext|>`
    роняло индексацию — tiktoken по умолчанию запрещает спецтокены. Теперь
    трактуем их как обычный текст (disallowed_special=())."""

    SPECIAL = "обычное сообщение\n<|endoftext|>\nещё <|im_start|> и <|fim_prefix|> внутри"

    def test_count_tokens_does_not_crash(self) -> None:
        from parser import count_tokens
        self.assertGreater(count_tokens(self.SPECIAL), 0)

    def test_semantic_chunking_does_not_crash(self) -> None:
        from parser import chunk_text_semantic
        self.assertTrue(chunk_text_semantic(self.SPECIAL * 40, 100, 20))

    def test_map_file_chunking_does_not_crash(self) -> None:
        chunks = chunk_text_for_map_file(Path("chat.html"), self.SPECIAL, 100, 20)
        self.assertTrue(chunks)
        self.assertTrue(any("endoftext" in c for c in chunks))  # текст сохранён как есть


class TestHeadingBoundaryChunking(unittest.TestCase):
    def test_heading_starts_new_segment(self) -> None:
        # По мотивам qmd: markdown-заголовок начинает новый сегмент, даже без
        # пустой строки до него → секции разных тем не сливаются в один чанк.
        from parser import _segment_paragraphs_and_sentences
        text = "# Beta\nсодержимое Beta\n## Alpha\nсодержимое Alpha"
        segs = _segment_paragraphs_and_sentences(text, 8000)
        heads = [s for s in segs if s.lstrip().startswith("#")]
        self.assertEqual(len(heads), 2)
        self.assertTrue(any("Alpha" in s and s.lstrip().startswith("##") for s in segs))

    def test_no_headings_unchanged(self) -> None:
        from parser import _segment_paragraphs_and_sentences
        segs = _segment_paragraphs_and_sentences("Абзац один.\n\nАбзац два.", 8000)
        self.assertEqual(segs, ["Абзац один.", "Абзац два."])


class TestImageExtensions(unittest.TestCase):
    def test_image_kind_in_registry(self) -> None:
        from file_extractors import IMAGE_EXTENSIONS

        self.assertIn(".png", IMAGE_EXTENSIONS)
        self.assertIn(".jpg", IMAGE_EXTENSIONS)


if __name__ == "__main__":
    unittest.main()
