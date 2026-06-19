# SPDX-License-Identifier: AGPL-3.0-or-later
"""Vision-at-index-time: describe images so their content is retrievable."""
from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chunking import build_document_chunks
from vision_index import make_image_describer

# 1x1 PNG
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


class TestImageDescriber(unittest.TestCase):
    def test_none_without_model(self) -> None:
        self.assertIsNone(make_image_describer("", "u", "k"))
        self.assertIsNone(make_image_describer("(выберите модель)", "u", "k"))
        self.assertIsNotNone(make_image_describer("qwen2.5-vl", "u", "k"))

    def test_caches_by_content(self) -> None:
        calls: list[int] = []

        def fake_describe(*_a, **_k):
            calls.append(1)
            return "описание схемы"

        with patch("vision_index.describe_image", new=fake_describe):
            d = make_image_describer("vl", "u", "k")
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "a.png"
                p.write_bytes(_PNG)
                assert d is not None
                self.assertEqual(d(p), "описание схемы")
                self.assertEqual(d(p), "описание схемы")  # второй раз — из кеша
        self.assertEqual(len(calls), 1)


class TestVisionChunk(unittest.TestCase):
    def test_description_appended_to_image_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "diagram.png"
            p.write_bytes(_PNG)
            chunks = build_document_chunks(
                p, 512, vision_describe=lambda _img: "На схеме: ВМ host-01, подсистема Beta.")
        self.assertTrue(chunks)
        self.assertIn("ВМ host-01", chunks[0].text)
        self.assertIn("подсистема Beta", chunks[0].text)

    def test_no_describer_leaves_chunk_header_only(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "diagram.png"
            p.write_bytes(_PNG)
            chunks = build_document_chunks(p, 512)  # без vision_describe
        self.assertTrue(chunks)
        self.assertNotIn("ВМ host-01", chunks[0].text)


if __name__ == "__main__":
    unittest.main()
