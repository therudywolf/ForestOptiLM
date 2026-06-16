# SPDX-License-Identifier: AGPL-3.0-or-later
"""Studio material generation: digest, prompts, flashcard parsing."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import notebook_store as nbs
import notebook_studio as st


class TestPureFunctions(unittest.TestCase):
    def test_read_index_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            meta = Path(tmp) / "chunks_meta.jsonl"
            meta.write_text(
                json.dumps({"text": "alpha"}) + "\n" + json.dumps({"text": "beta"}) + "\n"
                + "\n" + "{not json}\n",
                encoding="utf-8",
            )
            self.assertEqual(st.read_index_chunks(Path(tmp)), ["alpha", "beta"])

    def test_digest_full_when_small(self) -> None:
        d = st.gather_corpus_digest(["one", "two", "three"], max_tokens=100000)
        self.assertFalse(d.sampled)
        self.assertEqual(d.chunks_used, 3)
        self.assertEqual(d.chunks_total, 3)

    def test_digest_samples_when_large(self) -> None:
        chunks = [f"chunk number {i} " * 50 for i in range(500)]
        d = st.gather_corpus_digest(chunks, max_tokens=200)
        self.assertTrue(d.sampled)
        self.assertLess(d.chunks_used, d.chunks_total)
        self.assertGreater(d.chunks_used, 0)

    def test_digest_empty(self) -> None:
        d = st.gather_corpus_digest([], max_tokens=100)
        self.assertEqual(d.chunks_used, 0)
        self.assertEqual(d.text, "")

    def test_build_material_messages_notes_sampling(self) -> None:
        d = st.CorpusDigest(text="DATA", chunks_used=10, chunks_total=500, sampled=True)
        msgs = st.build_material_messages(st.MATERIALS["study_guide"], d, notebook_name="NB")
        self.assertEqual(msgs[0]["role"], "system")
        self.assertIn("DATA", msgs[1]["content"])
        self.assertIn("выборка", msgs[1]["content"])
        self.assertIn("NB", msgs[1]["content"])

    def test_parse_flashcards_fenced(self) -> None:
        raw = 'Вот карточки:\n```json\n[{"front":"Q1","back":"A1"},{"front":"Q2","back":"A2"}]\n```'
        cards = st.parse_flashcards(raw)
        self.assertEqual(cards, [{"front": "Q1", "back": "A1"}, {"front": "Q2", "back": "A2"}])

    def test_parse_flashcards_plain_array(self) -> None:
        cards = st.parse_flashcards('[{"question":"Q","answer":"A"}]')
        self.assertEqual(cards, [{"front": "Q", "back": "A"}])

    def test_parse_flashcards_invalid(self) -> None:
        self.assertEqual(st.parse_flashcards("совсем не json"), [])

    def test_all_specs_have_order(self) -> None:
        self.assertEqual(set(st.MATERIAL_ORDER), set(st.MATERIALS.keys()))


class TestGenerateMaterial(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        os.environ["NOCTURNE_NOTEBOOKS_DIR"] = self._tmp.name
        self.addCleanup(lambda: os.environ.pop("NOCTURNE_NOTEBOOKS_DIR", None))
        self.nb = nbs.create_notebook("nb")
        self.nb.index_dir.mkdir(parents=True, exist_ok=True)
        (self.nb.index_dir / "chunks_meta.jsonl").write_text(
            json.dumps({"text": "Корпус про xz backdoor."}) + "\n", encoding="utf-8")

    def test_generate_markdown_material(self) -> None:
        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "## Краткий обзор\nСодержимое гайда."

        with mock.patch("processor.call_llm", new=fake_call_llm):
            path, content = asyncio.run(st.generate_material(
                self.nb, "study_guide", base_url="u", api_key="", chat_model="m"))
        self.assertEqual(Path(path).name, "study_guide.md")
        self.assertIn("Содержимое гайда", content)
        self.assertTrue(Path(path).is_file())

    def test_generate_flashcards_saves_json_and_md(self) -> None:
        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return '[{"front":"Что такое xz?","back":"Утилита сжатия"}]'

        with mock.patch("processor.call_llm", new=fake_call_llm):
            path, content = asyncio.run(st.generate_material(
                self.nb, "flashcards", base_url="u", api_key="", chat_model="m"))
        self.assertEqual(Path(path).name, "flashcards.json")
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.assertEqual(data[0]["front"], "Что такое xz?")
        # Человекочитаемая версия тоже сохранена.
        names = [n.name for n in self.nb.list_notes()]
        self.assertIn("flashcards.md", names)

    def test_generate_raises_without_index(self) -> None:
        nb2 = nbs.create_notebook("empty")
        with self.assertRaises(RuntimeError):
            asyncio.run(st.generate_material(
                nb2, "faq", base_url="u", api_key="", chat_model="m"))


if __name__ == "__main__":
    unittest.main()
