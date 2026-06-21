# SPDX-License-Identifier: AGPL-3.0-or-later
"""Слой скомпилированных знаний (LLM-Wiki): index.md/log.md + compile_wiki."""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import unittest
from unittest import mock

import notebook_store as nbs
import notebook_wiki as wk


class TestPureFunctions(unittest.TestCase):
    def test_page_summary_skips_heading_and_tables(self) -> None:
        md = "# Обзор\n\n| a | b |\nКорпус про архитектуру подсистем."
        self.assertEqual(wk.page_summary(md), "Корпус про архитектуру подсистем.")

    def test_page_summary_strips_bullet(self) -> None:
        self.assertEqual(wk.page_summary("# T\n- первый пункт"), "первый пункт")

    def test_page_summary_truncates(self) -> None:
        s = wk.page_summary("текст " * 50, max_len=20)
        self.assertTrue(s.endswith("…"))
        self.assertLessEqual(len(s), 21)

    def test_page_summary_empty(self) -> None:
        self.assertEqual(wk.page_summary("# Только заголовок"), "")

    def test_build_index_lists_pages_with_links(self) -> None:
        idx = wk.build_wiki_index(
            [("Обзор", "overview.md", "о чём корпус"), ("Сущности", "entities.md", "")],
            notebook_name="МойБлокнот")
        self.assertIn("# Вики блокнота: МойБлокнот", idx)
        self.assertIn("- [Обзор](overview.md) — о чём корпус", idx)
        self.assertIn("- [Сущности](entities.md)", idx)

    def test_append_log_creates_header_then_appends(self) -> None:
        first = wk.append_log("", "compile", "4 страниц(ы)", "2026-06-21")
        self.assertIn("# Журнал операций", first)
        self.assertIn("## [2026-06-21] compile | 4 страниц(ы)", first)
        second = wk.append_log(first, "query", "вопрос про ВМ", "2026-06-22")
        # append-only: старая запись на месте, новая снизу
        self.assertIn("## [2026-06-21] compile", second)
        self.assertIn("## [2026-06-22] query | вопрос про ВМ", second)
        self.assertLess(second.index("2026-06-21"), second.index("2026-06-22"))


class TestCompileWiki(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        os.environ["NOCTURNE_NOTEBOOKS_DIR"] = self._tmp.name
        self.addCleanup(lambda: os.environ.pop("NOCTURNE_NOTEBOOKS_DIR", None))
        self.nb = nbs.create_notebook("nb")
        self.nb.index_dir.mkdir(parents=True, exist_ok=True)
        (self.nb.index_dir / "chunks_meta.jsonl").write_text(
            json.dumps({"text": "Подсистема Alpha на host-07. Beta на host-01."}) + "\n",
            encoding="utf-8")

    def test_compiles_pages_index_and_log(self) -> None:
        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "# Стр\n\nСодержимое скомпилированной страницы про host-07."

        with mock.patch("processor.call_llm", new=fake_call_llm):
            res = asyncio.run(wk.compile_wiki(
                self.nb, base_url="u", api_key="", chat_model="m", ts="2026-06-21"))

        self.assertEqual(len(res["pages"]), len(wk.WIKI_PAGES))
        wiki = self.nb.wiki_dir
        for spec in wk.WIKI_PAGES:
            self.assertTrue((wiki / spec.filename).is_file())
        index_md = (wiki / wk.WIKI_INDEX_FILE).read_text(encoding="utf-8")
        self.assertIn("overview.md", index_md)
        log_md = (wiki / wk.WIKI_LOG_FILE).read_text(encoding="utf-8")
        self.assertIn("## [2026-06-21] compile", log_md)

    def test_log_is_append_only_across_runs(self) -> None:
        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "# X\n\nтекст"

        with mock.patch("processor.call_llm", new=fake_call_llm):
            asyncio.run(wk.compile_wiki(self.nb, base_url="u", api_key="", chat_model="m", ts="2026-06-21"))
            asyncio.run(wk.compile_wiki(self.nb, base_url="u", api_key="", chat_model="m", ts="2026-06-22"))
        log_md = (self.nb.wiki_dir / wk.WIKI_LOG_FILE).read_text(encoding="utf-8")
        self.assertEqual(log_md.count("] compile |"), 2)  # обе записи на месте

    def test_raises_without_index(self) -> None:
        nb2 = nbs.create_notebook("empty")
        with self.assertRaises(RuntimeError):
            asyncio.run(wk.compile_wiki(nb2, base_url="u", api_key="", chat_model="m"))

    def test_stop_flag_halts_generation(self) -> None:
        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "# X\n\nтекст"

        with mock.patch("processor.call_llm", new=fake_call_llm):
            res = asyncio.run(wk.compile_wiki(
                self.nb, base_url="u", api_key="", chat_model="m",
                ts="2026-06-21", stop_flag=lambda: True))
        self.assertEqual(res["pages"], [])  # остановлено до первой страницы


if __name__ == "__main__":
    unittest.main()
