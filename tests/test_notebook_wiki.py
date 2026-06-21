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


class TestSaveAnswerPage(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        os.environ["NOCTURNE_NOTEBOOKS_DIR"] = self._tmp.name
        self.addCleanup(lambda: os.environ.pop("NOCTURNE_NOTEBOOKS_DIR", None))
        self.nb = nbs.create_notebook("nb")

    def test_slug_is_filename_safe(self) -> None:
        self.assertEqual(wk._slug("На каких ВМ?! / Alpha"), "на-каких-вм-Alpha")
        self.assertEqual(wk._slug("   "), "answer")

    def test_saves_page_and_logs(self) -> None:
        path = wk.save_answer_page(
            self.nb, "Где Alpha?", "На host-07 [1].",
            citations=["entities.md · стр. 1"], ts="2026-06-21")
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent.name, "answers")
        body = path.read_text(encoding="utf-8")
        self.assertIn("# Где Alpha?", body)
        self.assertIn("На host-07 [1].", body)
        self.assertIn("entities.md", body)
        log = (self.nb.wiki_dir / wk.WIKI_LOG_FILE).read_text(encoding="utf-8")
        self.assertIn("] answer | Где Alpha?", log)

    def test_distinct_answers_same_day_do_not_overwrite(self) -> None:
        # Ревью beta.11: разные ответы одного дня раньше перезаписывали друг друга.
        p1 = wk.save_answer_page(self.nb, "Вопрос про ВМ", "Ответ А", ts="2026-06-21")
        p2 = wk.save_answer_page(self.nb, "Вопрос про ВМ", "Ответ Б (другой)", ts="2026-06-21")
        self.assertNotEqual(p1, p2)
        self.assertTrue(p1.is_file() and p2.is_file())
        # идентичный Q+A → тот же файл (идемпотентно)
        p3 = wk.save_answer_page(self.nb, "Вопрос про ВМ", "Ответ А", ts="2026-06-21")
        self.assertEqual(p1, p3)

    def test_indexes_answer_when_wiki_index_exists(self) -> None:
        # Ревью beta.11 #5: подшитый ответ доиндексируется, если вики уже в индексе.
        self.nb.wiki_index_dir.mkdir(parents=True, exist_ok=True)
        (self.nb.wiki_index_dir / "chunks_meta.jsonl").write_text("{}\n", encoding="utf-8")
        self.assertTrue(self.nb.has_wiki_index)
        called = {}

        def fake_add(*, input_paths, index_dir, **kw):
            called["paths"] = [str(p) for p in input_paths]
            called["dir"] = str(index_dir)
            return (None, True)

        with mock.patch("pipeline.add_to_index", new=fake_add):
            path = wk.save_answer_page(self.nb, "Q", "A", ts="2026-06-21",
                                       base_url="u", api_key="", embedding_model="nomic")
        self.assertEqual(called["dir"], str(self.nb.wiki_index_dir))
        self.assertIn(str(path), called["paths"])  # новый ответ в наборе источников


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

    def test_pages_have_obsidian_wikilinks_footer(self) -> None:
        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "# Стр\n\nконтент"

        with mock.patch("processor.call_llm", new=fake_call_llm):
            asyncio.run(wk.compile_wiki(self.nb, base_url="u", api_key="", chat_model="m", ts="2026-06-21"))
        overview = (self.nb.wiki_dir / "overview.md").read_text(encoding="utf-8")
        self.assertIn("См. также", overview)
        self.assertIn("[[entities]]", overview)       # ссылка на соседнюю страницу
        self.assertNotIn("[[overview]]", overview)    # не ссылается сам на себя

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


try:
    import faiss  # noqa: F401
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False


@unittest.skipUnless(_HAVE_FAISS, "faiss not installed")
class TestWikiRetrieval(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        os.environ["NOCTURNE_NOTEBOOKS_DIR"] = self._tmp.name
        self.addCleanup(lambda: os.environ.pop("NOCTURNE_NOTEBOOKS_DIR", None))
        self.nb = nbs.create_notebook("nb")

    def _build(self, index_dir, chunks):
        from models import DocumentChunk
        from retrieval import LocalFaissStore
        import pipeline
        pipeline._STORE_CACHE.clear()  # не тащить кэш между индексами теста
        dcs = [DocumentChunk(c[0], c[1], c[2], 6, {}) for c in chunks]
        vecs = [[1.0, 0.0, 0.0, 0.0] for _ in dcs]
        LocalFaissStore(index_dir=index_dir).build(dcs, vecs, embedding_model="fake")

    def test_query_puts_wiki_hits_first(self) -> None:
        # raw-индекс и wiki-индекс содержат разный контент про «Alpha»;
        # вики-фрагмент должен оказаться ВПЕРЕДИ сырых (без сервера → BM25-путь).
        self._build(self.nb.index_dir, [
            ("r1", "raw.txt", "сырой фрагмент про Alpha из документа")])
        self._build(self.nb.wiki_index_dir, [
            ("w1", "wiki/entities.md", "скомпилированное знание про Alpha")])
        self.assertTrue(self.nb.has_wiki_index)
        # эмбеддинг недоступен → быстрый BM25-путь (без сетевых ретраев)
        with mock.patch("embeddings.EmbeddingClient.embed_texts",
                        side_effect=ConnectionError("offline")):
            hits = self.nb.query("Alpha", base_url="http://127.0.0.1:1", api_key="",
                                 embedding_model="fake", top_k=5)
        ids = [h.chunk_id for h in hits]
        self.assertIn("w1", ids)
        self.assertIn("r1", ids)
        self.assertEqual(ids[0], "w1")  # вики впереди

    def test_query_without_wiki_returns_raw_only(self) -> None:
        self._build(self.nb.index_dir, [("r1", "raw.txt", "только сырьё про Beta")])
        self.assertFalse(self.nb.has_wiki_index)
        with mock.patch("embeddings.EmbeddingClient.embed_texts",
                        side_effect=ConnectionError("offline")):
            hits = self.nb.query("Beta", base_url="http://127.0.0.1:1", api_key="",
                                 embedding_model="fake", top_k=5)
        self.assertEqual([h.chunk_id for h in hits], ["r1"])


class TestCompileBuildsWikiIndex(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        os.environ["NOCTURNE_NOTEBOOKS_DIR"] = self._tmp.name
        self.addCleanup(lambda: os.environ.pop("NOCTURNE_NOTEBOOKS_DIR", None))
        self.nb = nbs.create_notebook("nb")
        self.nb.index_dir.mkdir(parents=True, exist_ok=True)
        (self.nb.index_dir / "chunks_meta.jsonl").write_text(
            json.dumps({"text": "Корпус про подсистемы."}) + "\n", encoding="utf-8")

    def test_indexed_true_when_embedding_given(self) -> None:
        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "# X\n\nконтент страницы"

        captured = {}

        def fake_build_index(*, input_paths, index_dir, **kw):
            captured["paths"] = [str(p) for p in input_paths]
            index_dir.mkdir(parents=True, exist_ok=True)
            (index_dir / "chunks_meta.jsonl").write_text("{}\n", encoding="utf-8")

        with mock.patch("processor.call_llm", new=fake_call_llm), \
                mock.patch("pipeline.build_index", new=fake_build_index):
            res = asyncio.run(wk.compile_wiki(
                self.nb, base_url="u", api_key="", chat_model="m",
                embedding_model="nomic", ts="2026-06-21"))
        self.assertTrue(res["indexed"])
        # индексируем страницы, а не index.md/log.md
        self.assertTrue(all("index.md" not in p and "log.md" not in p for p in captured["paths"]))
        self.assertIn("проиндексировано", (self.nb.wiki_dir / wk.WIKI_LOG_FILE).read_text(encoding="utf-8"))

    def test_incomplete_compile_removes_stale_index(self) -> None:
        # Ревью beta.11 #2/#8: при остановке/без embed старый wiki-индекс устаревает
        # и не должен цитироваться → его удаляем.
        self.nb.wiki_index_dir.mkdir(parents=True, exist_ok=True)
        (self.nb.wiki_index_dir / "chunks_meta.jsonl").write_text("stale\n", encoding="utf-8")
        self.assertTrue(self.nb.has_wiki_index)

        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "# X\n\nновый контент"

        with mock.patch("processor.call_llm", new=fake_call_llm):
            res = asyncio.run(wk.compile_wiki(  # без embedding → индекс не пересобрать
                self.nb, base_url="u", api_key="", chat_model="m", ts="2026-06-21"))
        self.assertFalse(res["indexed"])
        self.assertFalse(self.nb.has_wiki_index)  # устаревший индекс убран

    def test_stopped_compile_is_not_completed(self) -> None:
        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "# X\n\nтекст"

        with mock.patch("processor.call_llm", new=fake_call_llm):
            res = asyncio.run(wk.compile_wiki(
                self.nb, base_url="u", api_key="", chat_model="m", ts="2026-06-21",
                stop_flag=lambda: True))
        self.assertFalse(res["completed"])

    def test_indexed_false_without_embedding(self) -> None:
        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            return "# X\n\nконтент"

        with mock.patch("processor.call_llm", new=fake_call_llm):
            res = asyncio.run(wk.compile_wiki(
                self.nb, base_url="u", api_key="", chat_model="m", ts="2026-06-21"))
        self.assertFalse(res["indexed"])

    def test_schema_roundtrips_and_steers_compile(self) -> None:
        # B4: схема домена сохраняется и подмешивается в дайджест при компиляции.
        self.nb.set_meta(schema="Домен: ACME; сущности — подсистемы и ВМ.")
        reloaded = nbs.load_notebook(self.nb.id)
        self.assertIn("ACME", reloaded.schema)

        captured: dict = {}

        async def fake_call_llm(messages, model, base_url, api_key, semaphore, **kw):
            captured["user"] = messages[1]["content"]
            return "# X\n\nтекст"

        with mock.patch("processor.call_llm", new=fake_call_llm):
            asyncio.run(wk.compile_wiki(reloaded, base_url="u", api_key="", chat_model="m", ts="2026-06-21"))
        self.assertIn("[Контекст домена]", captured["user"])
        self.assertIn("ACME", captured["user"])


if __name__ == "__main__":
    unittest.main()
