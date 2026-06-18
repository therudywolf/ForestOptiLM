# SPDX-License-Identifier: AGPL-3.0-or-later
"""Notebook store: data-root resolution, CRUD, sources, chat history."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import notebook_store as nbs


class TestNotebookChunkTokens(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = os.environ.pop("NOCTURNE_NB_CHUNK_TOKENS", None)

    def tearDown(self) -> None:
        if self._saved is None:
            os.environ.pop("NOCTURNE_NB_CHUNK_TOKENS", None)
        else:
            os.environ["NOCTURNE_NB_CHUNK_TOKENS"] = self._saved

    def test_default_is_small(self) -> None:
        # Дефолт — маленький чанк под embedding (а не тысячи токенов под чат-модель).
        self.assertEqual(nbs.notebook_index_chunk_tokens(), 512)

    def test_env_override(self) -> None:
        os.environ["NOCTURNE_NB_CHUNK_TOKENS"] = "768"
        self.assertEqual(nbs.notebook_index_chunk_tokens(), 768)

    def test_clamped_high_and_low(self) -> None:
        os.environ["NOCTURNE_NB_CHUNK_TOKENS"] = "9000"
        self.assertEqual(nbs.notebook_index_chunk_tokens(), 1024)
        os.environ["NOCTURNE_NB_CHUNK_TOKENS"] = "10"
        self.assertEqual(nbs.notebook_index_chunk_tokens(), 256)

    def test_bad_env_falls_back(self) -> None:
        os.environ["NOCTURNE_NB_CHUNK_TOKENS"] = "abc"
        self.assertEqual(nbs.notebook_index_chunk_tokens(), 512)


class TestNotebooksRoot(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in ("NOCTURNE_NOTEBOOKS_DIR", "NOCTURNE_CACHE_DIR")}
        for k in self._saved:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_explicit_override_wins(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            os.environ["NOCTURNE_NOTEBOOKS_DIR"] = tmp
            os.environ["NOCTURNE_CACHE_DIR"] = "/some/other/.nocturne_cache"
            self.assertEqual(nbs.notebooks_root(), Path(tmp))

    def test_derived_from_cache_dir(self) -> None:
        # Упакованный .exe выставляет NOCTURNE_CACHE_DIR=NocturneData/.nocturne_cache;
        # блокноты должны лечь рядом, в NocturneData/notebooks.
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp) / "NocturneData" / ".nocturne_cache"
            os.environ["NOCTURNE_CACHE_DIR"] = str(cache)
            # notebooks_root() резолвит cache; сравниваем так же (macOS/Windows temp
            # бывает симлинком/8.3-путём, нерезолвленное сравнение ложно упадёт).
            expected = cache.resolve().parent / "notebooks"
            self.assertEqual(nbs.notebooks_root(), expected)

    def test_default_local(self) -> None:
        root = nbs.notebooks_root()
        self.assertEqual(root.name, "notebooks")
        self.assertEqual(root.parent.name, ".local")

    def test_frozen_fallback_next_to_exe(self) -> None:
        # В упакованном .exe без env-переменных блокноты должны лечь рядом с .exe,
        # а не внутрь read-only бандла.
        import sys
        from unittest import mock

        fake_exe = str(Path(tempfile.gettempdir()) / "NocturneDataForge.exe")
        with mock.patch.object(sys, "frozen", True, create=True), \
                mock.patch.object(sys, "executable", fake_exe):
            root = nbs.notebooks_root()
        self.assertEqual(root, Path(fake_exe).resolve().parent / "NocturneData" / "notebooks")


class TestNotebookCrud(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        os.environ["NOCTURNE_NOTEBOOKS_DIR"] = self._tmp.name
        self.addCleanup(lambda: os.environ.pop("NOCTURNE_NOTEBOOKS_DIR", None))

    def test_create_list_load_delete(self) -> None:
        nb = nbs.create_notebook("Анализ рынка")
        self.assertTrue(nb.dir.is_dir())
        self.assertEqual(nbs.get_last_active(), nb.id)
        listed = nbs.list_notebooks()
        self.assertEqual([n.id for n in listed], [nb.id])
        again = nbs.load_notebook(nb.id)
        self.assertIsNotNone(again)
        self.assertEqual(again.name, "Анализ рынка")  # type: ignore[union-attr]
        self.assertTrue(nbs.delete_notebook(nb.id))
        self.assertEqual(nbs.list_notebooks(), [])
        self.assertEqual(nbs.get_last_active(), "")

    def test_unicode_name_roundtrip(self) -> None:
        nb = nbs.create_notebook("Изучение ИБ — дампы")
        reloaded = nbs.load_notebook(nb.id)
        self.assertEqual(reloaded.name, "Изучение ИБ — дампы")  # type: ignore[union-attr]

    def test_cyrillic_name_yields_ascii_dir(self) -> None:
        # faiss on Windows can't open non-ASCII index paths — the folder name MUST
        # be ASCII even when the user's notebook name is fully Cyrillic.
        nb = nbs.create_notebook("Изучение ИБ")
        self.assertTrue(nb.id.isascii(), f"non-ASCII notebook id: {nb.id!r}")
        self.assertTrue(str(nb.dir).isascii(), f"non-ASCII notebook dir: {nb.dir!r}")
        self.assertEqual(nb.name, "Изучение ИБ")  # display name preserved
        # Mixed name keeps its ASCII part.
        nb2 = nbs.create_notebook("E2E проверка")
        self.assertTrue(nb2.id.isascii())
        self.assertIn("E2E", nb2.id)

    def test_add_path_source_dedup(self) -> None:
        nb = nbs.create_notebook("nb")
        f = Path(self._tmp.name) / "a.txt"
        f.write_text("hello", encoding="utf-8")
        s1 = nb.add_path_source(f)
        s2 = nb.add_path_source(f)  # тот же путь → без дубля
        self.assertEqual(s1.id, s2.id)
        self.assertEqual(len(nb.sources), 1)
        self.assertEqual(s1.kind, "file")

    def test_url_source_written_and_removable(self) -> None:
        nb = nbs.create_notebook("nb")
        src = nb.add_url_source("https://example.com/x", "тело страницы", "Заголовок")
        p = Path(src.path)
        self.assertTrue(p.is_file())
        self.assertIn("SOURCE_URL", p.read_text(encoding="utf-8"))
        self.assertEqual(src.kind, "url")
        self.assertTrue(nb.remove_source(src.id))
        self.assertFalse(p.exists())  # производный файл удалён
        self.assertEqual(nb.sources, [])

    def test_index_input_paths_filters_missing(self) -> None:
        nb = nbs.create_notebook("nb")
        f = Path(self._tmp.name) / "real.txt"
        f.write_text("x", encoding="utf-8")
        nb.add_path_source(f)
        # Подсунем источник с несуществующим путём напрямую.
        nb.sources.append(nbs.Source(id="src_ghost", kind="file", display="ghost",
                                     path=str(Path(self._tmp.name) / "missing.txt")))
        paths = nb.index_input_paths()
        self.assertEqual([p.name for p in paths], ["real.txt"])

    def test_chat_history_roundtrip(self) -> None:
        nb = nbs.create_notebook("nb")
        nb.append_chat_turn("user", "вопрос?")
        nb.append_chat_turn("assistant", "ответ [1]", [{"n": 1, "display": "a.txt"}])
        hist = nb.load_chat()
        self.assertEqual(len(hist), 2)
        self.assertEqual(hist[0]["role"], "user")
        self.assertEqual(hist[1]["citations"][0]["n"], 1)
        nb.clear_chat()
        self.assertEqual(nb.load_chat(), [])

    def test_notes_save_and_list(self) -> None:
        nb = nbs.create_notebook("nb")
        p = nb.save_note("study_guide.md", "# Гайд")
        self.assertTrue(p.is_file())
        names = [n.name for n in nb.list_notes()]
        self.assertIn("study_guide.md", names)

    def test_cover_assigned_on_create(self) -> None:
        nb = nbs.create_notebook("nb")
        self.assertTrue(nb.color.startswith("#"))
        self.assertTrue(nb.emoji)

    def test_cover_backfilled_for_legacy_meta(self) -> None:
        nb = nbs.create_notebook("nb")
        # Эмулируем старый notebook.json без полей обложки.
        import json
        meta = json.loads((nb.dir / "notebook.json").read_text(encoding="utf-8"))
        meta.pop("emoji", None)
        meta.pop("color", None)
        (nb.dir / "notebook.json").write_text(json.dumps(meta), encoding="utf-8")
        reloaded = nbs.load_notebook(nb.id)
        self.assertTrue(reloaded.color.startswith("#"))  # type: ignore[union-attr]
        self.assertTrue(reloaded.emoji)  # type: ignore[union-attr]

    def test_set_meta_roundtrip(self) -> None:
        nb = nbs.create_notebook("Старое имя")
        nb.set_meta(name="Новое имя", description="Описание корпуса", emoji="🔬")
        reloaded = nbs.load_notebook(nb.id)
        self.assertEqual(reloaded.name, "Новое имя")  # type: ignore[union-attr]
        self.assertEqual(reloaded.description, "Описание корпуса")  # type: ignore[union-attr]
        self.assertEqual(reloaded.emoji, "🔬")  # type: ignore[union-attr]

    def test_meta_survives_reload(self) -> None:
        nb = nbs.create_notebook("nb")
        nb.embedding_model = "emb-1"
        nb.index_chunks = 42
        nb.index_files = 3
        nb.save()
        reloaded = nbs.load_notebook(nb.id)
        self.assertEqual(reloaded.embedding_model, "emb-1")  # type: ignore[union-attr]
        self.assertEqual(reloaded.index_chunks, 42)  # type: ignore[union-attr]


if __name__ == "__main__":
    unittest.main()
