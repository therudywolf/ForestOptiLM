# SPDX-License-Identifier: AGPL-3.0-or-later
"""Hybrid (vector + BM25 / RRF) retrieval over a real FAISS index with fake vectors."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

try:
    import faiss  # noqa: F401
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False

from models import DocumentChunk


@unittest.skipUnless(_HAVE_FAISS, "faiss not installed")
class TestHybridSearch(unittest.TestCase):
    def _store(self, index_dir: Path):
        from retrieval import LocalFaissStore

        chunks = [
            DocumentChunk("c0", "a.txt", "memory leak in cache module", 6, {}),
            DocumentChunk("c1", "b.txt", "reflected xss in login form", 6, {}),
            DocumentChunk("c2", "c.txt", "remote code execution CVE-2024-3094 xz backdoor", 7, {}),
        ]
        # Fake unit vectors: query will be closest to c0 by vector, but the exact
        # CVE term lives only in c2 — BM25 must pull c2 up via RRF.
        vectors = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        store = LocalFaissStore(index_dir=index_dir)
        store.build(chunks, vectors, embedding_model="fake")
        return store

    def test_exact_cve_found_via_bm25_even_with_offtarget_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self._store(Path(td))
            # Vector points at c0; lexical query is the CVE that's only in c2.
            hits = store.hybrid_search(
                query_text="CVE-2024-3094", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=3,
            )
            ids = [h.chunk_id for h in hits]
            self.assertIn("c2", ids)

    def test_bm25_only_when_no_vector(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self._store(Path(td))
            hits = store.hybrid_search(query_text="xss login", query_vector=None, top_k=2)
            self.assertTrue(hits)
            self.assertEqual(hits[0].chunk_id, "c1")

    def test_min_score_ratio_trims_tail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            store = self._store(Path(td))
            full = store.hybrid_search(
                query_text="memory leak", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=3,
            )
            trimmed = store.hybrid_search(
                query_text="memory leak", query_vector=[1.0, 0.0, 0.0, 0.0], top_k=3,
                min_score_ratio=0.95,  # агрессивно: оставить лишь почти-лучшие
            )
            self.assertLessEqual(len(trimmed), len(full))
            self.assertTrue(trimmed)  # лучший фрагмент никогда не выкидываем
            # Все оставшиеся — не ниже порога относительно лучшего.
            top = trimmed[0].score
            self.assertTrue(all(h.score >= top * 0.95 - 1e-9 for h in trimmed))

    def test_store_for_reuses_instance_per_dir(self) -> None:
        import pipeline
        with tempfile.TemporaryDirectory() as td:
            pipeline._STORE_CACHE.clear()
            a = pipeline._store_for(Path(td))
            b = pipeline._store_for(Path(td))
            self.assertIs(a, b)  # тот же инстанс → кэш FAISS/BM25 переживает запросы

    def test_missing_index_mid_query_returns_empty(self) -> None:
        # Ревью beta.8: блокнот удалили в соседнем потоке между .exists() и stat()
        # → раньше летел FileNotFoundError. Теперь — пустой результат, без падения.
        with tempfile.TemporaryDirectory() as td:
            store = self._store(Path(td))
            store.index_file.unlink()
            store.meta_file.unlink()
            self.assertEqual(store.hybrid_search("x", [1.0, 0.0, 0.0, 0.0], top_k=3), [])
            self.assertEqual(store.search([1.0, 0.0, 0.0, 0.0], top_k=3), [])

    def test_evict_store_drops_cache_entry(self) -> None:
        import pipeline
        with tempfile.TemporaryDirectory() as td:
            pipeline._STORE_CACHE.clear()
            a = pipeline._store_for(Path(td))
            pipeline._evict_store(Path(td))
            b = pipeline._store_for(Path(td))
            self.assertIsNot(a, b)  # после пересборки кэш сброшен → новый инстанс

    def test_cache_invalidates_on_same_mtime_content_change(self) -> None:
        # Регрессия (ревью beta.8): на FS с грубым mtime (FAT32/exFAT/USB/сеть)
        # пересборка может попасть в тот же mtime. Инвалидация по mtime+РАЗМЕРУ
        # файлов ловит смену содержимого даже без сдвига mtime → не отдаём устаревший
        # корпус (та самая боль «нет ответа» после переиндексации).
        import os

        from retrieval import LocalFaissStore
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            store = self._store(d)  # c0,c1,c2
            store.hybrid_search("memory leak", [1.0, 0.0, 0.0, 0.0], top_k=3)  # прогрев кэша
            old_idx = store.index_file.stat().st_mtime_ns
            old_meta = store.meta_file.stat().st_mtime_ns

            # Пересобираем НА ДИСКЕ другим содержимым (1 чанк вместо 3 → другой размер).
            LocalFaissStore(index_dir=d).build(
                [DocumentChunk("z0", "z.txt", "completely different zebra content", 5, {})],
                [[0.0, 0.0, 0.0, 1.0]], embedding_model="fake")
            # Имитируем грубый mtime: возвращаем файлам СТАРОЕ время.
            os.utime(store.index_file, ns=(old_idx, old_idx))
            os.utime(store.meta_file, ns=(old_meta, old_meta))

            # Тот же закэшированный стор обязан увидеть НОВОЕ содержимое (размер сменился).
            hits = store.hybrid_search("zebra", [0.0, 0.0, 0.0, 1.0], top_k=3)
            ids = [h.chunk_id for h in hits]
            self.assertIn("z0", ids)
            self.assertNotIn("c0", ids)

    def test_meta_pickle_cache_written_and_reused(self) -> None:
        # Холодный старт: первый запрос парсит JSONL и пишет meta_cache.pkl;
        # свежий инстанс (пустой in-memory кэш) обязан поднять корпус из pickle.
        from retrieval import LocalFaissStore
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            store = self._store(d)
            store.hybrid_search("memory leak", [1.0, 0.0, 0.0, 0.0], top_k=3)
            self.assertTrue(store.meta_cache_file.is_file())  # кэш записан

            fresh = LocalFaissStore(index_dir=d)  # без прогретого in-memory кэша
            hits = fresh.hybrid_search("CVE-2024-3094", [1.0, 0.0, 0.0, 0.0], top_k=3)
            self.assertIn("c2", [h.chunk_id for h in hits])  # данные из pickle корректны

    def test_meta_pickle_cache_ignored_on_stale_signature(self) -> None:
        # Пересборка меняет сигнатуру → устаревший pickle игнорируется, JSONL
        # перечитывается. Иначе получили бы старый корпус (боль «нет ответа»).
        from retrieval import LocalFaissStore
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            store = self._store(d)
            store.hybrid_search("memory leak", [1.0, 0.0, 0.0, 0.0], top_k=3)  # пишет pickle

            LocalFaissStore(index_dir=d).build(
                [DocumentChunk("z0", "z.txt", "completely different zebra content", 5, {})],
                [[0.0, 0.0, 0.0, 1.0]], embedding_model="fake")

            fresh = LocalFaissStore(index_dir=d)
            hits = fresh.hybrid_search("zebra", [0.0, 0.0, 0.0, 1.0], top_k=3)
            ids = [h.chunk_id for h in hits]
            self.assertIn("z0", ids)
            self.assertNotIn("c0", ids)  # старый корпус из stale pickle не всплыл

    def test_meta_pickle_cache_corrupt_falls_back(self) -> None:
        # Битый pickle не должен ронять запрос — молча перечитываем JSONL.
        from retrieval import LocalFaissStore
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            store = self._store(d)
            store.meta_cache_file.write_bytes(b"\x00 not a pickle \xff")

            fresh = LocalFaissStore(index_dir=d)
            hits = fresh.hybrid_search("CVE-2024-3094", [1.0, 0.0, 0.0, 0.0], top_k=3)
            self.assertIn("c2", [h.chunk_id for h in hits])
            # После фолбэка кэш перезаписан валидным содержимым.
            import pickle
            payload = pickle.loads(store.meta_cache_file.read_bytes())
            self.assertIsInstance(payload, dict)
            self.assertIn("meta", payload)

    def test_meta_cache_concurrent_writers_no_corruption_no_litter(self) -> None:
        # Ревью beta.24: фикс. tmp-путь ронял конкурентные записи (WinError 32/5)
        # и оставлял мусор. Уникальный tmp (pid+uuid) → много писателей разом дают
        # валидный финальный pickle и НЕ оставляют *.tmp в каталоге индекса.
        import pickle
        import threading
        from retrieval import LocalFaissStore
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            store = self._store(d)
            sig = store._index_signature()
            meta = store._read_meta()

            barrier = threading.Barrier(6)

            def writer() -> None:
                barrier.wait()  # стартуем одновременно → максимальное перекрытие
                store._write_meta_cache(sig, meta)

            threads = [threading.Thread(target=writer) for _ in range(6)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # Финальный кэш валиден и совпадает по сигнатуре.
            payload = pickle.loads(store.meta_cache_file.read_bytes())
            self.assertEqual(tuple(payload["sig"]), tuple(sig))
            self.assertEqual(len(payload["meta"]), len(meta))
            # Ни одного осиротевшего tmp (уникальные имена + cleanup в finally).
            leftovers = list(d.glob("meta_cache.pkl.*.tmp")) + list(d.glob("*.tmp"))
            self.assertEqual(leftovers, [], f"осиротевшие tmp: {leftovers}")

    def test_read_meta_cached_returns_needs_write_flag(self) -> None:
        # Фикс #2: dump вынесен из-под лока. _read_meta_cached сигналит
        # (meta, needs_write): miss → True (писать надо), hit → False.
        from retrieval import LocalFaissStore
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            store = self._store(d)
            sig = store._index_signature()
            store.meta_cache_file.unlink(missing_ok=True)  # гарантируем промах

            meta, needs_write = store._read_meta_cached(sig)
            self.assertTrue(needs_write)               # промах → нужна запись
            self.assertEqual(len(meta), 3)
            self.assertFalse(store.meta_cache_file.exists())  # сам метод НЕ пишет

            store._write_meta_cache(sig, meta)         # пишет вызывающая сторона
            meta2, needs_write2 = store._read_meta_cached(sig)
            self.assertFalse(needs_write2)             # попадание → писать нечего
            self.assertEqual(len(meta2), 3)

    def test_verbose_exact_query_still_finds_bm25_hit(self) -> None:
        # B1: точный токен (CVE) в МНОГОСЛОВНОМ запросе не должен теряться —
        # entity-aware BM25 ищет по терсовой сущности, а не по всему вопросу.
        with tempfile.TemporaryDirectory() as td:
            store = self._store(Path(td))
            hits = store.hybrid_search(
                query_text="расскажи подробно что это за уязвимость CVE-2024-3094 и опасна ли она",
                query_vector=[1.0, 0.0, 0.0, 0.0],  # вектор целит в c0, не в c2
                top_k=3,
            )
            self.assertIn("c2", [h.chunk_id for h in hits])  # точный CVE поднят BM25-плечом


class TestBm25Query(unittest.TestCase):
    def test_entity_extraction_and_fallback(self) -> None:
        from retrieval import _bm25_query
        self.assertEqual(_bm25_query("составь портрет @johndoe по его сообщениям"), "@johndoe")
        self.assertEqual(_bm25_query("что это за CVE-2024-3094 backdoor"), "CVE-2024-3094")
        self.assertEqual(_bm25_query("проверь хост srv-db-01 срочно"), "srv-db-01")
        self.assertIn("точную фразу", _bm25_query('найди "точную фразу" в логах'))
        # нет сущностей → полный запрос без изменений (обычные факт-вопросы)
        q = "какие пороговые значения указаны в уведомлениях"
        self.assertEqual(_bm25_query(q), q)
        self.assertEqual(_bm25_query(""), "")


if __name__ == "__main__":
    unittest.main()
