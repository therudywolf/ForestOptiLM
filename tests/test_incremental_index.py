# SPDX-License-Identifier: AGPL-3.0-or-later
"""Incremental notebook index: append new files, rebuild on remove / model change."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pipeline


class _FakeEmb:
    """Deterministic 8-dim embeddings — no server needed; real faiss underneath."""

    def __init__(self, **kw) -> None:
        pass

    def embed_texts(self, texts, batch_size: int = 16, task=None, on_batch=None):
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([b / 255.0 for b in h[:8]])
        if on_batch:
            on_batch(len(texts), len(texts))
        return out


@mock.patch("pipeline.EmbeddingClient", _FakeEmb)
class TestIncrementalIndex(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.index_dir = self.root / "index"

    def _add(self, files, model="emb-1", chunk=2000):
        return pipeline.add_to_index(
            input_paths=files, index_dir=self.index_dir,
            base_url="u", api_key="", embedding_model=model, chunk_size_tokens=chunk,
        )

    def _info(self) -> dict:
        return json.loads((self.index_dir / "index_info.json").read_text(encoding="utf-8"))

    def test_incremental_then_rebuild(self) -> None:
        f1 = self.root / "a.txt"
        f1.write_text("Альфа: уникальное содержимое про релиз и сборки.", encoding="utf-8")
        # first call: no index yet → full build
        stats1, inc1 = self._add([f1])
        self.assertFalse(inc1)
        self.assertGreaterEqual(stats1.files_total, 1)
        n1 = stats1.chunks_total

        # add a second file → incremental append
        f2 = self.root / "b.txt"
        f2.write_text("Бета: совсем другой текст про инциденты и аудит.", encoding="utf-8")
        stats2, inc2 = self._add([f1, f2])
        self.assertTrue(inc2)
        self.assertEqual(stats2.files_total, 2)
        self.assertGreater(stats2.chunks_total, n1)

        # adding the same set again → incremental, nothing new
        stats3, inc3 = self._add([f1, f2])
        self.assertTrue(inc3)
        self.assertEqual(stats3.chunks_total, stats2.chunks_total)

        # removing a file → cannot delete from flat index → full rebuild
        stats4, inc4 = self._add([f1])
        self.assertFalse(inc4)
        self.assertEqual(stats4.files_total, 1)

        # changing embedding model → full rebuild
        stats5, inc5 = self._add([f1], model="emb-2")
        self.assertFalse(inc5)

    def test_chunk_size_stored_in_info(self) -> None:
        # Регрессия: размер чанка должен сохраняться в index_info.json, иначе
        # инкрементальное обновление не заметит смену размера.
        f1 = self.root / "a.txt"
        f1.write_text("Контент про сборки и релизы. " * 5, encoding="utf-8")
        self._add([f1], chunk=512)
        self.assertEqual(int(self._info().get("chunk_size_tokens")), 512)

    def test_rebuild_on_chunk_size_change(self) -> None:
        # Главный фикс: смена размера чанка → ПОЛНАЯ пересборка (а не incremental),
        # иначе остаются старые слишком крупные чанки и поиск не работает.
        f1 = self.root / "a.txt"
        f1.write_text("Альфа контент про подсистемы и сервисы.", encoding="utf-8")
        _s1, inc1 = self._add([f1], chunk=2000)
        self.assertFalse(inc1)  # нет индекса → полная сборка
        _s2, inc2 = self._add([f1], chunk=512)  # те же файлы/модель, другой размер
        self.assertFalse(inc2)  # → пересборка
        self.assertEqual(int(self._info().get("chunk_size_tokens")), 512)
        _s3, inc3 = self._add([f1], chunk=512)  # тот же размер → incremental
        self.assertTrue(inc3)

    def test_rebuild_when_legacy_index_lacks_chunk_size(self) -> None:
        # Старый индекс (без поля chunk_size_tokens) должен мигрировать пересборкой.
        f1 = self.root / "a.txt"
        f1.write_text("Гамма контент.", encoding="utf-8")
        self._add([f1], chunk=512)
        p = self.index_dir / "index_info.json"
        info = self._info()
        info.pop("chunk_size_tokens", None)
        p.write_text(json.dumps(info, ensure_ascii=False), encoding="utf-8")
        _s, inc = self._add([f1], chunk=512)
        self.assertFalse(inc)  # отсутствует поле (prev=0 != 512) → пересборка

    def test_rebuild_when_prefix_scheme_missing(self) -> None:
        # Старый индекс без nomic-префиксов (поле prefix_scheme отсутствует) при
        # nomic-модели должен ПОЛНОСТЬЮ пересобраться, иначе пространства запрос/док
        # не совпадут.
        f1 = self.root / "a.txt"
        f1.write_text("Контент про подсистемы.", encoding="utf-8")
        self._add([f1], model="text-embedding-nomic-embed-text-v1.5")
        info_p = self.index_dir / "index_info.json"
        info = json.loads(info_p.read_text(encoding="utf-8"))
        self.assertEqual(info.get("prefix_scheme"), "nomic-v2")
        info.pop("prefix_scheme", None)  # эмулируем legacy-индекс
        info_p.write_text(json.dumps(info), encoding="utf-8")
        _s, inc = self._add([f1], model="text-embedding-nomic-embed-text-v1.5")
        self.assertFalse(inc)  # полная пересборка-миграция

    def test_rebuild_when_vision_toggled(self) -> None:
        # Включение «описывать картинки» меняет содержимое индекса → нужна
        # полная пересборка, иначе картинки не будут переописаны.
        f1 = self.root / "a.txt"
        f1.write_text("обычный текст без картинок", encoding="utf-8")
        self._add([f1])  # без vision
        info = json.loads((self.index_dir / "index_info.json").read_text(encoding="utf-8"))
        self.assertFalse(info.get("has_vision"))
        _s, inc = pipeline.add_to_index(
            input_paths=[f1], index_dir=self.index_dir, base_url="u", api_key="",
            embedding_model="emb-1", chunk_size_tokens=2000, vision_model="qwen2.5-vl")
        self.assertFalse(inc)  # полная пересборка из-за смены vision-флага
        info2 = json.loads((self.index_dir / "index_info.json").read_text(encoding="utf-8"))
        self.assertTrue(info2.get("has_vision"))

    def test_query_after_incremental(self) -> None:
        f1 = self.root / "a.txt"
        f1.write_text("Альфа уникальный маркер APPLE про релиз.", encoding="utf-8")
        self._add([f1])
        f2 = self.root / "b.txt"
        f2.write_text("Бета уникальный маркер BANANA про аудит.", encoding="utf-8")
        _stats, inc = self._add([f1, f2])
        self.assertTrue(inc)
        hits = pipeline.query_index(
            "BANANA аудит", self.index_dir, "u", "", "emb-1", top_k=5)
        self.assertTrue(any("BANANA" in h.text for h in hits))


if __name__ == "__main__":
    unittest.main()
