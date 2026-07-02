# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ForestOptiLM is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Affero General Public
# License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with ForestOptiLM. If not, see <https://www.gnu.org/licenses/>.
from __future__ import annotations

import json
import logging
import pickle
import threading
from pathlib import Path

import faiss
import numpy as np

from bm25 import BM25Index, fuse_rankings
from models import DocumentChunk, RetrievalHit, IndexStats

logger = logging.getLogger("nocturne")


class LocalFaissStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.index_dir / "chunks.faiss"
        self.meta_file = self.index_dir / "chunks_meta.jsonl"
        self.info_file = self.index_dir / "index_info.json"
        self._cached_index: faiss.Index | None = None
        self._cached_meta: list[dict] | None = None
        # Сигнатура — КОНТЕНТ-зависимая: (max mtime_ns, размер индекса, размер meta).
        # Только mtime недостаточно: на FAT32/exFAT/USB/сетевых дисках (а данные у
        # пользователя лежат рядом с exe) быстрая пересборка может попасть в тот же
        # mtime-бакет → кэш бы не инвалидировался и отдавал устаревший корпус.
        self._cached_sig: tuple = ()
        self._cached_dim: int | None = None
        self._cached_bm25: BM25Index | None = None
        self._cached_bm25_sig: tuple = ()
        # Стор переиспользуется между запросами (один на index_dir) → кэш FAISS/BM25
        # живёт, но к нему могут обращаться из нескольких потоков — защищаем загрузку.
        self._lock = threading.Lock()

    def build(self, chunks: list[DocumentChunk], vectors: list[list[float]],
              embedding_model: str, chunk_size_tokens: int = 0,
              prefix_scheme: str = "none", has_vision: bool = False) -> IndexStats:
        if not chunks:
            raise ValueError("No chunks to index")
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors lengths mismatch")
        dim = len(vectors[0])
        x = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(x)
        index = faiss.IndexFlatIP(dim)
        index.add(x)
        faiss.write_index(index, str(self.index_file))

        with self.meta_file.open("w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(self._meta_line(chunk))

        files_total = len({c.source_path for c in chunks})
        info = {
            "embedding_model": embedding_model,
            "chunks_total": len(chunks),
            "files_total": files_total,
            "dim": dim,
            "chunk_size_tokens": int(chunk_size_tokens or 0),
            "prefix_scheme": str(prefix_scheme or "none"),
            "has_vision": bool(has_vision),
        }
        self.info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        # Прогреть BM25-кэш прямо на сборке: первый поиск пользователя не будет
        # платить за токенизацию всего корпуса (на сотнях тысяч чанков — десятки
        # секунд). Ошибка прогрева не критична — кэш построится на первом запросе.
        try:
            sig = self._index_signature()
            if sig:
                bm = BM25Index().fit([c.chunk_id for c in chunks], [c.text for c in chunks])
                bm.save(self.bm25_cache_file, sig)
                # Прогреть meta-кэш (pickle): холодный старт не парсит сотни тысяч
                # JSON-строк заново. Собираем ровно те же dict-и, что даёт _read_meta().
                self._write_meta_cache(sig, [
                    {"chunk_id": c.chunk_id, "source_path": c.source_path,
                     "text": c.text, "tokens": c.tokens, "metadata": c.metadata}
                    for c in chunks
                ])
        except Exception as exc:  # noqa: BLE001
            logger.warning("bm25/meta: прогрев кэша при сборке не удался — %s", exc)
        return IndexStats(
            chunks_total=len(chunks),
            files_total=files_total,
            index_dir=self.index_dir,
            embedding_model=embedding_model,
        )

    @staticmethod
    def _meta_line(chunk: DocumentChunk) -> str:
        return json.dumps(
            {
                "chunk_id": chunk.chunk_id,
                "source_path": chunk.source_path,
                "text": chunk.text,
                "tokens": chunk.tokens,
                "metadata": chunk.metadata,
            },
            ensure_ascii=False,
        ) + "\n"

    def has_index(self) -> bool:
        return self.index_file.exists() and self.meta_file.exists()

    def info(self) -> dict:
        try:
            return json.loads(self.info_file.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def existing_chunk_ids(self) -> set[str]:
        return {str(m.get("chunk_id")) for m in self._read_meta() if m.get("chunk_id")}

    def indexed_source_paths(self) -> set[str]:
        return {str(m.get("source_path")) for m in self._read_meta() if m.get("source_path")}

    def append(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> IndexStats:
        """Дозаписать новые чанки в существующий FAISS-индекс (без пересборки).

        IndexFlatIP поддерживает инкрементальный ``.add``; meta дописываем построчно.
        """
        if not self.has_index():
            raise RuntimeError("No existing index to append to")
        if len(chunks) != len(vectors):
            raise ValueError("Chunks and vectors lengths mismatch")
        info = self.info()
        if chunks:
            dim = len(vectors[0])
            if info.get("dim") and int(info["dim"]) != dim:
                raise ValueError(
                    f"Embedding dim mismatch (index={info.get('dim')}, new={dim}); rebuild required"
                )
            index = faiss.read_index(str(self.index_file))
            x = np.asarray(vectors, dtype="float32")
            faiss.normalize_L2(x)
            index.add(x)
            faiss.write_index(index, str(self.index_file))
            with self.meta_file.open("a", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(self._meta_line(chunk))
        # пересчитываем счётчики из meta (источник истины)
        all_meta = self._read_meta()
        chunks_total = len(all_meta)
        files_total = len({str(m.get("source_path")) for m in all_meta if m.get("source_path")})
        info.update({"chunks_total": chunks_total, "files_total": files_total})
        self.info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        # сбросить кэш, чтобы следующий поиск перечитал индекс+meta
        self._cached_index = None
        self._cached_meta = None
        self._cached_sig = ()
        self._cached_bm25 = None
        self._cached_bm25_sig = ()
        # Перегреть BM25-кэш под новую сигнатуру: иначе первый поиск после
        # добавления источника заново токенизирует весь корпус.
        try:
            sig = self._index_signature()
            if sig:
                ids = [self._meta_id(i, m) for i, m in enumerate(all_meta)]
                texts = [str(m.get("text") or "") for m in all_meta]
                BM25Index().fit(ids, texts).save(self.bm25_cache_file, sig)
                self._write_meta_cache(sig, all_meta)  # meta уже распарсена — переиспользуем
        except Exception as exc:  # noqa: BLE001
            logger.warning("bm25/meta: перегрев кэша при append не удался — %s", exc)
        return IndexStats(
            chunks_total=chunks_total,
            files_total=files_total,
            index_dir=self.index_dir,
            embedding_model=str(info.get("embedding_model") or ""),
        )

    def search(self, query_vector: list[float], top_k: int = 8) -> list[RetrievalHit]:
        if not self.index_file.exists() or not self.meta_file.exists():
            return []
        index, meta, dim, _sig = self._load_cached_index_meta()
        if index is None or not meta:  # индекс исчез во время запроса → пусто, не падаем
            return []
        if dim is not None and len(query_vector) != dim:
            raise ValueError(
                f"Query vector dim mismatch: got {len(query_vector)}, expected {dim}. "
                "Проверьте embedding model при build/query."
            )
        q = np.asarray([query_vector], dtype="float32")
        faiss.normalize_L2(q)
        scores, idx = index.search(q, top_k)
        hits: list[RetrievalHit] = []
        for score, i in zip(scores[0], idx[0]):
            if i < 0 or i >= len(meta):
                continue
            m = meta[i]
            hits.append(
                RetrievalHit(
                    chunk_id=m["chunk_id"],
                    score=float(score),
                    source_path=m["source_path"],
                    text=m["text"],
                    metadata=m.get("metadata", {}),
                )
            )
        return hits

    @staticmethod
    def _meta_id(pos: int, m: dict) -> str:
        return str(m.get("chunk_id") or pos)

    @property
    def bm25_cache_file(self) -> Path:
        return self.index_dir / "bm25_cache.pkl"

    def _ensure_bm25(self, meta: list[dict], signature: tuple) -> BM25Index:
        # Штампуем BM25 ИМЕННО той сигнатурой, из meta которой он построен (а не
        # текущим self._cached_sig — тот мог уйти вперёд из-за параллельного reload,
        # и тогда BM25 от старого корпуса выдавался бы под новой сигнатурой).
        with self._lock:
            if self._cached_bm25 is not None and self._cached_bm25_sig == signature:
                return self._cached_bm25
            # Дисковый кэш: на больших корпусах fit (токенизация всего корпуса)
            # занимает десятки секунд — кэш срезает холодный старт до секунд.
            bm = BM25Index.load(self.bm25_cache_file, signature)
            if bm is None or len(bm.ids) != len(meta):
                ids = [self._meta_id(i, m) for i, m in enumerate(meta)]
                texts = [str(m.get("text") or "") for m in meta]
                bm = BM25Index().fit(ids, texts)
                bm.save(self.bm25_cache_file, signature)
            self._cached_bm25 = bm
            self._cached_bm25_sig = signature
            return bm

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float] | None,
        top_k: int = 8,
        candidate_k: int | None = None,
        min_score_ratio: float = 0.0,
    ) -> list[RetrievalHit]:
        """FAISS (вектор) + BM25 (лексика) → score-aware слияние (fuse_rankings).
        Точные CVE/хосты/пакеты ловит BM25, семантику — вектор.

        Слияние учитывает не только ранги (RRF), но и величину скоров: на больших
        корпусах (сотни тысяч чанков) списки вектора и BM25 почти не пересекаются,
        и чистый RRF вырождался в связки 1/(k+1) со случайным порядком.

        ``min_score_ratio`` (0 = выкл) отбрасывает «хвост» — фрагменты со слитым
        скором ниже ``min_score_ratio × лучший_скор``. Консервативно (0.15)
        убирает явный шум, не задевая релевантные кандидаты — recall важнее для
        grounded-чата, поэтому по умолчанию фильтр выключен.
        """
        if not self.index_file.exists() or not self.meta_file.exists():
            return []
        index, meta, dim, signature = self._load_cached_index_meta()
        if not meta:
            return []
        # Глубокий пул кандидатов: на большом корпусе пересечения списков (главный
        # сигнал слияния) при мелком пуле практически не случаются.
        cand = candidate_k or max(top_k * 5, 50)
        cand = min(cand, len(meta))

        vec_ranked: list[tuple[str, float]] = []
        if query_vector and dim is not None and len(query_vector) == dim:
            q = np.asarray([query_vector], dtype="float32")
            faiss.normalize_L2(q)
            scores, idx = index.search(q, cand)
            vec_ranked = [
                (self._meta_id(i, meta[i]), float(s))
                for s, i in zip(scores[0], idx[0])
                if 0 <= i < len(meta)
            ]

        bm = self._ensure_bm25(meta, signature)
        bm_ranked = [(cid, float(s)) for cid, s in bm.search(query_text, top_k=cand)]

        rankings = [r for r in (vec_ranked, bm_ranked) if r]
        if not rankings:
            return []
        fused = fuse_rankings(rankings, top_k=top_k)
        if fused and min_score_ratio > 0.0:
            floor = fused[0][1] * min_score_ratio
            fused = [(cid, s) for cid, s in fused if s >= floor]
        by_id = {self._meta_id(pos, m): m for pos, m in enumerate(meta)}
        hits: list[RetrievalHit] = []
        for cid, score in fused:
            m = by_id.get(cid)
            if not m:
                continue
            hits.append(
                RetrievalHit(
                    chunk_id=m["chunk_id"],
                    score=float(score),
                    source_path=m["source_path"],
                    text=m["text"],
                    metadata=m.get("metadata", {}),
                )
            )
        return hits

    def _index_signature(self) -> tuple | None:
        """Контент-зависимая сигнатура индекса: (max mtime_ns, размер .faiss, размер meta).
        Размеры файлов ловят пересборку даже там, где mtime не изменился (грубая
        гранулярность времени на FAT32/exFAT/сетевых дисках). None — файлов уже нет
        (блокнот удалён прямо во время запроса в соседнем потоке)."""
        try:
            ix = self.index_file.stat()
            mt = self.meta_file.stat()
        except OSError:
            return None
        return (max(ix.st_mtime_ns, mt.st_mtime_ns), ix.st_size, mt.st_size)

    def _load_cached_index_meta(self) -> tuple[faiss.Index | None, list[dict], int | None, tuple]:
        signature = self._index_signature()
        if signature is None:
            return None, [], None, ()  # индекс исчез между .exists() и stat() → пусто
        with self._lock:
            if (
                self._cached_index is not None
                and self._cached_meta is not None
                and self._cached_sig == signature
            ):
                return self._cached_index, self._cached_meta, self._cached_dim, signature

            try:
                index = faiss.read_index(str(self.index_file))
                meta = self._read_meta_cached(signature)  # pickle-кэш ускоряет холодный старт
            except OSError:
                return None, [], None, ()  # каталог подменили/удалили после stat()
            dim = None
            try:
                info = json.loads(self.info_file.read_text(encoding="utf-8"))
                dim_val = info.get("dim")
                if isinstance(dim_val, int) and dim_val > 0:
                    dim = dim_val
            except Exception:
                dim = None
            self._cached_index = index
            self._cached_meta = meta
            self._cached_sig = signature
            self._cached_dim = dim
        return index, meta, dim, signature

    @property
    def meta_cache_file(self) -> Path:
        return self.index_dir / "meta_cache.pkl"

    def _read_meta_cached(self, signature: tuple) -> list[dict]:
        """Meta через pickle-кэш: парс 455k JSON-строк ~7с → загрузка pickle ~2-3с.
        Валидируется сигнатурой корпуса; несовпадение/сбой → перечитать JSONL и
        перезаписать кэш. Источник истины остаётся chunks_meta.jsonl."""
        p = self.meta_cache_file
        try:
            with p.open("rb") as f:
                payload = pickle.load(f)
            if isinstance(payload, dict) and tuple(payload.get("sig") or ()) == tuple(signature):
                meta = payload.get("meta")
                if isinstance(meta, list):
                    return meta
        except FileNotFoundError:
            pass
        except Exception as exc:  # noqa: BLE001
            logger.warning("meta cache %s не читается (%s) — перечитываю JSONL", p, exc)
        meta = self._read_meta()
        self._write_meta_cache(signature, meta)
        return meta

    def _write_meta_cache(self, signature: tuple, meta: list[dict]) -> None:
        """Атомарно записать pickle-кэш meta (tmp+replace). Сбой не критичен —
        кэш перестроится на первом запросе."""
        p = self.meta_cache_file
        try:
            tmp = p.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump({"sig": tuple(signature), "meta": meta}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning("meta cache: не удалось сохранить %s — %s", p, exc)

    def _read_meta(self) -> list[dict]:
        out: list[dict] = []
        with self.meta_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out
