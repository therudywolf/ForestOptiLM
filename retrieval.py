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
from pathlib import Path

import faiss
import numpy as np

from bm25 import BM25Index, reciprocal_rank_fusion
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
        self._cached_mtime_ns: int = -1
        self._cached_dim: int | None = None
        self._cached_bm25: BM25Index | None = None
        self._cached_bm25_sig: int = -1

    def build(self, chunks: list[DocumentChunk], vectors: list[list[float]],
              embedding_model: str, chunk_size_tokens: int = 0,
              prefix_scheme: str = "none") -> IndexStats:
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
        }
        self.info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
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
        self._cached_mtime_ns = -1
        self._cached_bm25 = None
        return IndexStats(
            chunks_total=chunks_total,
            files_total=files_total,
            index_dir=self.index_dir,
            embedding_model=str(info.get("embedding_model") or ""),
        )

    def search(self, query_vector: list[float], top_k: int = 8) -> list[RetrievalHit]:
        if not self.index_file.exists() or not self.meta_file.exists():
            return []
        index, meta, dim = self._load_cached_index_meta()
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

    def _ensure_bm25(self, meta: list[dict]) -> BM25Index:
        if self._cached_bm25 is not None and self._cached_bm25_sig == self._cached_mtime_ns:
            return self._cached_bm25
        ids = [self._meta_id(i, m) for i, m in enumerate(meta)]
        texts = [str(m.get("text") or "") for m in meta]
        bm = BM25Index().fit(ids, texts)
        self._cached_bm25 = bm
        self._cached_bm25_sig = self._cached_mtime_ns
        return bm

    def hybrid_search(
        self,
        query_text: str,
        query_vector: list[float] | None,
        top_k: int = 8,
        candidate_k: int | None = None,
    ) -> list[RetrievalHit]:
        """FAISS (вектор) + BM25 (лексика) → слияние RRF. Точные CVE/хосты/пакеты
        ловит BM25, семантику — вектор."""
        if not self.index_file.exists() or not self.meta_file.exists():
            return []
        index, meta, dim = self._load_cached_index_meta()
        if not meta:
            return []
        cand = candidate_k or max(top_k * 5, top_k)
        cand = min(cand, len(meta))

        vec_ids: list[str] = []
        if query_vector and dim is not None and len(query_vector) == dim:
            q = np.asarray([query_vector], dtype="float32")
            faiss.normalize_L2(q)
            _scores, idx = index.search(q, cand)
            vec_ids = [
                self._meta_id(i, meta[i]) for i in idx[0] if 0 <= i < len(meta)
            ]

        bm = self._ensure_bm25(meta)
        bm_ids = [cid for cid, _ in bm.search(query_text, top_k=cand)]

        rankings = [r for r in (vec_ids, bm_ids) if r]
        if not rankings:
            return []
        fused = reciprocal_rank_fusion(rankings, top_k=top_k)
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

    def _load_cached_index_meta(self) -> tuple[faiss.Index, list[dict], int | None]:
        idx_mtime = self.index_file.stat().st_mtime_ns
        meta_mtime = self.meta_file.stat().st_mtime_ns
        signature = max(idx_mtime, meta_mtime)
        if (
            self._cached_index is not None
            and self._cached_meta is not None
            and self._cached_mtime_ns == signature
        ):
            return self._cached_index, self._cached_meta, self._cached_dim

        index = faiss.read_index(str(self.index_file))
        meta = self._read_meta()
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
        self._cached_mtime_ns = signature
        self._cached_dim = dim
        return index, meta, dim

    def _read_meta(self) -> list[dict]:
        out: list[dict] = []
        with self.meta_file.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out
