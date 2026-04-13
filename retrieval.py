from __future__ import annotations

import json
import logging
from pathlib import Path

import faiss
import numpy as np

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

    def build(self, chunks: list[DocumentChunk], vectors: list[list[float]], embedding_model: str) -> IndexStats:
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
                f.write(
                    json.dumps(
                        {
                            "chunk_id": chunk.chunk_id,
                            "source_path": chunk.source_path,
                            "text": chunk.text,
                            "tokens": chunk.tokens,
                            "metadata": chunk.metadata,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        files_total = len({c.source_path for c in chunks})
        info = {
            "embedding_model": embedding_model,
            "chunks_total": len(chunks),
            "files_total": files_total,
            "dim": dim,
        }
        self.info_file.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
        return IndexStats(
            chunks_total=len(chunks),
            files_total=files_total,
            index_dir=self.index_dir,
            embedding_model=embedding_model,
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
