# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
"""Раннер eval-харнеса: гоняет реальный пайплайн проекта над набором вопросов.

Живёт в tools/ (не пакуется в .exe), поэтому спокойно опирается на «приватные»
методы стора — если они изменятся, сломается eval-инструмент, а не приложение.

Импорты проекта ленивые (внутри функций): metrics.py/schema.py остаются
самодостаточными и юнит-тестируются без faiss/сервера.

Разделение offline/сервер:
- dual_rankings + fuse-свип — считают vec/bm списки один раз, дальше слияние
  перебирается в памяти БЕЗ сервера (кроме одноразового эмбеда вопроса).
- generate() — требует живой LM Studio (генерация ответа).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent.parent


def _ensure_repo_on_path() -> None:
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))


def open_store_and_notebook(notebook_id: str, notebooks_dir: Path):
    """Открыть реальный блокнот по id, указав корень через NOCTURNE_NOTEBOOKS_DIR.
    Возвращает (notebook, LocalFaissStore)."""
    _ensure_repo_on_path()
    os.environ["NOCTURNE_NOTEBOOKS_DIR"] = str(notebooks_dir)
    from notebook_store import load_notebook
    nb = load_notebook(notebook_id)
    if nb is None:
        raise SystemExit(f"notebook '{notebook_id}' not found under {notebooks_dir}")
    from retrieval import LocalFaissStore
    store = LocalFaissStore(index_dir=nb.index_dir)
    return nb, store


def embed_query(text: str, base_url: str, api_key: str, model: str) -> list[float] | None:
    """Эмбед ОДНОГО вопроса через сервер (task='query'). Кэшируй результат —
    дальше все fusion-ablations идут offline."""
    _ensure_repo_on_path()
    from embeddings import EmbeddingClient
    client = EmbeddingClient(base_url=base_url, api_key=api_key, model=model)
    vecs = client.embed_texts([text], batch_size=1, task="query")
    return list(vecs[0]) if vecs else None


def dual_rankings(store, query_text: str, qvec: list[float] | None, cand: int):
    """Вернуть (vec_ranked, bm_ranked) — два списка (chunk_id, score) ДО слияния.
    Тяжёлая часть (faiss + BM25) считается ОДИН раз; потом fuse_rankings с разными
    параметрами перебирается в памяти. qvec=None → чисто BM25 (offline)."""
    _ensure_repo_on_path()
    import faiss
    import numpy as np
    index, meta, dim, signature = store._load_cached_index_meta()
    if not meta:
        return [], []
    cand = min(cand, len(meta))
    vec_ranked: list[tuple[str, float]] = []
    if qvec and dim is not None and len(qvec) == dim:
        q = np.asarray([qvec], dtype="float32")
        faiss.normalize_L2(q)
        scores, idx = index.search(q, cand)
        vec_ranked = [
            (store._meta_id(i, meta[i]), float(s))
            for s, i in zip(scores[0], idx[0])
            if 0 <= i < len(meta)
        ]
    bm = store._ensure_bm25(meta, signature)
    bm_ranked = [(cid, float(s)) for cid, s in bm.search(query_text, top_k=cand)]
    return vec_ranked, bm_ranked


def fuse_to_ids(vec_ranked, bm_ranked, *, k: int = 60, score_weight: float = 1.0,
                candidate_k: int | None = None, top_k: int = 20,
                min_score_ratio: float = 0.0) -> list[str]:
    """Слить два списка выбранными параметрами → ранжированные chunk_id.
    Всё в памяти, без сервера. candidate_k усекает оба списка перед слиянием."""
    _ensure_repo_on_path()
    from bm25 import fuse_rankings
    vr = vec_ranked[:candidate_k] if candidate_k else vec_ranked
    br = bm_ranked[:candidate_k] if candidate_k else bm_ranked
    rankings = [r for r in (vr, br) if r]
    if not rankings:
        return []
    fused = fuse_rankings(rankings, k=k, top_k=None, score_weight=score_weight)
    if min_score_ratio > 0 and fused:
        best = fused[0][1]
        fused = [(cid, s) for cid, s in fused if s >= best * min_score_ratio]
    return [cid for cid, _ in fused[:top_k]]


async def generate(nb, question: str, *, base_url: str, api_key: str, chat_model: str,
                   embedding_model: str, enhanced: bool = False, deep_mode: str = "off",
                   deep_depth: str = "full", top_k: int = 16,
                   max_answer_tokens: int = 1500) -> dict:
    """Полный grounded-ответ проекта (ТРЕБУЕТ сервер). Возвращает dict для судьи."""
    _ensure_repo_on_path()
    from notebook_chat import answer_question
    res = await answer_question(
        nb, question, base_url=base_url, api_key=api_key, chat_model=chat_model,
        embedding_model=embedding_model, api_mode="native", top_k=top_k,
        max_answer_tokens=max_answer_tokens, enhanced=enhanced,
        deep_mode=deep_mode, deep_depth=deep_depth,
    )
    # Сохраняем и САМИ извлечённые контексты: чтобы судья оценивал grounding по
    # ТОМУ, что видел ответ (а не по чужой референс-выжимке) — иначе широкий gather
    # deep-режима штрафуется как «выдумки». Компактно: источник + срез текста.
    ctx = []
    for c in (res.contexts or []):
        if isinstance(c, dict):
            src = c.get("source_path") or c.get("display") or c.get("source") or ""
            txt = c.get("text") or c.get("quote") or ""
        else:
            src = getattr(c, "source_path", "") or getattr(c, "display", "")
            txt = getattr(c, "text", "") or getattr(c, "quote", "")
        ctx.append({"source": str(src), "text": str(txt)[:500]})
    return {
        "answer": res.answer,
        "refused": bool(res.refused),
        "model": res.model,
        "n_citations": len(res.citations),
        "n_contexts": len(res.contexts),
        "contexts": ctx,
    }
