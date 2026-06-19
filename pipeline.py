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

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Iterable

from embeddings import EmbeddingClient, embedding_prefix_scheme
from file_extractors import (
    ARCHIVE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    TABLE_EXTRACTORS,
    TEXT_EXTRACTORS,
    _SKIP_EXTENSIONS,
)
from models import DocumentChunk, IndexStats, RetrievalHit
from chunking import build_document_chunks, strip_chunk_headers
from retrieval import LocalFaissStore

logger = logging.getLogger("nocturne")

_INDEX_MAX_WORKERS = max(2, min(16, (os.cpu_count() or 4)))

# Один LocalFaissStore на каталог индекса, переиспользуемый между запросами: его
# внутренний кэш (FAISS-индекс + BM25) тогда не перестраивается на каждый вопрос
# чата. Инвалидация — по mtime внутри самого стора (пересборка индекса меняет файлы).
_STORE_CACHE: dict[str, LocalFaissStore] = {}


def _store_for(index_dir: Path) -> LocalFaissStore:
    key = str(Path(index_dir).resolve()).lower()  # Windows: пути регистронезависимы
    store = _STORE_CACHE.get(key)
    if store is None:
        store = LocalFaissStore(index_dir=index_dir)
        _STORE_CACHE[key] = store
    return store
_ALLOWED_SUFFIXES = (
    set(TEXT_EXTRACTORS.keys()) | set(TABLE_EXTRACTORS.keys())
    | set(ARCHIVE_EXTENSIONS) | {".tar.gz"} | IMAGE_EXTENSIONS
) - _SKIP_EXTENSIONS


def _is_hidden_or_system(path: Path) -> bool:
    name = path.name.lower()
    return name.startswith(".") or name in {"__pycache__", "node_modules", ".git", ".idea", ".venv", "venv"}


def _iter_files(paths: Iterable[Path]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        if p.is_file():
            suffix = ".tar.gz" if p.name.lower().endswith(".tar.gz") else p.suffix.lower()
            if suffix in _ALLOWED_SUFFIXES:
                files.append(p)
        elif p.is_dir():
            for root, dirnames, filenames in os.walk(p):
                dirnames[:] = [d for d in dirnames if not _is_hidden_or_system(Path(d))]
                root_path = Path(root)
                for fname in filenames:
                    # Пропускаем служебные файлы инструмента (в т.ч. устаревший
                    # .nocturne_manifest.json), чтобы не засорять корпус и не
                    # дестабилизировать corpus_fingerprint / job_id.
                    if fname.lower().startswith(".nocturne"):
                        continue
                    file_path = root_path / fname
                    suffix = ".tar.gz" if fname.lower().endswith(".tar.gz") else file_path.suffix.lower()
                    if suffix in _ALLOWED_SUFFIXES:
                        files.append(file_path)
    return files


def _to_chunks(
    path: Path,
    chunk_size_tokens: int = 4000,
    overlap_tokens: int = 200,
    root_dir: Path | None = None,
    extract_meta: bool = True,
    vision_describe: Callable[[Path], str] | None = None,
) -> list[DocumentChunk]:
    return build_document_chunks(
        path,
        chunk_size_tokens,
        overlap_tokens,
        root_dir=root_dir,
        extract_meta=extract_meta,
        vision_describe=vision_describe,
    )


def build_index(
    input_paths: list[Path],
    index_dir: Path,
    base_url: str,
    api_key: str,
    embedding_model: str,
    chunk_size_tokens: int,
    on_progress: Callable[[int, int, str], None] | None = None,
    max_workers: int | None = None,
    vision_model: str = "",
) -> IndexStats:
    all_files = _iter_files(input_paths)
    if not all_files:
        raise RuntimeError("No supported files found for indexing")

    # Derive a common root_dir so relative FILE_PATH headers are consistent in the index.
    root_dir: Path | None = None
    if len(input_paths) == 1 and input_paths[0].is_dir():
        root_dir = input_paths[0]
    elif len(input_paths) == 1 and input_paths[0].is_file():
        root_dir = input_paths[0].parent

    # Описатель картинок (vision-модель) — общий, с кешем; None если модель не задана.
    from vision_index import make_image_describer
    describer = make_image_describer(vision_model, base_url, api_key)

    # Картинки описываем vision-моделью последовательно (1 тяжёлый инференс/картинка,
    # чтобы не завалить сервер); текстовые файлы — параллельно.
    pool_workers = 1 if describer else max(1, min(32, max_workers if max_workers is not None else _INDEX_MAX_WORKERS))
    all_chunks: list[DocumentChunk] = []
    processed_files = 0
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        futures = {
            pool.submit(_to_chunks, file_path, chunk_size_tokens, 200, root_dir, True, describer): file_path
            for file_path in all_files
        }
        for fut in as_completed(futures):
            file_chunks = fut.result()
            processed_files += 1
            if on_progress:
                on_progress(processed_files, len(all_files), "index_extract")
            if file_chunks:
                all_chunks.extend(file_chunks)
    if not all_chunks:
        raise RuntimeError("No chunks extracted for indexing")

    # De-duplicate equal chunk payloads to reduce embedding load.
    unique_chunks: list[DocumentChunk] = []
    seen: set[str] = set()
    for chunk in all_chunks:
        key = hashlib.sha256(chunk.text.strip().encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        unique_chunks.append(chunk)

    logger.info(
        "Index build candidates: files=%s chunks=%s unique_chunks=%s workers=%s",
        len(all_files),
        len(all_chunks),
        len(unique_chunks),
        pool_workers,
    )

    emb_client = EmbeddingClient(base_url=base_url, api_key=api_key, model=embedding_model)
    _embp = (lambda done, total: on_progress(done, total, "index_embed")) if on_progress else None
    if on_progress:
        on_progress(0, max(1, len(unique_chunks)), "index_embed")
    # task="document" → nomic-префикс; эмбеддим БЕЗ служебных заголовков (они забивают
    # окно nomic), но в индексе/цитатах остаётся полный текст. Прогресс по батчам.
    _embed_texts = [strip_chunk_headers(c.text) or c.text for c in unique_chunks]
    vectors = emb_client.embed_texts(_embed_texts, batch_size=16, task="document", on_batch=_embp)
    store = LocalFaissStore(index_dir=index_dir)
    return store.build(unique_chunks, vectors, embedding_model=embedding_model,
                       chunk_size_tokens=chunk_size_tokens,
                       prefix_scheme=embedding_prefix_scheme(embedding_model),
                       has_vision=bool(describer))


def add_to_index(
    input_paths: list[Path],
    index_dir: Path,
    base_url: str,
    api_key: str,
    embedding_model: str,
    chunk_size_tokens: int,
    on_progress: Callable[[int, int, str], None] | None = None,
    max_workers: int | None = None,
    vision_model: str = "",
) -> tuple[IndexStats, bool]:
    """Инкрементально дозаписать НОВЫЕ файлы в существующий индекс.

    Возвращает (stats, incremental): ``incremental=False`` означает, что пришлось
    сделать полную пересборку (нет индекса, сменилась embedding-модель, или файлы
    были удалены — из FAISS flat-index удалять нельзя).
    """
    store = LocalFaissStore(index_dir=index_dir)
    all_files = _iter_files(input_paths)
    info = store.info()
    prev_chunk = int(info.get("chunk_size_tokens") or 0)
    prev_prefix = str(info.get("prefix_scheme") or "none")
    prev_vision = bool(info.get("has_vision"))
    _vm = (vision_model or "").strip()
    want_vision = bool(_vm) and not _vm.startswith("(")
    if (
        not store.has_index()
        or str(info.get("embedding_model") or "") != embedding_model
        # Размер чанка сменился ИЛИ старый индекс без этого поля (был собран
        # огромными чанками — поиск не работал) → полная пересборка с миграцией.
        or prev_chunk != int(chunk_size_tokens or 0)
        # Сменилась схема префиксов эмбеддера (старый индекс без nomic-префиксов) →
        # пересборка, иначе пространства документов/запросов не совпадут.
        or prev_prefix != embedding_prefix_scheme(embedding_model)
        # Включили/выключили описание картинок → пересобрать, чтобы (пере)описать.
        or prev_vision != want_vision
    ):
        return build_index(input_paths, index_dir, base_url, api_key, embedding_model,
                           chunk_size_tokens, on_progress, max_workers, vision_model), False

    current = {str(f) for f in all_files}
    indexed = store.indexed_source_paths()
    if indexed - current:  # источник удалён → flat-index не умеет удалять → пересборка
        return build_index(input_paths, index_dir, base_url, api_key, embedding_model,
                           chunk_size_tokens, on_progress, max_workers, vision_model), False

    new_files = [f for f in all_files if str(f) not in indexed]
    if not new_files:
        return IndexStats(
            chunks_total=int(info.get("chunks_total") or 0),
            files_total=int(info.get("files_total") or 0),
            index_dir=index_dir, embedding_model=embedding_model,
        ), True

    root_dir: Path | None = None
    if len(input_paths) == 1 and input_paths[0].is_dir():
        root_dir = input_paths[0]
    elif len(input_paths) == 1 and input_paths[0].is_file():
        root_dir = input_paths[0].parent

    from vision_index import make_image_describer
    describer = make_image_describer(vision_model, base_url, api_key)
    new_chunks: list[DocumentChunk] = []
    pool_workers = 1 if describer else max(1, min(32, max_workers if max_workers is not None else _INDEX_MAX_WORKERS))
    done = 0
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        futures = {pool.submit(_to_chunks, f, chunk_size_tokens, 200, root_dir, True, describer): f
                   for f in new_files}
        for fut in as_completed(futures):
            done += 1
            if on_progress:
                on_progress(done, len(new_files), "index_extract")
            new_chunks.extend(fut.result() or [])

    # Дедуп новых чанков по содержимому + против уже проиндексированного.
    existing_hashes = {
        hashlib.sha256(str(m.get("text") or "").strip().encode("utf-8")).hexdigest()
        for m in store._read_meta()
    }
    unique: list[DocumentChunk] = []
    seen: set[str] = set()
    for ch in new_chunks:
        key = hashlib.sha256(ch.text.strip().encode("utf-8")).hexdigest()
        if key in existing_hashes or key in seen:
            continue
        seen.add(key)
        unique.append(ch)
    if not unique:
        return IndexStats(
            chunks_total=int(info.get("chunks_total") or 0),
            files_total=int(info.get("files_total") or 0),
            index_dir=index_dir, embedding_model=embedding_model,
        ), True

    emb_client = EmbeddingClient(base_url=base_url, api_key=api_key, model=embedding_model)
    _embp = (lambda done, total: on_progress(done, total, "index_embed")) if on_progress else None
    if on_progress:
        on_progress(0, len(unique), "index_embed")
    _embed_texts = [strip_chunk_headers(c.text) or c.text for c in unique]
    vectors = emb_client.embed_texts(_embed_texts, batch_size=16, task="document", on_batch=_embp)
    logger.info("Incremental index add: new_files=%s new_chunks=%s", len(new_files), len(unique))
    return store.append(unique, vectors), True


def query_index(
    question: str,
    index_dir: Path,
    base_url: str,
    api_key: str,
    embedding_model: str,
    top_k: int = 8,
    hybrid: bool = True,
) -> list[RetrievalHit]:
    """Гибридный поиск (вектор + BM25, RRF) по умолчанию; вектор как fallback.

    Если эмбеддинги недоступны (нет модели/сервера), всё равно работает чистый
    BM25 — точный поиск по CVE/хостам/пакетам не зависит от эмбеддера.
    """
    store = _store_for(index_dir)
    qvec: list[float] | None = None
    try:
        emb_client = EmbeddingClient(base_url=base_url, api_key=api_key, model=embedding_model)
        qvecs = emb_client.embed_texts([question], batch_size=1, task="query")  # nomic query-префикс
        qvec = qvecs[0] if qvecs else None
    except Exception as exc:
        logger.warning("Embedding query failed, falling back to BM25 only: %s", exc)
        qvec = None
    if hybrid:
        # min_score_ratio=0.15 убирает «хвост» совсем слабых фрагментов из контекста
        # чата (меньше шума для модели), не задевая релевантные кандидаты.
        hits = store.hybrid_search(question, qvec, top_k=top_k, min_score_ratio=0.15)
        if hits:
            return hits
    if qvec is None:
        return []
    return store.search(qvec, top_k=top_k)
