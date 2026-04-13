from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import logging
import os
from pathlib import Path
from typing import Callable, Iterable

from embeddings import EmbeddingClient
from file_extractors import (
    ARCHIVE_EXTENSIONS,
    IMAGE_EXTENSIONS,
    ParseError,
    TABLE_EXTRACTORS,
    TEXT_EXTRACTORS,
    _SKIP_EXTENSIONS,
    extract_content,
    extract_file_metadata,
)
from models import DocumentChunk, IndexStats, RetrievalHit
from parser import chunk_text_for_map_file, count_tokens
from retrieval import LocalFaissStore

logger = logging.getLogger("nocturne")

_INDEX_MAX_WORKERS = max(2, min(16, (os.cpu_count() or 4)))
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
) -> list[DocumentChunk]:
    try:
        kind, content = extract_content(path)
    except ParseError:
        return []
    except PermissionError as e:
        logger.info("skip locked file %s: %s", path, e)
        return []
    except Exception as e:
        logger.warning("extract failed %s: %s", path, e)
        return []

    # Compute display path: relative to root_dir when available.
    if root_dir is not None:
        try:
            display_path = str(path.relative_to(root_dir)).replace("\\", "/")
        except ValueError:
            display_path = path.name
    else:
        display_path = path.name

    # Fast metadata sampling (title, labels, format) for chunk header enrichment.
    file_meta: dict[str, str] | None = None
    if extract_meta:
        try:
            file_meta = extract_file_metadata(path, root_dir=root_dir)
        except Exception as exc:
            logger.debug("extract_file_metadata failed for %s: %s", path, exc)

    chunks: list[DocumentChunk] = []
    if kind == "vision":
        img_path = Path(str(content))
        if not img_path.is_file():
            return []
        meta_line = ""
        if file_meta:
            parts: list[str] = []
            if file_meta.get("title"):
                parts.append(f"[FILE_TITLE: {file_meta['title']}]")
            if file_meta.get("labels"):
                parts.append(f"[FILE_LABELS: {file_meta['labels']}]")
            if file_meta.get("format"):
                parts.append(f"[FILE_FORMAT: {file_meta['format']}]")
            if parts:
                meta_line = "".join(parts) + "\n"
        t = (
            f"[FILE_PATH: {display_path}]\n"
            f"{meta_line}"
            f"[Файл: {display_path}]\n"
            f"[VISION_FILE: {img_path.resolve()}]\n"
        )
        cid = hashlib.sha256(f"{path}:vision".encode("utf-8")).hexdigest()[:24]
        base_meta: dict = {"kind": "vision", "chunk_index": 0}
        if file_meta:
            base_meta.update(file_meta)
        chunks.append(
            DocumentChunk(
                chunk_id=cid,
                source_path=str(path),
                text=t,
                tokens=count_tokens(t),
                metadata=base_meta,
            )
        )
    elif kind == "text":
        text = str(content).strip()
        if not text:
            return []
        chunk_texts = chunk_text_for_map_file(
            path, text, chunk_size_tokens, overlap_tokens,
            root_dir=root_dir, file_meta=file_meta,
        )
        for idx, c in enumerate(chunk_texts):
            cid = hashlib.sha256(f"{path}:{idx}:{c[:200]}".encode("utf-8")).hexdigest()[:24]
            chunk_meta: dict = {"chunk_index": idx}
            if file_meta:
                chunk_meta.update(file_meta)
            chunks.append(
                DocumentChunk(
                    chunk_id=cid,
                    source_path=str(path),
                    text=c,
                    tokens=count_tokens(c),
                    metadata=chunk_meta,
                )
            )
    else:
        import pandas as pd

        if not isinstance(content, pd.DataFrame) or content.empty:
            return []
        text = content.to_csv(index=False)
        cid = hashlib.sha256(f"{path}:table".encode("utf-8")).hexdigest()[:24]
        table_meta: dict = {"kind": "table"}
        if file_meta:
            table_meta.update(file_meta)
        chunks.append(
            DocumentChunk(
                chunk_id=cid,
                source_path=str(path),
                text=text,
                tokens=count_tokens(text),
                metadata=table_meta,
            )
        )
    return chunks


def build_index(
    input_paths: list[Path],
    index_dir: Path,
    base_url: str,
    api_key: str,
    embedding_model: str,
    chunk_size_tokens: int,
    on_progress: Callable[[int, int, str], None] | None = None,
    max_workers: int | None = None,
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

    pool_workers = max(1, min(32, max_workers if max_workers is not None else _INDEX_MAX_WORKERS))
    all_chunks: list[DocumentChunk] = []
    processed_files = 0
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        futures = {
            pool.submit(_to_chunks, file_path, chunk_size_tokens, 200, root_dir, True): file_path
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
    if on_progress:
        on_progress(0, max(1, len(unique_chunks)), "index_embed")
    vectors = emb_client.embed_texts([c.text for c in unique_chunks], batch_size=16)
    if on_progress:
        on_progress(max(1, len(unique_chunks)), max(1, len(unique_chunks)), "index_embed")
    store = LocalFaissStore(index_dir=index_dir)
    return store.build(unique_chunks, vectors, embedding_model=embedding_model)


def query_index(
    question: str,
    index_dir: Path,
    base_url: str,
    api_key: str,
    embedding_model: str,
    top_k: int = 8,
) -> list[RetrievalHit]:
    emb_client = EmbeddingClient(base_url=base_url, api_key=api_key, model=embedding_model)
    qvecs = emb_client.embed_texts([question], batch_size=1)
    if not qvecs:
        return []
    store = LocalFaissStore(index_dir=index_dir)
    return store.search(qvecs[0], top_k=top_k)
