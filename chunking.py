# SPDX-License-Identifier: AGPL-3.0-or-later
"""Единый API чанкинга для Map-Reduce и RAG."""
from __future__ import annotations

from pathlib import Path

from file_extractors import ParseError, extract_content, extract_file_metadata
from models import DocumentChunk
from parser import chunk_text_for_map_file, count_tokens


def build_document_chunks(
    path: Path,
    chunk_size_tokens: int,
    overlap_tokens: int = 200,
    root_dir: Path | None = None,
    *,
    extract_meta: bool = True,
) -> list[DocumentChunk]:
    """
    Извлечь файл и вернуть DocumentChunk (текст, vision, table).
    Используется pipeline.build_index и gui folder batch.
    """
    import hashlib

    import pandas as pd

    path = Path(path)
    try:
        kind, content = extract_content(path)
    except ParseError:
        return []
    except PermissionError:
        return []
    except Exception:
        return []

    if root_dir is not None:
        try:
            display_path = str(path.relative_to(root_dir)).replace("\\", "/")
        except ValueError:
            display_path = path.name
    else:
        display_path = path.name

    file_meta: dict[str, str] | None = None
    if extract_meta:
        try:
            file_meta = extract_file_metadata(path, root_dir=root_dir)
        except Exception:
            file_meta = None

    # Record-aware: структурированные отчёты (JSON/JSONL/XML) → записи как чанки,
    # без хардкода форматов. Делаем до общего text/table-роутинга.
    try:
        from record_chunking import build_record_chunks, record_aware_enabled

        if record_aware_enabled():
            rec_chunks = build_record_chunks(
                path, chunk_size_tokens, root_dir=root_dir, file_meta=file_meta,
            )
            if rec_chunks:
                out: list[DocumentChunk] = []
                for idx, c in enumerate(rec_chunks):
                    cid = hashlib.sha256(f"{path}:rec:{idx}".encode()).hexdigest()[:24]
                    rmeta: dict = {"chunk_index": idx, "kind": "record"}
                    if file_meta:
                        rmeta.update(file_meta)
                    out.append(
                        DocumentChunk(
                            chunk_id=cid, source_path=str(path), text=c,
                            tokens=count_tokens(c), metadata=rmeta,
                        )
                    )
                return out
    except Exception:
        pass

    chunks: list[DocumentChunk] = []
    if kind == "vision":
        img_path = Path(str(content))
        if not img_path.is_file():
            return []
        meta_line = _meta_header_line(file_meta)
        t = (
            f"[FILE_PATH: {display_path}]\n{meta_line}"
            f"[Файл: {display_path}]\n[VISION_FILE: {img_path.resolve()}]\n"
        )
        cid = hashlib.sha256(f"{path}:vision".encode()).hexdigest()[:24]
        meta: dict = {"kind": "vision", "chunk_index": 0}
        if file_meta:
            meta.update(file_meta)
        chunks.append(
            DocumentChunk(chunk_id=cid, source_path=str(path), text=t, tokens=count_tokens(t), metadata=meta)
        )
    elif kind == "text":
        text = str(content).strip()
        if not text:
            return []
        for idx, c in enumerate(
            chunk_text_for_map_file(
                path, text, chunk_size_tokens, overlap_tokens, root_dir=root_dir, file_meta=file_meta,
            )
        ):
            cid = hashlib.sha256(f"{path}:{idx}:{c[:200]}".encode()).hexdigest()[:24]
            cmeta: dict = {"chunk_index": idx}
            if file_meta:
                cmeta.update(file_meta)
            chunks.append(
                DocumentChunk(
                    chunk_id=cid, source_path=str(path), text=c, tokens=count_tokens(c), metadata=cmeta,
                )
            )
    else:
        if not isinstance(content, pd.DataFrame) or content.empty:
            return []
        text = content.to_csv(index=False)
        cid = hashlib.sha256(f"{path}:table".encode()).hexdigest()[:24]
        tmeta: dict = {"kind": "table"}
        if file_meta:
            tmeta.update(file_meta)
        chunks.append(
            DocumentChunk(
                chunk_id=cid, source_path=str(path), text=text, tokens=count_tokens(text), metadata=tmeta,
            )
        )
    return chunks


def chunks_to_map_strings(chunks: list[DocumentChunk]) -> list[str]:
    return [c.text for c in chunks]


def _meta_header_line(file_meta: dict[str, str] | None) -> str:
    if not file_meta:
        return ""
    parts: list[str] = []
    if file_meta.get("title"):
        parts.append(f"[FILE_TITLE: {file_meta['title']}]")
    if file_meta.get("labels"):
        parts.append(f"[FILE_LABELS: {file_meta['labels']}]")
    if file_meta.get("format"):
        parts.append(f"[FILE_FORMAT: {file_meta['format']}]")
    return "".join(parts) + "\n" if parts else ""
