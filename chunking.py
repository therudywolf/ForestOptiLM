# SPDX-License-Identifier: AGPL-3.0-or-later
"""Единый API чанкинга для Map-Reduce и RAG."""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from file_extractors import ParseError, extract_content, extract_file_metadata
from models import DocumentChunk
from parser import chunk_text_for_map_file, count_tokens

_HEADER_RE = re.compile(
    r"^\s*\[(FILE_PATH|FILE_TITLE|FILE_LABELS|FILE_FORMAT|FILE_PART|CHUNK_INDEX|"
    r"VISION_FILE|SOURCE_URL|SOURCE_TITLE|Файл)\b")


def strip_chunk_headers(text: str) -> str:
    """Убрать служебные [FILE_PATH]/[FILE_TITLE]/[Файл:…]-строки — для ЭМБЕДДИНГА.

    Сами чанки хранят заголовки (нужны для цитат/привязки), но эмбеддить их вредно:
    у nomic окно ~2048 ток., и одинаковые заголовки в начале каждого чанка делают
    векторы неразличимыми. В индекс/цитаты идёт полный текст, в эмбеддер — очищенный.
    """
    lines = [ln for ln in (text or "").splitlines() if not _HEADER_RE.match(ln)]
    return "\n".join(lines).strip()


def _raster_ext(blob: bytes) -> str:
    """Расширение по магическим байтам — только растровые (vision их понимает).

    EMF/WMF (вектор, частый формат диаграмм в docx) пропускаем — vision-модель не
    обработает их без растеризации.
    """
    if blob[:4] == b"\x89PNG":
        return ".png"
    if blob[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if blob[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    if blob[:2] == b"BM":
        return ".bmp"
    if blob[:4] == b"RIFF" and blob[8:12] == b"WEBP":
        return ".webp"
    return ""


def _describe_doc_images(path: Path, vision_describe: Callable[[Path], str] | None) -> str:
    """Описать картинки, ВСТРОЕННЫЕ в .docx (диаграммы/схемы), vision-моделью."""
    if vision_describe is None or path.suffix.lower() != ".docx":
        return ""
    try:
        from file_extractors import extract_docx_images
        blobs = extract_docx_images(path)
    except Exception:
        return ""
    import tempfile
    descs: list[str] = []
    for i, blob in enumerate(blobs):
        ext = _raster_ext(blob or b"")
        if not ext:
            continue  # вектор/неизвестный формат — пропускаем
        tmp: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(blob)
                tmp = Path(f.name)
            d = vision_describe(tmp)
            if d and d.strip():
                descs.append(f"[Изображение {i + 1} из документа]\n{d.strip()}")
        except Exception:
            pass
        finally:
            if tmp is not None:
                try:
                    tmp.unlink()
                except Exception:
                    pass
    return ("\n\n[Изображения из документа]\n" + "\n\n".join(descs)) if descs else ""


def build_document_chunks(
    path: Path,
    chunk_size_tokens: int,
    overlap_tokens: int = 200,
    root_dir: Path | None = None,
    *,
    extract_meta: bool = True,
    vision_describe: Callable[[Path], str] | None = None,
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
        # Описать картинку vision-моделью → её содержимое становится искомым
        # (схемы/таблицы/диаграммы, а не только имя файла).
        if vision_describe is not None:
            try:
                desc = vision_describe(img_path)
            except Exception:
                desc = ""
            if desc:
                t += "\n" + desc
        cid = hashlib.sha256(f"{path}:vision".encode()).hexdigest()[:24]
        meta: dict = {"kind": "vision", "chunk_index": 0}
        if file_meta:
            meta.update(file_meta)
        chunks.append(
            DocumentChunk(chunk_id=cid, source_path=str(path), text=t, tokens=count_tokens(t), metadata=meta)
        )
    elif kind == "text":
        text = str(content).strip()
        # Картинки, встроенные в документ (диаграммы/схемы в .docx), описываем
        # vision-моделью и дописываем к тексту → их содержимое тоже ищется.
        img_block = _describe_doc_images(path, vision_describe)
        if img_block:
            text = (text + img_block).strip()
        if not text:
            return []
        # Page breaks (\f, from PDF extraction) → page numbers for citations.
        page_breaks = [m.start() for m in re.finditer("\f", text)]
        marker = f"[Файл: {display_path}]\n"
        search_from = 0
        for idx, c in enumerate(
            chunk_text_for_map_file(
                path, text, chunk_size_tokens, overlap_tokens, root_dir=root_dir, file_meta=file_meta,
            )
        ):
            cmeta: dict = {"chunk_index": idx}
            if file_meta:
                cmeta.update(file_meta)
            # Locate this chunk in the source to derive its line/page.
            bs = c.split(marker, 1)[-1].strip()
            start_probe = bs[:48]
            loc = text.find(start_probe, search_from) if start_probe else -1
            if loc < 0 and start_probe:
                loc = text.find(start_probe)
            if loc >= 0:
                cmeta["line_start"] = text.count("\n", 0, loc) + 1
                search_from = loc + 1
            if page_breaks and bs:
                # Page from the chunk MIDDLE — the dominant page, unskewed by the
                # overlap tail that a boundary chunk repeats from the previous page.
                off = min(len(bs) // 2, max(0, len(bs) - 48))
                mid_probe = bs[off:off + 48]
                mloc = text.find(mid_probe, max(0, search_from - 4000)) if mid_probe else -1
                if mloc < 0 and mid_probe:
                    mloc = text.find(mid_probe)
                anchor = mloc if mloc >= 0 else loc
                if anchor >= 0:
                    cmeta["page"] = sum(1 for b in page_breaks if b < anchor) + 1
            if page_breaks:
                c = c.replace("\f", "\n")  # don't leak the page-break marker into chunks
            cid = hashlib.sha256(f"{path}:{idx}:{c[:200]}".encode()).hexdigest()[:24]
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
