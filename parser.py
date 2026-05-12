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
"""
Nocturne Data Forge — парсинг файлов и разбиение на чанки/батчи.
Использует file_extractors для извлечения контента; семантический чанкинг и батчи с заголовками.
"""
from __future__ import annotations

import hashlib
import os
import re
import logging
from pathlib import Path
from typing import Literal

import pandas as pd
import tiktoken

from file_extractors import ParseError, extract_content

logger = logging.getLogger("nocturne")

# Кэш кодировки tiktoken
_encoding: tiktoken.Encoding | None = None

# Батч таблицы: (заголовки, список строк как dict)
TableBatch = tuple[list[str], list[dict[str, object]]]


def get_encoding() -> tiktoken.Encoding:
    """Загрузить и закэшировать tiktoken cl100k_base."""
    global _encoding
    if _encoding is None:
        _encoding = tiktoken.get_encoding("cl100k_base")
    return _encoding


def count_tokens(text: str) -> int:
    """Подсчёт токенов через tiktoken (cl100k_base)."""
    if not text or not text.strip():
        return 0
    return len(get_encoding().encode(text))


def compute_dynamic_chunk_size(
    max_model_tokens: int,
    system_prompt: str,
    user_query: str,
    response_reserve: int = 4096,
    max_chunk_tokens_cap: int | None = None,
) -> int:
    """
    Вычислить размер чанка: max - system - query - reserve.
    Нижняя граница 500, верхняя — max_model_tokens.
    max_chunk_tokens_cap — верхний предел (guardrail): слишком крупные чанки ухудшают MAP.
    Переменная окружения NOCTURNE_MAX_CHUNK_TOKENS (0 = без ограничения).
    """
    cap_env = os.getenv("NOCTURNE_MAX_CHUNK_TOKENS", "6000").strip()
    env_cap: int | None = None
    if cap_env and cap_env != "0":
        try:
            env_cap = int(cap_env)
        except ValueError:
            env_cap = 6000
    effective_cap = max_chunk_tokens_cap if max_chunk_tokens_cap is not None else env_cap

    system_tokens = count_tokens(system_prompt)
    query_tokens = count_tokens(user_query)
    available = max_model_tokens - system_tokens - query_tokens - response_reserve
    chunk_size = max(500, min(available, max_model_tokens))
    if effective_cap is not None and effective_cap > 0:
        chunk_size = min(chunk_size, effective_cap)
    logger.info(
        "dynamic_chunk_size: max=%s system=%s query=%s reserve=%s cap=%s -> %s",
        max_model_tokens, system_tokens, query_tokens, response_reserve, effective_cap, chunk_size,
    )
    return chunk_size


def _segment_paragraphs_and_sentences(text: str, chunk_size_tokens: int = 8000) -> list[str]:
    """Разбить текст на сегменты по абзацам и предложениям."""
    # Use chunk_size_tokens as the paragraph-size threshold; cap at 8000 to avoid excessive segments.
    para_limit = min(8000, chunk_size_tokens)
    segments: list[str] = []
    paragraphs = re.split(r"\n\s*\n", text)
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if count_tokens(para) <= para_limit:
            segments.append(para)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", para)
        current: list[str] = []
        current_tokens = 0
        for sent in sentences:
            t = count_tokens(sent)
            if current_tokens + t > para_limit and current:
                segments.append(" ".join(current))
                current = []
                current_tokens = 0
            current.append(sent)
            current_tokens += t
        if current:
            segments.append(" ".join(current))
    return segments


def chunk_text_semantic(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int = 200,
) -> list[str]:
    """
    Семантический чанкинг: границы по абзацам и предложениям, не превышая chunk_size_tokens.
    Overlap — последний абзац/сегмент предыдущего чанка повторяется в начале следующего.
    """
    if not text or not text.strip():
        return []
    enc = get_encoding()
    segments = _segment_paragraphs_and_sentences(text, chunk_size_tokens=chunk_size_tokens)
    if not segments:
        tokens = enc.encode(text)
        if not tokens:
            return []
        step = max(1, chunk_size_tokens - overlap_tokens)
        out: list[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + chunk_size_tokens, len(tokens))
            out.append(enc.decode(tokens[start:end]))
            if end >= len(tokens):
                break
            start = start + step
        return out

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0
    for seg in segments:
        seg_tokens = count_tokens(seg)
        if seg_tokens > chunk_size_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current = []
                current_tokens = 0
            enc_seg = enc.encode(seg)
            step = max(1, chunk_size_tokens - overlap_tokens)
            for start in range(0, len(enc_seg), step):
                end = min(start + chunk_size_tokens, len(enc_seg))
                chunks.append(enc.decode(enc_seg[start:end]))
                if end >= len(enc_seg):
                    break
            continue
        if current_tokens + seg_tokens > chunk_size_tokens and current:
            chunks.append("\n\n".join(current))
            overlap_seg = current[-1] if current else ""
            current = [overlap_seg] if overlap_seg and overlap_tokens > 0 else []
            current_tokens = count_tokens(overlap_seg)
        current.append(seg)
        current_tokens += seg_tokens
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _chunk_text_by_tokens(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Резерв: разбиение по токенам без учёта границ (для очень длинных сегментов уже внутри chunk_text_semantic)."""
    enc = get_encoding()
    tokens = enc.encode(text)
    if not tokens:
        return []
    step = max(1, chunk_size_tokens - overlap_tokens)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size_tokens, len(tokens))
        chunks.append(enc.decode(tokens[start:end]))
        if end >= len(tokens):
            break
        start = start + step
    return chunks


def chunk_text_for_map_file(
    path: Path,
    raw_text: str,
    chunk_size_tokens: int,
    overlap_tokens: int = 200,
    root_dir: Path | None = None,
    file_meta: dict[str, str] | None = None,
) -> list[str]:
    """
    Чанки для MAP с идентификатором файла и при сверхдлинном тексте — частями FILE_PART i/n.

    FILE_PATH показывает путь, относительный root_dir (или просто имя файла, если root_dir не задан).
    file_meta: опциональный словарь с ключами title, labels, format для дополнительных заголовков.
    Модель должна использовать FILE_PATH в evidence_refs.file.
    """
    path = Path(path)
    # Relative path for human-readable evidence refs.
    if root_dir is not None:
        try:
            display_path = str(path.relative_to(root_dir)).replace("\\", "/")
        except ValueError:
            display_path = path.name
    else:
        display_path = path.name

    # Build optional metadata header line
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

    total_t = count_tokens(raw_text)
    try:
        mega_th = int(os.getenv("NOCTURNE_MEGA_FILE_TOKEN_THRESHOLD", "80000"))
    except ValueError:
        mega_th = 80000
    try:
        part_factor = max(4, int(os.getenv("NOCTURNE_MEGA_PART_FACTOR", "6")))
    except ValueError:
        part_factor = 6

    if total_t <= mega_th:
        chunks = chunk_text_semantic(raw_text, chunk_size_tokens, overlap_tokens)
        return [
            f"[FILE_PATH: {display_path}][FILE_PART: 1/1][CHUNK_INDEX: {i + 1}]\n"
            f"{meta_line}"
            f"[Файл: {display_path}]\n{c}"
            for i, c in enumerate(chunks)
        ]

    part_size = min(
        chunk_size_tokens * part_factor,
        max(chunk_size_tokens * 2, total_t // 8 + 1),
    )
    part_texts = chunk_text_semantic(raw_text, part_size, overlap_tokens)
    n_parts = len(part_texts)
    out: list[str] = []
    for pi, part in enumerate(part_texts):
        sub = chunk_text_semantic(part, chunk_size_tokens, overlap_tokens)
        for c in sub:
            idx = len(out) + 1
            out.append(
                f"[FILE_PATH: {display_path}][FILE_PART: {pi + 1}/{n_parts}][CHUNK_INDEX: {idx}]\n"
                f"{meta_line}"
                f"[Файл: {display_path}]\n{c}"
            )
    logger.info(
        "Mega-file chunking: %s tokens=%s parts=%s chunks=%s",
        display_path, total_t, n_parts, len(out),
    )
    return out


def _estimate_tokens_per_row(df: pd.DataFrame, sample_rows: int = 50) -> float:
    """Оценка токенов на строку."""
    header_len = sum(len(str(c)) for c in df.columns)
    n = min(sample_rows, len(df))
    if n == 0:
        return max(1.0, header_len / 3)
    total_chars = 0
    for i in range(n):
        row = df.iloc[i]
        total_chars += header_len + sum(len(str(v)) for v in row)
    return max(1.0, (total_chars / n) / 3)


def parse_file(
    path: Path,
    dynamic_chunk_size: int,
    overlap_tokens: int = 200,
    root_dir: Path | None = None,
) -> tuple[
    Literal["text", "table", "vision"],
    list[str] | list[TableBatch],
    pd.DataFrame | None,
]:
    """
    Извлечь контент через file_extractors; вернуть чанки (текст) или батчи с заголовками (таблица).
    - ("text", list[str]) — чанки для Map-Reduce; третий элемент None.
    - ("vision", list[str]) — один чанк с [VISION_FILE:...] для vision-модели.
    - ("table", list[tuple[list[str], list[dict]]]) — батчи (header, rows); третий элемент — исходный DataFrame.
    """
    path = Path(path)
    if not path.exists():
        raise ParseError(f"File not found: {path}")

    kind, content = extract_content(path)

    if root_dir is not None:
        try:
            display_path = str(path.relative_to(root_dir)).replace("\\", "/")
        except ValueError:
            display_path = path.name
    else:
        display_path = path.name

    if kind == "vision":
        img_path = Path(str(content))
        if not img_path.is_file():
            raise ParseError(f"Image not found: {img_path}")
        chunk = f"[FILE_PATH: {display_path}]\n[Файл: {display_path}]\n[VISION_FILE: {img_path}]\n"
        logger.info("Parsed vision image: %s", display_path)
        return ("vision", [chunk], None)

    if kind == "text":
        raw = content.strip()
        if not raw:
            raise ParseError(f"Empty text content: {path}")
        chunks = chunk_text_for_map_file(path, raw, dynamic_chunk_size, overlap_tokens, root_dir=root_dir)
        logger.info("Parsed text: %s chunks", len(chunks))
        return ("text", chunks, None)

    assert kind == "table" and isinstance(content, pd.DataFrame)
    df = content
    if df.empty:
        raise ParseError(f"Empty table: {path}")
    columns = list(df.columns)
    tokens_per_row = _estimate_tokens_per_row(df)
    rows_per_batch = max(1, int(dynamic_chunk_size / tokens_per_row))
    batches: list[TableBatch] = []
    for start in range(0, len(df), rows_per_batch):
        slice_df = df.iloc[start : start + rows_per_batch]
        rows = slice_df.to_dict(orient="records")
        batches.append((columns, rows))
    logger.info("Parsed table: %s rows, %s batches", len(df), len(batches))
    return ("table", batches, df)
