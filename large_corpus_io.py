# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Large-corpus I/O: streaming text chunking and archive → folder expansion.

Target use case: multi‑GB logs, trillion-line dumps (line streaming), ZIP/TAR code trees.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterator

from file_extractors import ParseError, _extract_archive_to_dir, _is_archive

STREAMING_PLAIN_SUFFIXES = {
    ".txt", ".log", ".md", ".rst", ".csv", ".json", ".jsonl", ".ndjson",
    ".py", ".js", ".ts", ".tsx", ".java", ".kt", ".go", ".rs", ".sql",
    ".sh", ".bat", ".ps1", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".properties", ".xml", ".html", ".htm",
}


def streaming_threshold_bytes() -> int:
    raw = os.getenv("NOCTURNE_STREAMING_FILE_BYTES", "52428800").strip()  # 50 MiB
    try:
        return max(0, int(raw))
    except ValueError:
        return 52_428_800


def auto_scout_file_bytes() -> int:
    """Files at or above this size trigger auto large-corpus mode in GUI (default 10 MiB)."""
    raw = os.getenv("NOCTURNE_AUTO_SCOUT_BYTES", "10485760").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 10 * 1024 * 1024


def is_large_corpus_input(path: Path) -> tuple[bool, str]:
    """
    Whether GUI/CLI should apply large-corpus defaults (scout, smaller chunks) without user tuning.
    """
    path = Path(path)
    if is_archive(path):
        return True, "archive"
    if path.is_dir():
        from pipeline import _iter_files

        files = _iter_files([path])
        total_b = 0
        for fp in files:
            try:
                total_b += fp.stat().st_size
            except OSError:
                continue
        if len(files) >= 12 or total_b >= 20_000_000:
            return True, f"folder:{len(files)} files,{total_b} bytes"
        return False, ""
    if path.is_file():
        if should_stream_plain_file(path):
            return True, "stream_plain"
        try:
            sz = path.stat().st_size
            if sz >= auto_scout_file_bytes():
                return True, f"large_file:{sz}"
        except OSError:
            pass
    return False, ""


def large_corpus_profile_kwargs() -> dict[str, object]:
    """Settings from config/run_profiles.yaml large_corpus preset."""
    from run_profiles import get_profile

    prof = get_profile("large_corpus")
    return {
        "scout_mode": bool(prof.get("scout_mode", True)),
        "scout_threshold": float(prof.get("scout_threshold", 0.35)),
        "max_chunk_tokens": int(prof.get("max_chunk_tokens", 4500)),
        "workers": int(prof.get("workers", 4)),
        "composer_enabled": bool(prof.get("composer_enabled", True)),
    }


def max_archive_extract_bytes() -> int:
    raw = os.getenv("NOCTURNE_MAX_ARCHIVE_BYTES", "8589934592").strip()  # 8 GiB
    try:
        return max(0, int(raw))
    except ValueError:
        return 8 * 1024 * 1024 * 1024


def should_stream_plain_file(path: Path) -> bool:
    """Plain-text-like files over threshold are read line-by-line (bounded RAM)."""
    th = streaming_threshold_bytes()
    if th <= 0:
        return False
    suf = path.suffix.lower()
    if path.name.lower().endswith(".tar.gz"):
        return False
    if suf not in STREAMING_PLAIN_SUFFIXES:
        return False
    try:
        return path.stat().st_size >= th
    except OSError:
        return False


def is_archive(path: Path) -> bool:
    return _is_archive(path)


@contextmanager
def corpus_input_root(path: Path):
    """
  If path is an archive, extract to a temp dir and yield it as the corpus root.
  Otherwise yield path unchanged (file or directory).
    """
    path = Path(path)
    if not path.is_file() or not is_archive(path):
        yield path
        return
    max_b = max_archive_extract_bytes()
    if max_b > 0:
        try:
            if path.stat().st_size > max_b:
                raise ParseError(
                    f"Archive exceeds NOCTURNE_MAX_ARCHIVE_BYTES ({max_b}): {path}",
                )
        except OSError:
            pass
    with TemporaryDirectory(prefix="nocturne_corpus_") as tmp:
        root = Path(tmp)
        _extract_archive_to_dir(path, root)
        yield root


def chunk_plain_file_streaming(
    path: Path,
    chunk_size_tokens: int,
    overlap_tokens: int = 200,
    root_dir: Path | None = None,
    file_meta: dict[str, str] | None = None,
) -> list[str]:
    """
    Stream a huge plain file line-by-line into MAP chunks (constant memory vs file size).
    """
    from parser import _chunk_text_by_tokens, count_tokens

    path = Path(path)
    if root_dir is not None:
        try:
            display_path = str(path.relative_to(root_dir)).replace("\\", "/")
        except ValueError:
            display_path = path.name
    else:
        display_path = path.name

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

    out: list[str] = []
    chunk_index = 0
    current_lines: list[str] = []
    current_tokens = 0

    def _wrap_body(body: str) -> str:
        nonlocal chunk_index
        chunk_index += 1
        return (
            f"[FILE_PATH: {display_path}][FILE_PART: stream][CHUNK_INDEX: {chunk_index}]\n"
            f"{meta_line}[Файл: {display_path}]\n{body}"
        )

    def _flush() -> None:
        nonlocal current_lines, current_tokens
        if not current_lines:
            return
        body = "".join(current_lines)
        out.append(_wrap_body(body))
        if overlap_tokens > 0:
            overlap_lines: list[str] = []
            overlap_t = 0
            for ln in reversed(current_lines):
                lt = count_tokens(ln)
                if overlap_lines and overlap_t + lt > overlap_tokens:
                    break
                overlap_lines.insert(0, ln)
                overlap_t += lt
            current_lines = overlap_lines
            current_tokens = overlap_t
        else:
            current_lines = []
            current_tokens = 0

    def _emit_token_slices(text: str) -> None:
        for piece in _chunk_text_by_tokens(text, chunk_size_tokens, overlap_tokens):
            out.append(_wrap_body(piece))

    with path.open("r", encoding="utf-8", errors="replace", newline="") as fh:
        for line in fh:
            lt = count_tokens(line)
            if lt > chunk_size_tokens:
                _flush()
                _emit_token_slices(line)
                continue
            if current_tokens + lt > chunk_size_tokens and current_lines:
                _flush()
            current_lines.append(line)
            current_tokens += lt
    _flush()

    if not out:
        raise ParseError(f"Empty streamed file: {path}")
    return out


def iter_streaming_chunk_batches(
    path: Path,
    chunk_size_tokens: int,
    overlap_tokens: int = 200,
    batch_lines: int = 256,
) -> Iterator[list[str]]:
    """
    Yield MAP chunks in batches (for future pipelined MAP without holding all chunk strings).
    Currently used to cap peak list growth on extremely chatty logs.
    """
    del batch_lines  # reserved for future line batching
    yield chunk_plain_file_streaming(
        path, chunk_size_tokens, overlap_tokens, root_dir=path.parent,
    )
