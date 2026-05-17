# SPDX-License-Identifier: AGPL-3.0-or-later
"""Corpus planning: estimates, dry-run, file-level relevance (no LLM)."""
from __future__ import annotations

import os
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from large_corpus_io import corpus_input_root, is_archive, is_large_corpus_input


@dataclass(slots=True)
class CorpusPlan:
    source: Path
    files_total: int
    files_after_file_filter: int
    files_skipped_heuristic: int
    bytes_total: int
    chunks_estimated: int
    chunks_after_scout_est: int
    chunk_size_tokens: int
    scout_mode: bool
    scout_threshold: float
    large_corpus: bool
    warnings: list[str] = field(default_factory=list)

    @property
    def eta_minutes(self) -> float | None:
        if self.chunks_after_scout_est <= 0:
            return 0.0
        workers = max(1, min(4, int(os.getenv("NOCTURNE_PLANNER_WORKERS", "4"))))
        sec = self.chunks_after_scout_est * 8.0 / workers
        if self.scout_mode:
            sec += self.chunks_after_scout_est * 2.0
        return round(sec / 60.0, 1)


def _query_terms(query: str) -> set[str]:
    return {w.lower() for w in re.findall(r"\w{3,}", query.lower())}


def file_relevance_heuristic(path: Path, query: str) -> float:
    """0..1 from filename + first 4KB (file-level pre-scout, no LLM)."""
    terms = _query_terms(query)
    score = 0.25
    name = path.name.lower()
    for t in terms:
        if t in name:
            score += 0.15
    try:
        if path.stat().st_size == 0:
            return 0.0
        sample = path.read_bytes()[:4096].decode("utf-8", errors="ignore").lower()
        hits = sum(1 for t in terms if t in sample)
        score += min(0.45, hits * 0.12)
    except OSError:
        return 0.1
    return max(0.0, min(1.0, score))


def _estimate_chunks_for_file(path: Path, chunk_size_tokens: int) -> int:
    try:
        size = path.stat().st_size
    except OSError:
        return 1
    if size <= 0:
        return 0
    est_tokens = max(1, size // 4)
    return max(1, (est_tokens + chunk_size_tokens - 1) // chunk_size_tokens)


@contextmanager
def corpus_files_root(source: Path) -> Iterator[Path]:
    source = Path(source)
    if source.is_file() and is_archive(source):
        with corpus_input_root(source) as root:
            yield root
    else:
        yield source


def collect_files(root: Path) -> list[Path]:
    from pipeline import _iter_files

    if root.is_file():
        return [root]
    return _iter_files([root])


def filter_files_by_relevance(
    files: list[Path],
    query: str,
    *,
    scout_mode: bool,
    threshold: float,
) -> tuple[list[Path], int]:
    if not scout_mode or not query.strip():
        return files, 0
    kept: list[Path] = []
    skipped = 0
    floor = max(0.15, threshold * 0.85)
    for fp in files:
        if file_relevance_heuristic(fp, query) >= floor:
            kept.append(fp)
        else:
            skipped += 1
    if not kept and files:
        return files, 0
    return kept, skipped


def plan_corpus(
    source: Path,
    query: str,
    chunk_size_tokens: int,
    *,
    scout_mode: bool = False,
    scout_threshold: float = 0.35,
) -> CorpusPlan:
    source = Path(source)
    warnings: list[str] = []
    large, _ = is_large_corpus_input(source)
    chunk_size_tokens = max(500, chunk_size_tokens)

    with corpus_files_root(source) as root:
        files = collect_files(root)

    bytes_total = 0
    chunks_est = 0
    for fp in files:
        try:
            bytes_total += fp.stat().st_size
        except OSError:
            continue
        chunks_est += _estimate_chunks_for_file(fp, chunk_size_tokens)

    filtered, skipped = filter_files_by_relevance(
        files, query, scout_mode=scout_mode, threshold=scout_threshold,
    )
    chunks_filtered = sum(_estimate_chunks_for_file(fp, chunk_size_tokens) for fp in filtered)

    if scout_mode:
        skip_ratio = 1.0 - scout_threshold
        chunks_after = max(1, int(chunks_filtered * (1.0 - skip_ratio * 0.65)))
    else:
        chunks_after = chunks_filtered

    if chunks_est > 50_000:
        warnings.append("Очень много чанков: включите scout и узкий запрос.")
    if bytes_total > 2_000_000_000:
        warnings.append("Корпус >2GB: убедитесь в свободном месте на диске (chunk cache).")

    return CorpusPlan(
        source=source,
        files_total=len(files),
        files_after_file_filter=len(filtered),
        files_skipped_heuristic=skipped,
        bytes_total=bytes_total,
        chunks_estimated=chunks_est,
        chunks_after_scout_est=chunks_after,
        chunk_size_tokens=chunk_size_tokens,
        scout_mode=scout_mode,
        scout_threshold=scout_threshold,
        large_corpus=large,
        warnings=warnings,
    )


def format_plan_ru(plan: CorpusPlan) -> str:
    mb = plan.bytes_total / (1024 * 1024)
    lines = [
        f"Файлов: {plan.files_total}",
        f"После отбора по файлу: {plan.files_after_file_filter}"
        + (f" (отсечено {plan.files_skipped_heuristic})" if plan.files_skipped_heuristic else ""),
        f"Объём: {mb:.1f} MB",
        f"Чанков (оценка): ~{plan.chunks_estimated}",
    ]
    if plan.scout_mode:
        lines.append(
            f"После scout (~): ~{plan.chunks_after_scout_est} deep MAP "
            f"(порог {plan.scout_threshold})",
        )
    else:
        lines.append(f"Deep MAP (~): ~{plan.chunks_after_scout_est}")
    if plan.eta_minutes is not None:
        lines.append(f"Ориентир времени: ~{plan.eta_minutes} мин (зависит от LM Studio)")
    for w in plan.warnings:
        lines.append(f"⚠ {w}")
    return " | ".join(lines)
