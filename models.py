from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    chunk_id: str
    source_path: str
    text: str
    tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalHit:
    chunk_id: str
    score: float
    source_path: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IndexStats:
    chunks_total: int
    files_total: int
    index_dir: Path
    embedding_model: str
