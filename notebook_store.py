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
Notebooks — постоянные именованные коллекции источников (в духе NotebookLM).

Блокнот = самодостаточная папка::

    <root>/notebooks/<slug>_<ts>/
        notebook.json   — метаданные (имя, источники, модели, счётчики индекса)
        index/          — FAISS + BM25 (через pipeline.build_index)
        sources/        — производный контент (текст по URL / транскрипты);
                          большие дампы индексируются по месту (не копируем)
        notes/          — сгенерированные материалы Studio (гайд, FAQ, …)
        chat.jsonl      — многошаговая история чата с цитатами

Модуль намеренно держит только stdlib на верхнем уровне; тяжёлые зависимости
(faiss/pipeline) импортируются лениво внутри методов, чтобы блокноты можно было
перечислять/редактировать без загрузки RAG-стэка.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("nocturne")

# Источники-папки/файлы индексируются по месту; в sources/ копим только то,
# что породили мы сами (URL → текст). Имя реестрового файла блокнота.
_META_NAME = "notebook.json"
_LAST_ACTIVE = "last_active.txt"


def notebooks_root() -> Path:
    """Корень всех блокнотов.

    Приоритет: явный ``NOCTURNE_NOTEBOOKS_DIR`` → каталог рядом с
    ``NOCTURNE_CACHE_DIR`` (так упакованный .exe пишет в ``NocturneData/`` рядом
    с бинарником, а не внутрь read-only бандла) → ``.local/notebooks`` при
    запуске из исходников.
    """
    override = os.getenv("NOCTURNE_NOTEBOOKS_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    cache = os.getenv("NOCTURNE_CACHE_DIR", "").strip()
    if cache:
        return Path(cache).expanduser().resolve().parent / "notebooks"
    # В упакованном .exe писать внутрь бандла нельзя (read-only). Если bootstrap
    # по какой-то причине не выставил NOCTURNE_CACHE_DIR — кладём рядом с .exe.
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "NocturneData" / "notebooks"
    return Path(__file__).resolve().parent / ".local" / "notebooks"


def _slugify(name: str) -> str:
    """ASCII-safe folder name for the notebook.

    Deliberately ASCII-only: faiss on Windows opens index files via the narrow C
    ``fopen``, which cannot handle non-ASCII (e.g. Cyrillic) paths — so a notebook
    named «Изучение ИБ» must still live in an ASCII directory. The human-readable
    name (with any script) is preserved in notebook.json, not on the path.
    """
    safe = "".join(c if (c.isascii() and c.isalnum()) or c in "-_" else "_" for c in name.strip())
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe[:48] or "notebook"


def _now() -> int:
    return int(time.time())


@dataclass(slots=True)
class Source:
    """Один источник блокнота: файл, папка или загруженный URL."""

    id: str
    kind: str  # "file" | "folder" | "url"
    display: str
    # Реальный путь, который скармливается индексатору (для url — файл в sources/).
    path: str = ""
    url: str = ""
    added_at: int = field(default_factory=_now)
    bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "display": self.display,
            "path": self.path,
            "url": self.url,
            "added_at": self.added_at,
            "bytes": self.bytes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Source":
        return cls(
            id=str(d.get("id") or _new_id()),
            kind=str(d.get("kind") or "file"),
            display=str(d.get("display") or ""),
            path=str(d.get("path") or ""),
            url=str(d.get("url") or ""),
            added_at=int(d.get("added_at") or _now()),
            bytes=int(d.get("bytes") or 0),
        )


def _new_id() -> str:
    return "src_" + secrets.token_hex(4)


# Палитра обложек и эмодзи для галереи «архива исследований».
_COVER_PALETTE = [
    "#6366f1", "#0ea5e9", "#10b981", "#f59e0b",
    "#ef4444", "#ec4899", "#8b5cf6", "#14b8a6",
    "#3b82f6", "#f97316",
]
_COVER_EMOJI = ["📓", "📚", "🔬", "🗂️", "🧠", "📊", "🛰️", "🧩", "📡", "🗃️"]


def _auto_cover(seed: str) -> tuple[str, str]:
    """Детерминированная обложка (цвет, эмодзи) по идентификатору блокнота."""
    import hashlib

    h = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16)
    return _COVER_PALETTE[h % len(_COVER_PALETTE)], _COVER_EMOJI[(h // 7) % len(_COVER_EMOJI)]


@dataclass(slots=True)
class Notebook:
    """Метаданные + операции над одним блокнотом."""

    id: str
    name: str
    dir: Path
    created_at: int = field(default_factory=_now)
    updated_at: int = field(default_factory=_now)
    embedding_model: str = ""
    chat_model: str = ""
    sources: list[Source] = field(default_factory=list)
    index_chunks: int = 0
    index_files: int = 0
    index_built_at: int | None = None
    description: str = ""
    emoji: str = ""
    color: str = ""

    # --- пути внутри блокнота ------------------------------------------- #
    @property
    def index_dir(self) -> Path:
        return self.dir / "index"

    @property
    def sources_dir(self) -> Path:
        return self.dir / "sources"

    @property
    def notes_dir(self) -> Path:
        return self.dir / "notes"

    @property
    def chat_path(self) -> Path:
        return self.dir / "chat.jsonl"

    @property
    def has_index(self) -> bool:
        return (self.index_dir / "chunks_meta.jsonl").is_file()

    # --- сериализация ---------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "embedding_model": self.embedding_model,
            "chat_model": self.chat_model,
            "description": self.description,
            "emoji": self.emoji,
            "color": self.color,
            "index": {
                "chunks_total": self.index_chunks,
                "files_total": self.index_files,
                "built_at": self.index_built_at,
            },
            "sources": [s.to_dict() for s in self.sources],
        }

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.updated_at = _now()
        tmp = self.dir / (_META_NAME + ".tmp")
        tmp.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        tmp.replace(self.dir / _META_NAME)

    @classmethod
    def from_dir(cls, path: Path) -> "Notebook | None":
        meta_path = path / _META_NAME
        if not meta_path.is_file():
            return None
        try:
            d = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001 — повреждённый блокнот не должен ронять список
            logger.warning("notebook meta unreadable %s: %s", path, exc)
            return None
        index = d.get("index") or {}
        nb_id = str(d.get("id") or path.name)
        # Обложку старым блокнотам (без полей) достраиваем детерминированно.
        auto_color, auto_emoji = _auto_cover(nb_id)
        return cls(
            id=nb_id,
            name=str(d.get("name") or path.name),
            dir=path,
            created_at=int(d.get("created_at") or _now()),
            updated_at=int(d.get("updated_at") or _now()),
            embedding_model=str(d.get("embedding_model") or ""),
            chat_model=str(d.get("chat_model") or ""),
            sources=[Source.from_dict(s) for s in (d.get("sources") or []) if isinstance(s, dict)],
            index_chunks=int(index.get("chunks_total") or 0),
            index_files=int(index.get("files_total") or 0),
            index_built_at=index.get("built_at"),
            description=str(d.get("description") or ""),
            emoji=str(d.get("emoji") or auto_emoji),
            color=str(d.get("color") or auto_color),
        )

    # --- управление источниками ----------------------------------------- #
    def add_path_source(self, path: Path) -> Source:
        """Добавить файл или папку как источник (индексируется по месту)."""
        path = Path(path).resolve()
        kind = "folder" if path.is_dir() else "file"
        existing = next((s for s in self.sources if s.kind in {"file", "folder"}
                         and Path(s.path) == path), None)
        if existing is not None:
            return existing
        # Для папок НЕ обходим дерево ради размера: на многогигабайтных дампах это
        # бы заморозило вызывающий UI-поток. Размер тут лишь информационный.
        try:
            size = 0 if path.is_dir() else path.stat().st_size
        except Exception:
            size = 0
        src = Source(
            id=_new_id(),
            kind=kind,
            display=path.name + ("/" if kind == "folder" else ""),
            path=str(path),
            bytes=int(size),
        )
        self.sources.append(src)
        self.save()
        return src

    def add_url_source(self, url: str, text: str, title: str = "") -> Source:
        """Сохранить загруженный по URL текст в sources/ и зарегистрировать."""
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        sid = _new_id()
        out = self.sources_dir / f"{sid}.txt"
        header = f"[SOURCE_URL: {url}]\n"
        if title:
            header += f"[SOURCE_TITLE: {title}]\n"
        out.write_text(header + "\n" + (text or ""), encoding="utf-8")
        display = title.strip() or _short_url(url)
        src = Source(
            id=sid,
            kind="url",
            display=display,
            path=str(out.resolve()),
            url=url,
            bytes=out.stat().st_size,
        )
        self.sources.append(src)
        self.save()
        return src

    def remove_source(self, source_id: str) -> bool:
        src = next((s for s in self.sources if s.id == source_id), None)
        if src is None:
            return False
        self.sources = [s for s in self.sources if s.id != source_id]
        # Удаляем только то, что мы сами породили (sources/), не трогаем файлы юзера.
        # Обе стороны резолвим: на macOS/Windows temp бывает симлинком (/var→/private/var)
        # или 8.3-путём (RUNNER~1), и нерезолвленное сравнение не совпадёт.
        if src.kind == "url" and src.path:
            try:
                p = Path(src.path).resolve()
                if self.sources_dir.resolve() in p.parents:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
        self.save()
        return True

    def index_input_paths(self) -> list[Path]:
        """Существующие реальные пути для скармливания индексатору."""
        out: list[Path] = []
        for s in self.sources:
            if not s.path:
                continue
            p = Path(s.path)
            if p.exists():
                out.append(p)
        return out

    # --- индекс ---------------------------------------------------------- #
    def rebuild_index(
        self,
        *,
        base_url: str,
        api_key: str,
        embedding_model: str,
        chunk_size_tokens: int,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> "IndexStats":
        """Полная пересборка FAISS+BM25 индекса из всех источников.

        FAISS ``IndexFlatIP`` неизменяем — добавление источника означает
        пересборку из объединения; это сознательный компромисс ради простоты.
        """
        from pipeline import build_index

        paths = self.index_input_paths()
        if not paths:
            raise RuntimeError("В блокноте нет доступных источников для индексации")
        skipped = len(self.sources) - len(paths)
        if skipped > 0:
            logger.warning(
                "notebook %s: %s of %s sources unavailable (skipped during index)",
                self.id, skipped, len(self.sources),
            )
        self.index_dir.mkdir(parents=True, exist_ok=True)
        stats: IndexStats = build_index(
            input_paths=paths,
            index_dir=self.index_dir,
            base_url=base_url,
            api_key=api_key,
            embedding_model=embedding_model,
            chunk_size_tokens=chunk_size_tokens,
            on_progress=on_progress,
        )
        self.embedding_model = embedding_model
        self.index_chunks = stats.chunks_total
        self.index_files = stats.files_total
        self.index_built_at = _now()
        self.save()
        return stats

    def query(
        self,
        question: str,
        *,
        base_url: str,
        api_key: str,
        embedding_model: str = "",
        top_k: int = 8,
    ) -> list["RetrievalHit"]:
        from pipeline import query_index

        return query_index(
            question=question,
            index_dir=self.index_dir,
            base_url=base_url,
            api_key=api_key,
            embedding_model=embedding_model or self.embedding_model,
            top_k=top_k,
        )

    # --- история чата ---------------------------------------------------- #
    def append_chat_turn(
        self, role: str, content: str, citations: list[dict[str, Any]] | None = None
    ) -> None:
        rec = {"role": role, "content": content, "citations": citations or [], "ts": _now()}
        with self.chat_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def load_chat(self, limit: int | None = None) -> list[dict[str, Any]]:
        if not self.chat_path.is_file():
            return []
        out: list[dict[str, Any]] = []
        with self.chat_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
        if limit is not None and limit > 0:
            return out[-limit:]
        return out

    def clear_chat(self) -> None:
        try:
            self.chat_path.unlink(missing_ok=True)
        except Exception:
            pass

    # --- заметки Studio -------------------------------------------------- #
    def save_note(self, filename: str, content: str) -> Path:
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in filename)[:80] or "note.md"
        out = self.notes_dir / safe
        out.write_text(content, encoding="utf-8")
        return out

    def list_notes(self) -> list[Path]:
        if not self.notes_dir.is_dir():
            return []
        return sorted(
            (p for p in self.notes_dir.iterdir() if p.is_file()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

    # --- метаданные галереи --------------------------------------------- #
    def set_meta(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        emoji: str | None = None,
        color: str | None = None,
    ) -> None:
        if name is not None and name.strip():
            self.name = name.strip()
        if description is not None:
            self.description = description.strip()
        if emoji:
            self.emoji = emoji
        if color:
            self.color = color
        self.save()

    def total_source_bytes(self) -> int:
        return sum(int(s.bytes or 0) for s in self.sources)


def _short_url(url: str) -> str:
    u = re.sub(r"^https?://", "", url).rstrip("/")
    return (u[:60] + "…") if len(u) > 61 else u


# ---------------------------------------------------------------------- #
#  CRUD на уровне корня
# ---------------------------------------------------------------------- #
def create_notebook(name: str) -> Notebook:
    root = notebooks_root()
    root.mkdir(parents=True, exist_ok=True)
    slug = _slugify(name)
    nb_id = f"{slug}_{_now()}"
    nb_dir = root / nb_id
    # Защита от коллизии по времени.
    while nb_dir.exists():
        nb_id = f"{slug}_{_now()}_{secrets.token_hex(2)}"
        nb_dir = root / nb_id
    nb_dir.mkdir(parents=True)
    color, emoji = _auto_cover(nb_id)
    nb = Notebook(id=nb_id, name=name.strip() or slug, dir=nb_dir, color=color, emoji=emoji)
    nb.save()
    set_last_active(nb_id)
    return nb


def list_notebooks() -> list[Notebook]:
    root = notebooks_root()
    if not root.is_dir():
        return []
    out: list[Notebook] = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        nb = Notebook.from_dir(p)
        if nb is not None:
            out.append(nb)
    out.sort(key=lambda n: n.updated_at, reverse=True)
    return out


def load_notebook(notebook_id: str) -> Notebook | None:
    root = notebooks_root()
    nb_dir = root / notebook_id
    if not nb_dir.is_dir():
        return None
    return Notebook.from_dir(nb_dir)


def delete_notebook(notebook_id: str) -> bool:
    nb = load_notebook(notebook_id)
    if nb is None:
        return False
    try:
        shutil.rmtree(nb.dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("delete notebook %s failed: %s", notebook_id, exc)
        return False
    if get_last_active() == notebook_id:
        set_last_active("")
    return True


def get_last_active() -> str:
    p = notebooks_root() / _LAST_ACTIVE
    if not p.is_file():
        return ""
    try:
        return p.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def set_last_active(notebook_id: str) -> None:
    root = notebooks_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        (root / _LAST_ACTIVE).write_text(notebook_id, encoding="utf-8")
    except Exception:
        pass


# Аннотации типов для линтеров (отложенный импорт во избежание тяжёлых зависимостей).
if False:  # pragma: no cover - typing only
    from models import IndexStats, RetrievalHit  # noqa: F401
