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
Вкладка «Блокноты» (NotebookLM-режим) для NocturneApp.

Две view, как в продукте Google:
- **Архив исследований** — галерея карточек блокнотов: поиск, создание,
  открытие, удаление. Сюда возвращаешься к прошлым исследованиям.
- **Рабочее пространство** — открытый блокнот в три колонки
  (Источники · Чат · Studio) с хедером (эмодзи, название, мета,
  переименование/описание, удаление).

Вынесено отдельным mixin-классом, чтобы не раздувать gui.py. Тяжёлые операции
(индексация, ответ модели, генерация материалов) идут в daemon-потоках;
обновление виджетов — только через ``self.after(0, …)`` (конвенция проекта).
Опирается на готовые помощники NocturneApp: ``_append_log_line``,
``_pick_embedding_model``, ``_get_context_budget``, ``_get_response_reserve`` и
переменные подключения/моделей (``_url_var`` и т.д.).
"""
from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import customtkinter as ctk
from tkinter import messagebox

import notebook_store as nbs
from lmstudio_config import sanitize_for_log
from notebook_chat import ChatResult, answer_question
from notebook_studio import MATERIAL_ORDER, MATERIALS, generate_material
from processor import API_BASE, API_KEY
from url_ingest import UrlIngestError, fetch_url

logger = logging.getLogger("nocturne")

# --- палитра (light, dark) ------------------------------------------------- #
_ACCENT = "#2563eb"
_ACCENT_HOVER = "#1d4ed8"
_DANGER = "#b91c1c"
_DANGER_HOVER = "#991b1b"
_CARD_BG = ("#ffffff", "#212836")
_CARD_BORDER = ("#e2e8f0", "#2f3947")
_PANEL_BG = ("#f1f5f9", "#171c26")
_ROW_BG = ("#f8fafc", "#1b212c")
_MUTED = ("#64748b", "#94a3b8")
_HINT = "gray"
_USER_BUBBLE = ("#2563eb", "#2563eb")
_BOT_BUBBLE = ("#e7ebf3", "#28313f")
_EMOJI_CHOICES = ["📓", "📚", "🔬", "🗂️", "🧠", "📊", "🛰️", "🧩", "📡", "🗃️", "💼", "⚖️"]


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


class NotebookUIMixin:
    """Mixin для NocturneApp: вкладка блокнотов (архив + рабочее пространство)."""

    _nb_current: "nbs.Notebook | None"

    # ================================================================== #
    #  Каркас вкладки
    # ================================================================== #
    def _build_notebooks_tab(self, parent: ctk.CTkFrame) -> None:
        self._nb_current = None
        self._nb_busy = False
        self._nb_view = "archive"
        self._nb_relayout_after: str | None = None
        self._nb_search_var = ctk.StringVar(value="")

        self._nb_container = ctk.CTkFrame(parent, fg_color="transparent")
        self._nb_container.pack(fill="both", expand=True)

        self._nb_status = ctk.CTkLabel(parent, text="", text_color=_HINT, anchor="w")
        self._nb_status.pack(fill="x", pady=(4, 0))

        self._nb_archive = ctk.CTkFrame(self._nb_container, fg_color="transparent")
        self._nb_workspace = ctk.CTkFrame(self._nb_container, fg_color="transparent")
        self._nb_build_archive(self._nb_archive)
        self._nb_build_workspace(self._nb_workspace)

        self._nb_show("archive")
        self._nb_render_archive()

    def _nb_show(self, view: str) -> None:
        self._nb_view = view
        if view == "workspace":
            self._nb_archive.pack_forget()
            self._nb_workspace.pack(fill="both", expand=True)
        else:
            self._nb_workspace.pack_forget()
            self._nb_archive.pack(fill="both", expand=True)

    # ================================================================== #
    #  View 1 — Архив исследований (галерея)
    # ================================================================== #
    def _nb_build_archive(self, root: ctk.CTkFrame) -> None:
        header = ctk.CTkFrame(root, fg_color="transparent")
        header.pack(fill="x", pady=(2, 10))
        ctk.CTkLabel(header, text="📚  Архив исследований",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(side="left")
        ctk.CTkButton(header, text="＋  Новое исследование", height=36,
                      fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
                      command=self._nb_on_new).pack(side="right")
        ctk.CTkEntry(header, textvariable=self._nb_search_var, height=36, width=260,
                     placeholder_text="🔍  Поиск по названию и описанию…"
                     ).pack(side="right", padx=(0, 10))
        self._nb_search_var.trace_add("write", lambda *_: self._nb_render_archive())

        self._nb_card_w = 320
        self._nb_gallery_cols = 0
        self._nb_gallery = ctk.CTkScrollableFrame(root, fg_color="transparent")
        self._nb_gallery.pack(fill="both", expand=True)
        self._nb_gallery.bind("<Configure>", self._nb_on_gallery_resize, add="+")

    def _nb_on_gallery_resize(self, event: Any) -> None:
        # event.width — в Tk-пикселях; ширина карточки — в логических единицах
        # CustomTkinter. Приводим к одной шкале, иначе при HiDPI колонок насчитаем
        # больше, чем влезает.
        try:
            scaling = ctk.ScalingTracker.get_widget_scaling(self._nb_gallery)
        except Exception:
            scaling = 1.0
        avail = event.width / max(scaling, 0.1)
        cols = max(1, int((avail - 8) // (self._nb_card_w + 16)))
        if cols != self._nb_gallery_cols:
            self._nb_gallery_cols = cols
            # ВАЖНО: перерисовку откладываем ИЗ обработчика <Configure>. Уничтожать
            # и пересоздавать виджеты прямо внутри события ресайза их же контейнера —
            # частая причина жёсткого краша Tk на Windows (особенно в собранном .exe,
            # где пустой архив не падал, а карточки — да). after() и дебаунс рвут
            # реентрантность.
            if self._nb_relayout_after is not None:
                try:
                    self.after_cancel(self._nb_relayout_after)
                except Exception:
                    pass
            self._nb_relayout_after = self.after(60, self._nb_relayout_gallery)

    def _nb_relayout_gallery(self) -> None:
        self._nb_relayout_after = None
        if getattr(self, "_closing", False) or not self.winfo_exists():
            return
        if self._nb_view == "archive":
            self._nb_render_archive()

    def _nb_render_archive(self) -> None:
        for w in self._nb_gallery.winfo_children():
            w.destroy()
        notebooks = nbs.list_notebooks()
        q = self._nb_search_var.get().strip().lower()
        if q:
            notebooks = [n for n in notebooks
                         if q in n.name.lower() or q in (n.description or "").lower()]
        if not notebooks:
            self._nb_render_archive_empty(bool(q))
            return
        cols = self._nb_gallery_cols or 2
        # Колонки натуральной ширины + правый «распорный» столбец, чтобы карточки
        # были фиксированного размера и прижаты влево (как в галерее NotebookLM).
        for c in range(cols + 1):
            self._nb_gallery.grid_columnconfigure(c, weight=1 if c == cols else 0)
        for idx, nb in enumerate(notebooks):
            try:
                card = self._nb_build_card(self._nb_gallery, nb)
            except Exception:
                logger.exception("notebook card render failed: %s", getattr(nb, "id", "?"))
                continue
            card.grid(row=idx // cols, column=idx % cols, padx=8, pady=8, sticky="nw")

    def _nb_render_archive_empty(self, filtered: bool) -> None:
        box = ctk.CTkFrame(self._nb_gallery, fg_color="transparent")
        box.grid(row=0, column=0, columnspan=3, pady=60)
        if filtered:
            ctk.CTkLabel(box, text="🔍 Ничего не найдено",
                         font=ctk.CTkFont(size=16, weight="bold")).pack()
            ctk.CTkLabel(box, text="Измените запрос поиска.", text_color=_MUTED).pack(pady=(4, 0))
            return
        ctk.CTkLabel(box, text="📭 Архив пуст", font=ctk.CTkFont(size=18, weight="bold")).pack()
        ctk.CTkLabel(box, text="Создайте первое исследование: загрузите дампы/файлы/ссылки,\n"
                              "постройте индекс и спрашивайте — со ссылками на источники.",
                     text_color=_MUTED, justify="center").pack(pady=(6, 14))
        ctk.CTkButton(box, text="＋  Новое исследование", height=38, width=220,
                      fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
                      command=self._nb_on_new).pack()

    def _nb_build_card(self, parent: ctk.CTkFrame, nb: "nbs.Notebook") -> ctk.CTkFrame:
        # Карточки фиксированного размера → ровная адаптивная сетка.
        card = ctk.CTkFrame(parent, width=self._nb_card_w, height=210, fg_color=_CARD_BG,
                            border_width=1, border_color=_CARD_BORDER, corner_radius=14)
        card.pack_propagate(False)
        # Цветная «обложка».
        cover = ctk.CTkFrame(card, fg_color=nb.color or _ACCENT, height=50, corner_radius=0)
        cover.pack(fill="x")
        cover.pack_propagate(False)
        ctk.CTkLabel(cover, text=nb.emoji or "📓", font=ctk.CTkFont(size=24),
                     fg_color="transparent", text_color="white").pack(side="left", padx=12)
        ctk.CTkButton(cover, text="🗑", width=28, height=24, fg_color="transparent",
                      hover_color="gray25", text_color="white",
                      command=lambda: self._nb_delete(nb)).pack(side="right", padx=8)

        wrap = self._nb_card_w - 30
        body = ctk.CTkFrame(card, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(8, 0))
        title = ctk.CTkLabel(body, text=_clip(nb.name, 48), anchor="w", justify="left",
                             wraplength=wrap, font=ctk.CTkFont(size=15, weight="bold"))
        title.pack(anchor="w")
        ctk.CTkLabel(body, text=_clip(nb.description or "Без описания", 90), text_color=_MUTED,
                     anchor="w", justify="left", wraplength=wrap,
                     font=ctk.CTkFont(size=12)).pack(anchor="w", pady=(3, 0))

        footer = ctk.CTkFrame(card, fg_color="transparent")
        footer.pack(fill="x", padx=14, pady=(0, 2))
        stats = f"📄 {len(nb.sources)}   ◆ {nb.index_chunks}   {self._nb_fmt_ts(nb.updated_at)}"
        ctk.CTkLabel(footer, text=stats, text_color=_MUTED, font=ctk.CTkFont(size=11)).pack(side="left")
        ctk.CTkLabel(footer, text="● индекс" if nb.has_index else "○ нет индекса",
                     font=ctk.CTkFont(size=10),
                     text_color=("#16a34a", "#4ade80") if nb.has_index else _MUTED).pack(side="right")
        ctk.CTkButton(card, text="Открыть  →", height=30, fg_color=_ACCENT,
                      hover_color=_ACCENT_HOVER,
                      command=lambda: self._nb_open(nb.id)).pack(fill="x", padx=14, pady=(4, 12))

        for w in (cover, title):
            w.bind("<Button-1>", lambda _e, i=nb.id: self._nb_open(i))
        return card

    # ================================================================== #
    #  View 2 — Рабочее пространство блокнота
    # ================================================================== #
    def _nb_build_workspace(self, root: ctk.CTkFrame) -> None:
        hdr = ctk.CTkFrame(root, fg_color="transparent")
        hdr.pack(fill="x", pady=(2, 8))
        ctk.CTkButton(hdr, text="←  Архив", width=88, fg_color="transparent", border_width=1,
                      command=self._nb_back_to_archive).pack(side="left")
        self._nb_ws_emoji = ctk.CTkLabel(hdr, text="📓", font=ctk.CTkFont(size=22))
        self._nb_ws_emoji.pack(side="left", padx=(12, 8))
        titlebox = ctk.CTkFrame(hdr, fg_color="transparent")
        titlebox.pack(side="left", fill="x", expand=True)
        self._nb_ws_title = ctk.CTkLabel(titlebox, text="", anchor="w",
                                         font=ctk.CTkFont(size=18, weight="bold"))
        self._nb_ws_title.pack(anchor="w")
        self._nb_ws_meta = ctk.CTkLabel(titlebox, text="", anchor="w", text_color=_MUTED,
                                        font=ctk.CTkFont(size=11))
        self._nb_ws_meta.pack(anchor="w")
        ctk.CTkButton(hdr, text="🗑 Удалить", width=96, fg_color=_DANGER,
                      hover_color=_DANGER_HOVER, command=self._nb_on_delete).pack(side="right")
        ctk.CTkButton(hdr, text="Изменить", width=92, fg_color="transparent", border_width=1,
                      command=self._nb_edit_meta_dialog).pack(side="right", padx=(6, 8))

        cols = ctk.CTkFrame(root, fg_color="transparent")
        cols.pack(fill="both", expand=True)
        # Порядок pack важен: фиксированные боковые панели резервируем ПЕРВЫМИ
        # (left слева, right справа), и только потом растягиваем центр — иначе
        # expand-центр «съедает» место и правая колонка уезжает за край.
        left = ctk.CTkFrame(cols, width=290, fg_color=_PANEL_BG, corner_radius=12)
        left.pack(side="left", fill="y", padx=(0, 8))
        left.pack_propagate(False)
        right = ctk.CTkFrame(cols, width=230, fg_color=_PANEL_BG, corner_radius=12)
        right.pack(side="right", fill="y", padx=(8, 0))
        right.pack_propagate(False)
        center = ctk.CTkFrame(cols, fg_color="transparent")
        center.pack(side="left", fill="both", expand=True)
        self._nb_build_sources_panel(left)
        self._nb_build_chat_panel(center)
        self._nb_build_studio_panel(right)

    def _nb_build_sources_panel(self, left: ctk.CTkFrame) -> None:
        ctk.CTkLabel(left, text="Источники", font=ctk.CTkFont(size=14, weight="bold")
                     ).pack(anchor="w", padx=12, pady=(12, 2))
        ctk.CTkLabel(left, text="Дампы, файлы, папки, ссылки", text_color=_MUTED,
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12)
        add = ctk.CTkFrame(left, fg_color="transparent")
        add.pack(fill="x", padx=12, pady=(6, 0))
        ctk.CTkButton(add, text="＋ Файл", width=84, command=self._nb_add_files).pack(side="left", padx=(0, 4))
        ctk.CTkButton(add, text="＋ Папка", width=84, command=self._nb_add_folder).pack(side="left", padx=(0, 4))
        ctk.CTkButton(add, text="＋ URL", width=70, command=self._nb_add_url).pack(side="left")

        self._nb_sources_frame = ctk.CTkScrollableFrame(left, fg_color="transparent", label_text="")
        self._nb_sources_frame.pack(fill="both", expand=True, padx=8, pady=(8, 4))

        self._nb_index_info = ctk.CTkLabel(left, text="", text_color=_MUTED, anchor="w",
                                           wraplength=270, justify="left", font=ctk.CTkFont(size=11))
        self._nb_index_info.pack(anchor="w", padx=12)
        self._nb_index_progress = ctk.CTkProgressBar(left)
        self._nb_index_progress.pack(fill="x", padx=12, pady=(3, 4))
        self._nb_index_progress.set(0)
        ctk.CTkButton(left, text="🔨  Построить / обновить индекс", command=self._nb_build_index,
                      fg_color=_ACCENT, hover_color=_ACCENT_HOVER).pack(fill="x", padx=12, pady=(0, 12))

    def _nb_build_chat_panel(self, center: ctk.CTkFrame) -> None:
        bar = ctk.CTkFrame(center, fg_color="transparent")
        bar.pack(fill="x", pady=(0, 4))
        ctk.CTkLabel(bar, text="Чат по источникам", font=ctk.CTkFont(size=14, weight="bold")
                     ).pack(side="left")
        ctk.CTkButton(bar, text="Очистить", width=92, fg_color="transparent", border_width=1,
                      command=self._nb_clear_chat).pack(side="right")

        self._nb_chat_frame = ctk.CTkScrollableFrame(center, fg_color=("#f8fafc", "#0f141c"))
        self._nb_chat_frame.pack(fill="both", expand=True)

        input_row = ctk.CTkFrame(center, fg_color="transparent")
        input_row.pack(fill="x", pady=(6, 0))
        self._nb_question = ctk.CTkTextbox(input_row, height=64, wrap="word")
        self._nb_question.pack(side="left", fill="both", expand=True, padx=(0, 8))
        self._nb_ask_btn = ctk.CTkButton(input_row, text="Спросить ▶", width=110,
                                         fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
                                         command=self._nb_on_ask)
        self._nb_ask_btn.pack(side="left")
        self._nb_question.bind("<Control-Return>", lambda _e: (self._nb_on_ask(), "break")[1])

    def _nb_build_studio_panel(self, right: ctk.CTkFrame) -> None:
        ctk.CTkLabel(right, text="Studio", font=ctk.CTkFont(size=14, weight="bold")
                     ).pack(anchor="w", padx=12, pady=(12, 2))
        ctk.CTkLabel(right, text="Материалы по корпусу", text_color=_MUTED,
                     font=ctk.CTkFont(size=11)).pack(anchor="w", padx=12, pady=(0, 6))
        for kind in MATERIAL_ORDER:
            spec = MATERIALS[kind]
            ctk.CTkButton(right, text=spec.title, anchor="w",
                          command=lambda k=kind: self._nb_generate(k)).pack(fill="x", padx=12, pady=2)

        ctk.CTkLabel(right, text="Заметки", text_color=_MUTED, font=ctk.CTkFont(size=12)
                     ).pack(anchor="w", padx=12, pady=(10, 2))
        self._nb_notes_frame = ctk.CTkScrollableFrame(right, fg_color="transparent", label_text="")
        self._nb_notes_frame.pack(fill="both", expand=True, padx=8, pady=(0, 12))

    # ================================================================== #
    #  Открытие / создание / удаление / редактирование
    # ================================================================== #
    def _nb_open(self, notebook_id: str) -> None:
        nb = nbs.load_notebook(notebook_id)
        if nb is None:
            self._nb_set_status("Блокнот не найден", "#f59e0b")
            self._nb_render_archive()
            return
        self._nb_current = nb
        nbs.set_last_active(nb.id)
        self._nb_show("workspace")
        self._nb_render_workspace_header()
        self._nb_render_sources()
        self._nb_render_chat()
        self._nb_render_notes()
        self._nb_update_index_info()
        self._nb_set_status(f"Открыт блокнот «{nb.name}»", "lightgreen")

    def _nb_back_to_archive(self) -> None:
        self._nb_current = None
        self._nb_show("archive")
        self._nb_render_archive()
        self._nb_set_status("")

    def _nb_render_workspace_header(self) -> None:
        nb = self._nb_current
        if nb is None:
            return
        self._nb_ws_emoji.configure(text=nb.emoji or "📓")
        self._nb_ws_title.configure(text=nb.name)
        parts = [f"{len(nb.sources)} источ.", f"{nb.index_chunks} фрагм.",
                 f"обновлён {self._nb_fmt_ts(nb.updated_at, with_time=True)}"]
        if nb.description:
            parts.insert(0, nb.description)
        self._nb_ws_meta.configure(text="   ·   ".join(parts))

    def _nb_on_new(self) -> None:
        dlg = ctk.CTkInputDialog(text="Название исследования:", title="Новое исследование")
        name = (dlg.get_input() or "").strip()
        if not name:
            return
        try:
            nb = nbs.create_notebook(name)
        except Exception as exc:  # noqa: BLE001
            self._nb_set_status(f"Не удалось создать: {sanitize_for_log(str(exc))}", "#f87171")
            return
        self._append_log_line(f"[NB] создан блокнот {nb.id}", "general")
        self._nb_open(nb.id)

    def _nb_on_delete(self) -> None:
        if self._nb_current is not None:
            self._nb_delete(self._nb_current, back_to_archive=True)

    def _nb_delete(self, nb: "nbs.Notebook", back_to_archive: bool = False) -> None:
        if not messagebox.askyesno("Удалить исследование",
                                   f"Удалить «{nb.name}» со всеми источниками, чатом и заметками?"):
            return
        nbs.delete_notebook(nb.id)
        self._append_log_line(f"[NB] удалён блокнот {nb.id}", "general")
        if back_to_archive or (self._nb_current and self._nb_current.id == nb.id):
            self._nb_current = None
            self._nb_show("archive")
        self._nb_render_archive()
        self._nb_set_status("Исследование удалено", _HINT)

    def _nb_edit_meta_dialog(self) -> None:
        nb = self._nb_current
        if nb is None:
            return
        top = ctk.CTkToplevel(self)
        top.title("Изменить исследование")
        top.geometry("480x420")
        top.transient(self)
        chosen_emoji = {"v": nb.emoji or "📓"}

        ctk.CTkLabel(top, text="Название", anchor="w").pack(anchor="w", padx=16, pady=(16, 2))
        name_var = ctk.StringVar(value=nb.name)
        ctk.CTkEntry(top, textvariable=name_var).pack(fill="x", padx=16)

        ctk.CTkLabel(top, text="Описание", anchor="w").pack(anchor="w", padx=16, pady=(12, 2))
        desc_box = ctk.CTkTextbox(top, height=110, wrap="word")
        desc_box.pack(fill="x", padx=16)
        desc_box.insert("1.0", nb.description)

        ctk.CTkLabel(top, text="Иконка", anchor="w").pack(anchor="w", padx=16, pady=(12, 2))
        emoji_row = ctk.CTkFrame(top, fg_color="transparent")
        emoji_row.pack(fill="x", padx=12)
        emoji_btns: dict[str, ctk.CTkButton] = {}

        def pick(e: str) -> None:
            chosen_emoji["v"] = e
            for ev, b in emoji_btns.items():
                b.configure(fg_color=_ACCENT if ev == e else "transparent")

        for e in _EMOJI_CHOICES:
            b = ctk.CTkButton(emoji_row, text=e, width=34, height=34, fg_color="transparent",
                              command=lambda ev=e: pick(ev))
            b.pack(side="left", padx=2, pady=4)
            emoji_btns[e] = b
        pick(chosen_emoji["v"])

        bar = ctk.CTkFrame(top, fg_color="transparent")
        bar.pack(fill="x", padx=16, pady=14)

        def save() -> None:
            nb.set_meta(name=name_var.get(), description=desc_box.get("1.0", "end-1c"),
                        emoji=chosen_emoji["v"])
            self._nb_render_workspace_header()
            self._nb_set_status("Сохранено", "lightgreen")
            top.destroy()

        ctk.CTkButton(bar, text="Сохранить", fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
                      command=save).pack(side="left")
        ctk.CTkButton(bar, text="Отмена", fg_color="transparent", border_width=1,
                      command=top.destroy).pack(side="right")
        top.after(60, top.lift)

    # ================================================================== #
    #  Источники
    # ================================================================== #
    def _nb_add_files(self) -> None:
        if not self._nb_require_notebook() or not self._nb_check_idle():
            return
        paths = ctk.filedialog.askopenfilenames(title="Выберите файлы-источники")
        if not paths:
            return
        for p in paths:
            self._nb_current.add_path_source(Path(p))  # type: ignore[union-attr]
        self._nb_render_sources()
        self._nb_render_workspace_header()
        self._nb_set_status(f"Добавлено источников: {len(paths)}", "lightgreen")

    def _nb_add_folder(self) -> None:
        if not self._nb_require_notebook() or not self._nb_check_idle():
            return
        path = ctk.filedialog.askdirectory(title="Выберите папку-источник")
        if not path:
            return
        self._nb_current.add_path_source(Path(path))  # type: ignore[union-attr]
        self._nb_render_sources()
        self._nb_render_workspace_header()
        self._nb_set_status("Папка добавлена в источники", "lightgreen")

    def _nb_add_url(self) -> None:
        if not self._nb_require_notebook() or not self._nb_check_idle():
            return
        dlg = ctk.CTkInputDialog(text="URL веб-страницы:", title="Добавить источник по URL")
        url = (dlg.get_input() or "").strip()
        if not url:
            return
        nb = self._nb_current
        self._nb_set_busy(True)
        self._nb_set_status(f"Загружаю {url}…")

        def work() -> None:
            try:
                doc = fetch_url(url)
                nb.add_url_source(doc.url, doc.text, doc.title)  # type: ignore[union-attr]
                self._nb_after(self._nb_render_sources)
                self._nb_after(self._nb_render_workspace_header)
                self._nb_after(lambda: self._nb_set_status(
                    f"Добавлен URL: {doc.title or doc.url}", "lightgreen"))
                self._nb_after(lambda: self._append_log_line(
                    f"[NB] url добавлен: {doc.url} ({len(doc.text)} симв.)", "general"))
            except UrlIngestError as exc:
                self._nb_after(lambda e=exc: self._nb_set_status(f"URL: {e}", "#f87171"))
            except Exception as exc:  # noqa: BLE001
                safe = sanitize_for_log(str(exc))
                self._nb_after(lambda: self._nb_set_status(f"Ошибка URL: {safe}", "#f87171"))
            finally:
                self._nb_after(lambda: self._nb_set_busy(False))

        threading.Thread(target=work, daemon=True).start()

    def _nb_render_sources(self) -> None:
        for w in self._nb_sources_frame.winfo_children():
            w.destroy()
        nb = self._nb_current
        if nb is None or not nb.sources:
            ctk.CTkLabel(self._nb_sources_frame, text="Пока нет источников.\nДобавьте файл, папку или URL.",
                         text_color=_HINT, justify="left").pack(anchor="w", padx=4, pady=6)
            return
        icons = {"file": "📄", "folder": "📁", "url": "🔗"}
        for src in nb.sources:
            row = ctk.CTkFrame(self._nb_sources_frame, fg_color=_ROW_BG, corner_radius=8)
            row.pack(fill="x", pady=2)
            ctk.CTkLabel(row, text=icons.get(src.kind, "•"), width=22).pack(side="left", padx=(6, 0))
            ctk.CTkLabel(row, text=src.display, anchor="w", justify="left", wraplength=160,
                         font=ctk.CTkFont(size=12)).pack(side="left", fill="x", expand=True, padx=(2, 0))
            ctk.CTkButton(row, text="×", width=26, fg_color="transparent", hover_color="gray30",
                          command=lambda sid=src.id: self._nb_remove_source(sid)).pack(side="right", padx=(0, 2))
            ctk.CTkButton(row, text="↗", width=26, fg_color="transparent", hover_color="gray30",
                          command=lambda p=src.path: self._nb_open_path(p)).pack(side="right")

    def _nb_remove_source(self, source_id: str) -> None:
        if self._nb_current is None or not self._nb_check_idle():
            return
        self._nb_current.remove_source(source_id)
        self._nb_render_sources()
        self._nb_render_workspace_header()

    def _nb_update_index_info(self) -> None:
        nb = self._nb_current
        if nb is None:
            self._nb_index_info.configure(text="")
            return
        if nb.has_index:
            self._nb_index_info.configure(
                text=f"● Индекс: {nb.index_chunks} фрагм. / {nb.index_files} файлов")
            self._nb_index_progress.set(1)
        else:
            self._nb_index_info.configure(text="○ Индекс не построен")
            self._nb_index_progress.set(0)

    def _nb_build_index(self) -> None:
        nb = self._nb_current
        if not self._nb_require_notebook() or nb is None:
            return
        if not self._nb_check_idle():
            return
        if not nb.index_input_paths():
            self._nb_set_status("Добавьте хотя бы один источник", "#f59e0b")
            return
        emb = self._pick_embedding_model()
        if not emb:
            self._nb_set_status("Не выбрана embedding-модель (сайдбар)", "#f59e0b")
            return
        base_url = self._url_var.get().strip() or API_BASE
        api_key = self._api_key_var.get().strip() or API_KEY
        # Индекс блокнота — для RETRIEVAL по эмбеддингам: чанки маленькие (~512 ток.,
        # под окно embedding-модели), а НЕ под контекст чат-модели. См.
        # notebook_store.notebook_index_chunk_tokens.
        chunk_size = nbs.notebook_index_chunk_tokens()
        self._nb_set_busy(True)
        self._nb_set_status("Строю индекс…")
        self._nb_index_progress.set(0)

        def on_progress(cur: int, total: int, phase: str) -> None:
            frac = (cur / total) if total else 0
            self._nb_after(lambda: self._nb_index_progress.set(min(1.0, frac)))
            self._nb_after(lambda: self._nb_set_status(f"Индекс [{phase}]: {cur}/{total}…"))

        def work() -> None:
            try:
                stats, incremental = nb.update_index(
                    base_url=base_url, api_key=api_key, embedding_model=emb,
                    chunk_size_tokens=chunk_size, on_progress=on_progress,
                )
                mode = "обновлён" if incremental else "пересобран"
                self._nb_after(lambda: self._nb_set_status(
                    f"Индекс {mode}: {stats.chunks_total} фрагм. / {stats.files_total} файлов",
                    "lightgreen"))
                self._nb_after(self._nb_update_index_info)
                self._nb_after(self._nb_render_workspace_header)
                self._nb_after(lambda: self._append_log_line(
                    f"[NB] индекс {nb.id}: chunks={stats.chunks_total} files={stats.files_total}",
                    "summary"))
            except Exception as exc:  # noqa: BLE001
                safe = sanitize_for_log(str(exc))
                self._nb_after(lambda: self._nb_set_status(f"Ошибка индекса: {safe}", "#f87171"))
            finally:
                self._nb_after(lambda: self._nb_set_busy(False))

        threading.Thread(target=work, daemon=True).start()

    # ================================================================== #
    #  Чат
    # ================================================================== #
    def _nb_on_ask(self) -> None:
        nb = self._nb_current
        if not self._nb_require_notebook() or nb is None:
            return
        if not self._nb_check_idle():
            return
        question = self._nb_question.get("1.0", "end-1c").strip()
        if not question:
            self._nb_set_status("Введите вопрос", "#f59e0b")
            return
        if not nb.has_index:
            self._nb_set_status("Сначала постройте индекс блокнота", "#f59e0b")
            return
        model = self._model_var.get().strip()
        if not model or model.startswith("("):
            self._nb_set_status("Выберите LLM-модель (сайдбар)", "#f59e0b")
            return
        emb = self._pick_embedding_model()
        base_url = self._url_var.get().strip() or API_BASE
        api_key = self._api_key_var.get().strip() or API_KEY
        api_mode = self._api_mode_var.get().strip().lower()

        history = nb.load_chat()
        nb.append_chat_turn("user", question)
        self._nb_add_message_row("user", question, [])
        self._nb_question.delete("1.0", "end")
        self._nb_set_busy(True)
        self._nb_set_status("Ищу в источниках и формирую ответ…")
        thinking = self._nb_add_message_row("assistant", "…думаю…", [], placeholder=True)

        def work() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result: ChatResult = loop.run_until_complete(
                    answer_question(
                        nb, question, base_url=base_url, api_key=api_key,
                        chat_model=model, embedding_model=emb, api_mode=api_mode,
                        history=history,
                        on_log=lambda m: self._append_log_line(f"[NB chat] {m}", "general"),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                safe = sanitize_for_log(str(exc))
                self._nb_after(lambda: self._nb_finish_answer(
                    thinking, ChatResult(answer=f"Ошибка: {safe}", citations=[], contexts=[]), nb))
                return
            finally:
                loop.close()
                self._nb_after(lambda: self._nb_set_busy(False))
            self._nb_after(lambda: self._nb_finish_answer(thinking, result, nb))

        threading.Thread(target=work, daemon=True).start()

    def _nb_finish_answer(self, placeholder: ctk.CTkFrame, result: ChatResult, nb: Any) -> None:
        # Историю сохраняем всегда — в тот блокнот, по которому спрашивали.
        # Блокнот мог быть удалён, пока шёл ответ → не роняем поток.
        try:
            nb.append_chat_turn("assistant", result.answer, result.citations)
        except Exception as exc:  # noqa: BLE001
            self._append_log_line(f"[NB] не удалось сохранить ответ: {sanitize_for_log(str(exc))}", "error")
        # В UI дорисовываем только если пользователь не успел переключить блокнот.
        if self._nb_current is None or self._nb_current.id != nb.id:
            return
        try:
            placeholder.destroy()
        except Exception:
            pass
        self._nb_add_message_row("assistant", result.answer, result.citations,
                                 contexts=result.contexts, refused=result.refused)
        if result.refused:
            self._nb_set_status("В источниках нет ответа", "#f59e0b")
        else:
            self._nb_set_status("Ответ готов", "lightgreen")

    def _nb_render_chat(self) -> None:
        for w in self._nb_chat_frame.winfo_children():
            w.destroy()
        nb = self._nb_current
        if nb is None:
            return
        history = nb.load_chat()
        if not history:
            ctk.CTkLabel(self._nb_chat_frame,
                         text="Задайте вопрос — отвечу строго по источникам этого блокнота,\n"
                              "со ссылками [N] на фрагменты. Если ответа в источниках нет — скажу честно.",
                         text_color=_HINT, justify="left").pack(anchor="w", padx=10, pady=10)
            return
        for turn in history:
            self._nb_add_message_row(
                str(turn.get("role") or "assistant"),
                str(turn.get("content") or ""),
                list(turn.get("citations") or []),
            )

    def _nb_add_message_row(
        self, role: str, content: str, citations: list[dict[str, Any]],
        *, contexts: list[dict[str, Any]] | None = None,
        placeholder: bool = False, refused: bool = False,
    ) -> ctk.CTkFrame:
        is_user = role == "user"
        outer = ctk.CTkFrame(self._nb_chat_frame, fg_color="transparent")
        outer.pack(fill="x", padx=6, pady=4)
        ctk.CTkLabel(outer, text="Вы" if is_user else "Ассистент", text_color=_MUTED,
                     font=ctk.CTkFont(size=11)).pack(anchor="w")
        bubble = ctk.CTkFrame(outer, fg_color=_USER_BUBBLE if is_user else _BOT_BUBBLE,
                              corner_radius=12)
        bubble.pack(fill="x", anchor="w")
        text_color = "white" if is_user else ("#111827", "#f3f4f6")
        ctk.CTkLabel(bubble, text=content, wraplength=560, justify="left",
                     text_color=text_color, anchor="w").pack(anchor="w", padx=12, pady=9, fill="x")

        if citations and not placeholder:
            chips = ctk.CTkFrame(outer, fg_color="transparent")
            chips.pack(anchor="w", pady=(3, 0))
            ctk.CTkLabel(chips, text="Источники:", text_color=_MUTED,
                         font=ctk.CTkFont(size=11)).pack(side="left", padx=(0, 4))
            for cit in citations:
                n = cit.get("n", "?")
                disp = str(cit.get("display") or "источник")
                loc = str(cit.get("locator") or "")
                label = f"[{n}] {disp}" + (f" · {loc}" if loc else "")
                ctk.CTkButton(chips, text=label, height=24, font=ctk.CTkFont(size=11),
                              fg_color="transparent", border_width=1, anchor="w",
                              command=lambda c=cit: self._nb_show_citation(c)).pack(side="left", padx=2)
        self.after(0, self._nb_scroll_chat_to_end)
        return outer

    def _nb_scroll_chat_to_end(self) -> None:
        try:
            self._nb_chat_frame._parent_canvas.yview_moveto(1.0)
        except Exception:
            pass

    def _nb_show_citation(self, citation: dict[str, Any]) -> None:
        top = ctk.CTkToplevel(self)
        top.title(f"Источник [{citation.get('n', '?')}]")
        top.geometry("560x420")
        top.transient(self)
        disp = str(citation.get("display") or "источник")
        path = str(citation.get("source_path") or "")
        loc = str(citation.get("locator") or "")
        title = disp + (f"  ·  {loc}" if loc else "")
        ctk.CTkLabel(top, text=title, font=ctk.CTkFont(size=13, weight="bold"),
                     wraplength=520, justify="left").pack(anchor="w", padx=12, pady=(12, 2))
        ctk.CTkLabel(top, text=path, text_color=_MUTED, wraplength=520,
                     justify="left").pack(anchor="w", padx=12, pady=(0, 8))
        box = ctk.CTkTextbox(top, wrap="word")
        box.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        box.insert("1.0", str(citation.get("quote") or ""))
        box.configure(state="disabled")
        btn_row = ctk.CTkFrame(top, fg_color="transparent")
        btn_row.pack(fill="x", padx=12, pady=(0, 12))
        if path:
            ctk.CTkButton(btn_row, text="Открыть источник", fg_color=_ACCENT, hover_color=_ACCENT_HOVER,
                          command=lambda: self._nb_open_path(path)).pack(side="left")
        ctk.CTkButton(btn_row, text="Закрыть", fg_color="transparent", border_width=1,
                      command=top.destroy).pack(side="right")
        top.after(60, top.lift)

    def _nb_clear_chat(self) -> None:
        nb = self._nb_current
        if nb is None:
            return
        if not messagebox.askyesno("Очистить чат", "Удалить историю чата этого блокнота?"):
            return
        nb.clear_chat()
        self._nb_render_chat()

    # ================================================================== #
    #  Studio
    # ================================================================== #
    def _nb_generate(self, kind: str) -> None:
        nb = self._nb_current
        if not self._nb_require_notebook() or nb is None:
            return
        if not self._nb_check_idle():
            return
        if not nb.has_index:
            self._nb_set_status("Сначала постройте индекс блокнота", "#f59e0b")
            return
        model = self._model_var.get().strip()
        if not model or model.startswith("("):
            self._nb_set_status("Выберите LLM-модель (сайдбар)", "#f59e0b")
            return
        base_url = self._url_var.get().strip() or API_BASE
        api_key = self._api_key_var.get().strip() or API_KEY
        api_mode = self._api_mode_var.get().strip().lower()
        spec = MATERIALS[kind]
        self._nb_set_busy(True)
        self._nb_set_status(f"Генерирую: {spec.title}…")

        def work() -> None:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                path, _content = loop.run_until_complete(
                    generate_material(
                        nb, kind, base_url=base_url, api_key=api_key, chat_model=model,
                        api_mode=api_mode,
                        on_log=lambda m: self._append_log_line(f"[NB studio] {m}", "general"),
                    )
                )
                self._nb_after(lambda: self._nb_set_status(
                    f"Готово: {spec.title} → {Path(path).name}", "lightgreen"))
                self._nb_after(self._nb_render_notes)
            except Exception as exc:  # noqa: BLE001
                safe = sanitize_for_log(str(exc))
                self._nb_after(lambda: self._nb_set_status(f"Ошибка генерации: {safe}", "#f87171"))
            finally:
                loop.close()
                self._nb_after(lambda: self._nb_set_busy(False))

        threading.Thread(target=work, daemon=True).start()

    def _nb_render_notes(self) -> None:
        for w in self._nb_notes_frame.winfo_children():
            w.destroy()
        nb = self._nb_current
        notes = nb.list_notes() if nb is not None else []
        if not notes:
            ctk.CTkLabel(self._nb_notes_frame, text="Материалов пока нет",
                         text_color=_HINT).pack(anchor="w", padx=4, pady=2)
            return
        for note in notes:
            ctk.CTkButton(self._nb_notes_frame, text=f"📝 {note.name}", anchor="w",
                          fg_color=_ROW_BG, hover_color="gray30",
                          command=lambda p=note: self._nb_open_note(p)).pack(fill="x", pady=2)

    def _nb_open_note(self, path: Path) -> None:
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self._nb_set_status(f"Не удалось открыть заметку: {sanitize_for_log(str(exc))}", "#f87171")
            return
        top = ctk.CTkToplevel(self)
        top.title(path.name)
        top.geometry("680x560")
        top.transient(self)
        box = ctk.CTkTextbox(top, wrap="word")
        box.pack(fill="both", expand=True, padx=12, pady=12)
        box.insert("1.0", content)
        bar = ctk.CTkFrame(top, fg_color="transparent")
        bar.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkButton(bar, text="Открыть в системе",
                      command=lambda: self._nb_open_path(str(path))).pack(side="left")
        ctk.CTkButton(bar, text="Закрыть", fg_color="transparent", border_width=1,
                      command=top.destroy).pack(side="right")
        top.after(60, top.lift)

    # ================================================================== #
    #  Утилиты
    # ================================================================== #
    @staticmethod
    def _nb_fmt_ts(ts: int, with_time: bool = False) -> str:
        try:
            d = datetime.fromtimestamp(int(ts))
        except Exception:
            return ""
        return d.strftime("%d.%m.%Y %H:%M" if with_time else "%d.%m.%Y")

    def _nb_after(self, fn) -> None:
        """self.after, устойчивый к закрытию окна (иначе daemon-поток умрёт в
        finally и оставит _nb_busy=True, заморозив UI)."""
        try:
            self.after(0, fn)
        except Exception:
            pass

    def _nb_check_idle(self) -> bool:
        if self._nb_busy:
            self._nb_set_status("Дождитесь завершения текущей операции", "#f59e0b")
            return False
        return True

    def _nb_require_notebook(self) -> bool:
        if self._nb_current is None:
            self._nb_set_status("Сначала откройте или создайте блокнот", "#f59e0b")
            return False
        return True

    def _nb_open_path(self, path: str) -> None:
        p = Path(path)
        if not p.exists():
            self._nb_set_status(f"Файл недоступен: {p.name}", "#f59e0b")
            return
        try:
            if os.name == "nt":
                os.startfile(str(p))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as exc:  # noqa: BLE001
            self._nb_set_status(f"Не удалось открыть: {sanitize_for_log(str(exc))}", "#f87171")

    def _nb_set_busy(self, busy: bool) -> None:
        self._nb_busy = busy
        try:
            self._nb_ask_btn.configure(state="disabled" if busy else "normal")
        except Exception:
            pass

    def _nb_set_status(self, text: str, color: str = _HINT) -> None:
        try:
            self._nb_status.configure(text=text, text_color=color)
        except Exception:
            pass
