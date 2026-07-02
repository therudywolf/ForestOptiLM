# SPDX-License-Identifier: AGPL-3.0-or-later
"""Headless MD3 UI preview harness.

Builds representative app screens using the shared ``md3`` tokens and either
shows the window or screenshots it (via ``scrot``) so the design can be iterated
under Xvfb WITHOUT touching the user's real screen. Run via ``tools/ui_shot.sh``.

    python tools/ui_preview.py --screen workspace --shot out.png
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import customtkinter as ctk  # noqa: E402

import md3 as m  # noqa: E402


def _chip(parent, text, *, on=False):
    return ctk.CTkLabel(
        parent, text=text, fg_color=(m.PRIMARY_CONTAINER if on else m.SURFACE_CONTAINER_HIGH),
        text_color=(m.ON_PRIMARY_CONTAINER if on else m.ON_SURFACE_VARIANT),
        corner_radius=m.RADIUS_CHIP, font=m.font("label-sm"), padx=10, pady=3)


def _card(parent, emoji, title, desc, stats, accent):
    card = ctk.CTkFrame(parent, fg_color=m.SURFACE_CONTAINER, corner_radius=m.RADIUS_CARD,
                        border_width=1, border_color=m.OUTLINE_VARIANT)
    cover = ctk.CTkFrame(card, fg_color=accent, corner_radius=m.RADIUS_CARD, height=84)
    cover.pack(fill="x", padx=4, pady=(4, 0))
    cover.pack_propagate(False)
    ctk.CTkLabel(cover, text=emoji, font=ctk.CTkFont(size=34)).pack(expand=True)
    ctk.CTkLabel(card, text=title, font=m.font("title"), text_color=m.ON_SURFACE,
                 anchor="w").pack(fill="x", padx=14, pady=(10, 2))
    ctk.CTkLabel(card, text=desc, font=m.font("body"), text_color=m.ON_SURFACE_VARIANT,
                 anchor="w", justify="left", wraplength=240).pack(fill="x", padx=14)
    row = ctk.CTkFrame(card, fg_color="transparent")
    row.pack(fill="x", padx=12, pady=12)
    for s in stats:
        _chip(row, s).pack(side="left", padx=3)
    return card


def _topbar(root):
    bar = ctk.CTkFrame(root, fg_color=m.SURFACE_CONTAINER_LOW, corner_radius=0, height=58)
    bar.pack(fill="x")
    bar.pack_propagate(False)
    ctk.CTkLabel(bar, text="🐺  Nocturne Data Forge", font=m.font("title-lg"),
                 text_color=m.ON_SURFACE).pack(side="left", padx=20)
    tabs = ctk.CTkFrame(bar, fg_color="transparent")
    tabs.pack(side="left", padx=24)
    for i, t in enumerate(["📓 Блокноты", "📊 Результат", "📜 Логи", "🔎 RAG"]):
        on = i == 0
        ctk.CTkButton(tabs, text=t, width=120, height=36,
                      fg_color=(m.PRIMARY_CONTAINER if on else "transparent"),
                      hover_color=m.SURFACE_CONTAINER_HIGH,
                      text_color=(m.ON_PRIMARY_CONTAINER if on else m.ON_SURFACE_VARIANT),
                      corner_radius=m.RADIUS_BUTTON, font=m.font("label")).pack(side="left", padx=3)


def screen_archive(root):
    _topbar(root)
    body = ctk.CTkFrame(root, fg_color=m.SURFACE)
    body.pack(fill="both", expand=True)
    head = ctk.CTkFrame(body, fg_color="transparent")
    head.pack(fill="x", padx=24, pady=(20, 6))
    ctk.CTkLabel(head, text="Исследования", font=m.font("headline"),
                 text_color=m.ON_SURFACE).pack(side="left")
    ctk.CTkButton(head, text="＋ Новое исследование", height=40,
                  **m.button_filled(), font=m.font("label")).pack(side="right")
    ctk.CTkEntry(body, placeholder_text="🔍  Поиск по блокнотам…", height=42,
                 fg_color=m.SURFACE_CONTAINER, border_color=m.OUTLINE_VARIANT,
                 corner_radius=m.RADIUS_INPUT, font=m.font("body"),
                 text_color=m.ON_SURFACE).pack(fill="x", padx=24, pady=(4, 16))
    grid = ctk.CTkFrame(body, fg_color="transparent")
    grid.pack(fill="both", expand=True, padx=20)
    cards = [
        ("📘", "Архитектура ACME", "Подсистемы, виртуальные машины, схемы развёртывания и связи между сервисами.",
         ["📄 24 источ.", "◆ 1950 фрагм.", "вчера"], m.PRIMARY_CONTAINER),
        ("🛡", "Demo-логи", "Журналы и таблицы статусов электронной части, переписка и решения.",
         ["📄 38 источ.", "◆ 920 фрагм.", "2 дня назад"], m.TERTIARY_CONTAINER),
        ("📊", "Отчёты Q2", "Квартальные сводки, метрики и выводы по проектам.",
         ["📄 12 источ.", "◆ 430 фрагм.", "сегодня"], m.SECONDARY_CONTAINER),
    ]
    for i, c in enumerate(cards):
        card = _card(grid, *c)
        card.grid(row=0, column=i, padx=8, pady=8, sticky="nsew")
        grid.grid_columnconfigure(i, weight=1)


def _bubble(parent, who, text, *, user=False, cites=None):
    outer = ctk.CTkFrame(parent, fg_color="transparent")
    outer.pack(fill="x", padx=8, pady=6)
    ctk.CTkLabel(outer, text=who, font=m.font("label-sm"),
                 text_color=m.ON_SURFACE_VARIANT).pack(anchor="w")
    bub = ctk.CTkFrame(outer, fg_color=(m.PRIMARY_CONTAINER if user else m.SURFACE_CONTAINER_HIGH),
                       corner_radius=m.RADIUS_BUBBLE)
    bub.pack(fill="x", anchor="w")
    ctk.CTkLabel(bub, text=text, font=m.font("body"), justify="left", anchor="w",
                 wraplength=520, text_color=(m.ON_PRIMARY_CONTAINER if user else m.ON_SURFACE)
                 ).pack(anchor="w", padx=14, pady=10, fill="x")
    if cites:
        row = ctk.CTkFrame(outer, fg_color="transparent")
        row.pack(anchor="w", pady=(4, 0))
        ctk.CTkLabel(row, text="Источники:", font=m.font("label-sm"),
                     text_color=m.ON_SURFACE_VARIANT).pack(side="left", padx=(0, 4))
        for c in cites:
            ctk.CTkButton(row, text=c, height=26, **m.button_outlined(),
                          font=m.font("label-sm")).pack(side="left", padx=3)
        ctk.CTkButton(outer, text="📌 В знания", height=26, **m.button_text(),
                      font=m.font("label-sm")).pack(anchor="w", pady=(4, 0))


def screen_workspace(root):
    _topbar(root)
    ws = ctk.CTkFrame(root, fg_color=m.SURFACE)
    ws.pack(fill="both", expand=True)
    # Sources pane
    left = ctk.CTkFrame(ws, fg_color=m.SURFACE_CONTAINER_LOW, corner_radius=m.RADIUS_CARD, width=240)
    left.pack(side="left", fill="y", padx=(16, 8), pady=16)
    left.pack_propagate(False)
    ctk.CTkLabel(left, text="Источники", font=m.font("title"), text_color=m.ON_SURFACE
                 ).pack(anchor="w", padx=14, pady=(14, 8))
    for name in ["doc-a.docx", "doc-b.pdf", "data.xlsx", "chat.html"]:
        ctk.CTkLabel(left, text=f"📄  {name}", font=m.font("body"), anchor="w",
                     text_color=m.ON_SURFACE_VARIANT).pack(fill="x", padx=14, pady=3)
    ctk.CTkButton(left, text="🖼 Описывать картинки", **m.button_outlined(),
                  font=m.font("label-sm"), height=32).pack(fill="x", padx=12, pady=(10, 4))
    ctk.CTkButton(left, text="🔨 Построить индекс", **m.button_filled(),
                  font=m.font("label"), height=38).pack(fill="x", padx=12, pady=4)
    # Chat pane
    mid = ctk.CTkFrame(ws, fg_color="transparent")
    mid.pack(side="left", fill="both", expand=True, pady=16)
    bar = ctk.CTkFrame(mid, fg_color="transparent")
    bar.pack(fill="x")
    ctk.CTkLabel(bar, text="Чат по источникам", font=m.font("title"),
                 text_color=m.ON_SURFACE).pack(side="left")
    _chip(bar, "🎯 Точный поиск", on=True).pack(side="right", padx=(0, 8))
    ctk.CTkOptionMenu(bar, values=["Авто", "Вкл", "Выкл"], width=84,
                      font=m.font("small")).pack(side="right", padx=(0, 6))
    ctk.CTkLabel(bar, text="🔬 Глубокий анализ", font=m.font("small"),
                 text_color=m.ON_SURFACE).pack(side="right", padx=(0, 4))
    chat = ctk.CTkFrame(mid, fg_color=m.SURFACE_CONTAINER_LOWEST, corner_radius=m.RADIUS_CARD)
    chat.pack(fill="both", expand=True, pady=8)
    _bubble(chat, "Вы", "На каких ВМ развёрнута подсистема Alpha?", user=True)
    _bubble(chat, "Ассистент", "Подсистема Alpha развёрнута на host-07 и host-08; обе используют PostgreSQL [1][2].",
            cites=["[1] doc-a.docx", "[2] doc-b.pdf · стр. 3"])
    inrow = ctk.CTkFrame(mid, fg_color="transparent")
    inrow.pack(fill="x")
    ctk.CTkEntry(inrow, placeholder_text="Спросите по источникам…", height=46,
                 fg_color=m.SURFACE_CONTAINER, border_color=m.OUTLINE_VARIANT,
                 corner_radius=m.RADIUS_INPUT, font=m.font("body"),
                 text_color=m.ON_SURFACE).pack(side="left", fill="x", expand=True, padx=(0, 8))
    ctk.CTkButton(inrow, text="Спросить ▶", width=120, height=46, **m.button_filled(),
                  font=m.font("label")).pack(side="left")
    # Studio pane
    right = ctk.CTkFrame(ws, fg_color=m.SURFACE_CONTAINER_LOW, corner_radius=m.RADIUS_CARD, width=230)
    right.pack(side="left", fill="y", padx=(8, 16), pady=16)
    right.pack_propagate(False)
    ctk.CTkLabel(right, text="Studio", font=m.font("title"), text_color=m.ON_SURFACE
                 ).pack(anchor="w", padx=14, pady=(14, 8))
    for t in ["Учебный гайд", "FAQ", "Таймлайн", "Конспект", "Флеш-карточки"]:
        ctk.CTkButton(right, text=t, **m.button_tonal(), anchor="w", height=34,
                      font=m.font("label")).pack(fill="x", padx=12, pady=3)
    ctk.CTkButton(right, text="📚 Скомпилировать знания", **m.button_filled(), anchor="w",
                  height=38, font=m.font("label")).pack(fill="x", padx=12, pady=(10, 3))
    ctk.CTkButton(right, text="🔍 Проверить блокнот", **m.button_outlined(), anchor="w",
                  height=34, font=m.font("label")).pack(fill="x", padx=12, pady=3)


def screen_components(root):
    _topbar(root)
    body = ctk.CTkScrollableFrame(root, fg_color=m.SURFACE)
    body.pack(fill="both", expand=True, padx=24, pady=16)
    ctk.CTkLabel(body, text="Кнопки (MD3)", font=m.font("title-lg"),
                 text_color=m.ON_SURFACE).pack(anchor="w", pady=(4, 8))
    row = ctk.CTkFrame(body, fg_color="transparent")
    row.pack(anchor="w", pady=4)
    ctk.CTkButton(row, text="Filled", **m.button_filled(), font=m.font("label")).pack(side="left", padx=6)
    ctk.CTkButton(row, text="Tonal", **m.button_tonal(), font=m.font("label")).pack(side="left", padx=6)
    ctk.CTkButton(row, text="Outlined", **m.button_outlined(), font=m.font("label")).pack(side="left", padx=6)
    ctk.CTkButton(row, text="Text", **m.button_text(), font=m.font("label")).pack(side="left", padx=6)
    ctk.CTkButton(row, text="⏹ Стоп", **m.button_danger(), font=m.font("label")).pack(side="left", padx=6)
    ctk.CTkLabel(body, text="Статусы", font=m.font("title-lg"),
                 text_color=m.ON_SURFACE).pack(anchor="w", pady=(18, 8))
    srow = ctk.CTkFrame(body, fg_color="transparent")
    srow.pack(anchor="w", pady=4)
    for txt, col in [("Готово", m.SUCCESS), ("Внимание", m.WARNING), ("Ошибка", m.ERROR)]:
        ctk.CTkLabel(srow, text=f"● {txt}", text_color=col, font=m.font("body")).pack(side="left", padx=12)


SCREENS = {"archive": screen_archive, "workspace": screen_workspace, "components": screen_components}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default="workspace", choices=list(SCREENS))
    ap.add_argument("--shot", default="")
    ap.add_argument("--size", default="1280x820")
    args = ap.parse_args()

    m.apply()
    root = ctk.CTk()
    root.geometry(f"{args.size}+0+0")
    root.configure(fg_color=m.SURFACE)
    SCREENS[args.screen](root)
    root.update()
    root.update_idletasks()
    if args.shot:
        time.sleep(0.6)
        subprocess.run(["scrot", "-o", args.shot], check=True)
        root.destroy()
    else:
        root.mainloop()


if __name__ == "__main__":
    main()
