# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
# ForestOptiLM is free software: you can redistribute it and/or modify it
# under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
"""
Material Design 3 design tokens for the CustomTkinter UI (dark theme).

CustomTkinter is not a Material component library, so this maps MD3's *system*
— colour roles, type scale, shape scale, state layers, 4dp spacing — onto the
widget properties Tk exposes (``fg_color`` / ``text_color`` / ``hover_color`` /
``border_color`` / ``corner_radius`` / fonts / padding). The result follows MD3
principles (tonal surfaces, role-based colour, consistent shape & rhythm) within
Tk's limits.

Pure data + small helpers so the palette is unit-testable without a display; the
real app and the headless preview harness (``tools/ui_preview.py``) import the
SAME tokens, so a screenshot of the preview faithfully reflects the app.

Seed: indigo-violet (continuity with the previous brand ``#7c6cf0``).
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
#  Colour roles — MD3 dark scheme (indigo-violet seed)
# --------------------------------------------------------------------------- #
PRIMARY = "#BEC2FF"
PRIMARY_HOVER = "#D2D4FF"          # state layer ≈ +8% lighter
ON_PRIMARY = "#272B5F"
PRIMARY_CONTAINER = "#3E4178"
PRIMARY_CONTAINER_HOVER = "#494C86"
ON_PRIMARY_CONTAINER = "#E1E0FF"

SECONDARY = "#C6C4DD"
ON_SECONDARY = "#2E2F42"
SECONDARY_CONTAINER = "#454559"
SECONDARY_CONTAINER_HOVER = "#505064"
ON_SECONDARY_CONTAINER = "#E2E0F9"

TERTIARY = "#EAB9D2"
ON_TERTIARY = "#472639"
TERTIARY_CONTAINER = "#603C50"
ON_TERTIARY_CONTAINER = "#FFD8EB"

# Neutral surface ladder (MD3 dark tonal surfaces, low→high elevation)
SURFACE = "#131318"
SURFACE_DIM = "#131318"
SURFACE_BRIGHT = "#39383F"
SURFACE_CONTAINER_LOWEST = "#0E0E13"
SURFACE_CONTAINER_LOW = "#1B1B21"
SURFACE_CONTAINER = "#1F1F25"
SURFACE_CONTAINER_HIGH = "#2A2930"
SURFACE_CONTAINER_HIGHEST = "#35343B"

ON_SURFACE = "#E5E1E9"             # primary text
ON_SURFACE_VARIANT = "#C8C5D0"     # secondary / muted text
OUTLINE = "#928F9A"                # borders / dividers (prominent)
OUTLINE_VARIANT = "#46464F"        # subtle dividers

# Status / semantic (error is MD3; success/warning are app extensions in MD3 tone)
ERROR = "#FFB4AB"
ON_ERROR = "#690005"
ERROR_CONTAINER = "#93000A"
SUCCESS = "#7FD8A0"
WARNING = "#F4C25A"

# --------------------------------------------------------------------------- #
#  Shape scale (corner radius, dp) — MD3 tokens
# --------------------------------------------------------------------------- #
SHAPE_XS = 4
SHAPE_SM = 8
SHAPE_MD = 12
SHAPE_LG = 16
SHAPE_XL = 28
SHAPE_FULL = 1000

# Component shape (semantic aliases used across the app)
RADIUS_BUTTON = 20      # pill-ish filled/tonal buttons
RADIUS_CARD = SHAPE_LG  # cards / panels
RADIUS_INPUT = SHAPE_MD
RADIUS_CHIP = SHAPE_SM
RADIUS_BUBBLE = SHAPE_LG

# --------------------------------------------------------------------------- #
#  Spacing (4dp grid)
# --------------------------------------------------------------------------- #
SPACE_XS = 4
SPACE_SM = 8
SPACE_MD = 12
SPACE_LG = 16
SPACE_XL = 24

# --------------------------------------------------------------------------- #
#  Type scale — MD3 roles → (size, weight). Desktop-tuned subset.
# --------------------------------------------------------------------------- #
TYPE: dict[str, tuple[int, str]] = {
    "display": (30, "bold"),
    "headline": (24, "bold"),
    "title-lg": (20, "bold"),
    "title": (16, "bold"),
    "title-sm": (14, "bold"),
    "body-lg": (15, "normal"),
    "body": (14, "normal"),
    "label": (13, "normal"),
    "label-sm": (11, "normal"),
}


def type_spec(role: str) -> tuple[int, str]:
    """(size, weight) for an MD3 type role; falls back to 'body'."""
    return TYPE.get(role, TYPE["body"])


def font(role: str = "body"):
    """Create a ``CTkFont`` for a type role. Must be called after a CTk root
    exists (lazy import so this module stays importable without a display)."""
    import customtkinter as ctk
    size, weight = type_spec(role)
    return ctk.CTkFont(size=size, weight=weight)


def apply() -> None:
    """Configure CustomTkinter globally for the MD3 dark theme.

    Patches ``ThemeManager.theme`` so EVERY widget picks up MD3 shape + surface
    colours by default (not just ones styled per-widget): rounded corners, tonal
    surfaces, role-based entry/label/scrollbar colours. Per-widget accent colours
    (filled buttons etc.) are still set explicitly by the app.

    Default button text is kept LIGHT (on-surface) so transparent/outlined buttons
    stay readable; filled accent buttons set their own dark on-primary text.
    """
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    t = ctk.ThemeManager.theme

    def dual(v):
        return [v, v]  # CTk stores [light, dark]; app is dark-only → same value

    def patch(key: str, **vals) -> None:
        if key in t and isinstance(t[key], dict):
            for k, v in vals.items():
                if k in t[key]:
                    t[key][k] = v

    patch("CTk", fg_color=dual(SURFACE))
    patch("CTkToplevel", fg_color=dual(SURFACE))
    patch("CTkFrame", fg_color=dual(SURFACE_CONTAINER), top_fg_color=dual(SURFACE_CONTAINER_HIGH),
          border_color=dual(OUTLINE_VARIANT), corner_radius=RADIUS_CARD)
    patch("CTkButton", fg_color=dual(PRIMARY_CONTAINER), hover_color=dual(PRIMARY_CONTAINER_HOVER),
          text_color=dual(ON_SURFACE), text_color_disabled=dual(ON_SURFACE_VARIANT),
          border_color=dual(OUTLINE), corner_radius=RADIUS_BUTTON)
    patch("CTkLabel", text_color=dual(ON_SURFACE), fg_color="transparent")
    patch("CTkEntry", fg_color=dual(SURFACE_CONTAINER), border_color=dual(OUTLINE_VARIANT),
          text_color=dual(ON_SURFACE), placeholder_text_color=dual(ON_SURFACE_VARIANT),
          corner_radius=RADIUS_INPUT)
    patch("CTkTextbox", fg_color=dual(SURFACE_CONTAINER), border_color=dual(OUTLINE_VARIANT),
          text_color=dual(ON_SURFACE), scrollbar_button_color=dual(OUTLINE_VARIANT),
          corner_radius=RADIUS_INPUT)
    patch("CTkCheckBox", fg_color=dual(PRIMARY), hover_color=dual(PRIMARY_HOVER),
          checkmark_color=dual(ON_PRIMARY), text_color=dual(ON_SURFACE), border_color=dual(OUTLINE),
          corner_radius=SHAPE_XS)
    patch("CTkSwitch", progress_color=dual(PRIMARY), button_color=dual(ON_SURFACE_VARIANT),
          button_hover_color=dual(ON_SURFACE), text_color=dual(ON_SURFACE))
    patch("CTkOptionMenu", fg_color=dual(SURFACE_CONTAINER_HIGH), button_color=dual(PRIMARY_CONTAINER),
          button_hover_color=dual(PRIMARY_CONTAINER_HOVER), text_color=dual(ON_SURFACE),
          corner_radius=RADIUS_INPUT)
    patch("CTkComboBox", fg_color=dual(SURFACE_CONTAINER), border_color=dual(OUTLINE_VARIANT),
          button_color=dual(PRIMARY_CONTAINER), button_hover_color=dual(PRIMARY_CONTAINER_HOVER),
          text_color=dual(ON_SURFACE), corner_radius=RADIUS_INPUT)
    patch("CTkScrollbar", button_color=dual(OUTLINE_VARIANT), button_hover_color=dual(OUTLINE))
    patch("CTkScrollableFrame", label_fg_color=dual(SURFACE_CONTAINER_HIGH))
    patch("CTkSegmentedButton", fg_color=dual(SURFACE_CONTAINER_HIGH),
          selected_color=dual(PRIMARY_CONTAINER), selected_hover_color=dual(PRIMARY_CONTAINER_HOVER),
          unselected_color=dual(SURFACE_CONTAINER_HIGH), unselected_hover_color=dual(SURFACE_BRIGHT),
          text_color=dual(ON_SURFACE), corner_radius=RADIUS_BUTTON)
    patch("CTkProgressBar", fg_color=dual(SURFACE_CONTAINER_HIGHEST), progress_color=dual(PRIMARY))
    patch("DropdownMenu", fg_color=dual(SURFACE_CONTAINER_HIGH), hover_color=dual(SURFACE_BRIGHT),
          text_color=dual(ON_SURFACE))


# --------------------------------------------------------------------------- #
#  Component recipes — keyword dicts for common CTk widgets (consistency)
# --------------------------------------------------------------------------- #
def button_filled() -> dict:
    """Primary CTA: filled with the primary tone + dark on-primary text (MD3)."""
    return dict(fg_color=PRIMARY, hover_color=PRIMARY_HOVER, text_color=ON_PRIMARY,
                corner_radius=RADIUS_BUTTON)


def button_tonal() -> dict:
    """Secondary action: tonal (primary-container) — softer than filled."""
    return dict(fg_color=PRIMARY_CONTAINER, hover_color=PRIMARY_CONTAINER_HOVER,
                text_color=ON_PRIMARY_CONTAINER, corner_radius=RADIUS_BUTTON)


def button_outlined() -> dict:
    """Tertiary action: outlined / transparent with an outline."""
    return dict(fg_color="transparent", hover_color=SURFACE_CONTAINER_HIGH,
                text_color=ON_SURFACE, border_color=OUTLINE, border_width=1,
                corner_radius=RADIUS_BUTTON)


def button_text() -> dict:
    """Low-emphasis text button."""
    return dict(fg_color="transparent", hover_color=SURFACE_CONTAINER_HIGH,
                text_color=PRIMARY, corner_radius=RADIUS_BUTTON)


def button_danger() -> dict:
    """Destructive action (error tone)."""
    return dict(fg_color=ERROR_CONTAINER, hover_color="#A8000C",
                text_color="#FFDAD6", corner_radius=RADIUS_BUTTON)
