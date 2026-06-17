# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# Generates the Nocturne Data Forge app icon into assets/:
#   icon.ico  (Windows .exe + window)   icon.icns (macOS bundle)
#   icon.png  (512, Linux/.desktop)     icon_1024.png (preview/source)
#
# Concept: a luminous crescent moon ("nocturne" / night) cradling a small
# knowledge-graph of connected nodes (documents → data) with one warm
# forge-ember node, on a deep-indigo squircle. Drawn supersampled for crisp
# antialiasing, no external rasterizer required (Pillow + numpy only).
#
# Run:  py -3.13 scripts/make_icon.py
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

BASE = 1024
S = 4               # supersample factor
W = BASE * S
ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")


def _vgrad(top: tuple[int, int, int], bot: tuple[int, int, int]) -> np.ndarray:
    ys = np.linspace(0.0, 1.0, W, dtype=np.float32)
    out = np.empty((W, W, 3), dtype=np.float32)
    for c in range(3):
        out[:, :, c] = (top[c] + (bot[c] - top[c]) * ys)[:, None]
    return out


def _alpha_from_mask(mask: Image.Image, strength: float) -> Image.Image:
    """RGBA-ready alpha channel scaled from a blurred L mask."""
    a = (np.asarray(mask, dtype=np.float32) * strength).clip(0, 255).astype(np.uint8)
    return Image.fromarray(a, "L")


def build() -> Image.Image:
    # --- background: indigo→near-black vertical gradient + teal radial glow ---
    arr = _vgrad((30, 21, 64), (8, 9, 20))
    gx, gy = int(W * 0.42), int(W * 0.40)
    yy, xx = np.mgrid[0:W, 0:W]
    dist = np.sqrt(((xx - gx) ** 2 + (yy - gy) ** 2).astype(np.float32))
    glow = np.clip(1.0 - dist / (W * 0.58), 0, 1) ** 2
    teal = np.array([36, 196, 214], dtype=np.float32)
    for c in range(3):
        arr[:, :, c] = np.clip(arr[:, :, c] + glow * teal[c] * 0.50, 0, 255)
    base = Image.fromarray(arr.astype(np.uint8), "RGB").convert("RGBA")
    del arr, dist, glow, xx, yy

    # --- crescent moon mask (moon disk minus offset shadow disk) ---
    cx, cy, rm = int(W * 0.45), int(W * 0.45), int(W * 0.30)
    moon = Image.new("L", (W, W), 0)
    ImageDraw.Draw(moon).ellipse([cx - rm, cy - rm, cx + rm, cy + rm], fill=255)
    shadow = Image.new("L", (W, W), 0)
    ox, oy, rs = int(W * 0.090), int(W * -0.015), int(rm * 1.02)
    ImageDraw.Draw(shadow).ellipse(
        [cx + ox - rs, cy + oy - rs, cx + ox + rs, cy + oy + rs], fill=255
    )
    cres_np = (np.asarray(moon, int) - np.asarray(shadow, int)).clip(0, 255).astype(np.uint8)
    cres = Image.fromarray(cres_np, "L")

    # --- moon outer glow (blurred crescent, teal) ---
    glow_mask = cres.filter(ImageFilter.GaussianBlur(W * 0.022))
    glow_rgba = Image.new("RGBA", (W, W), (94, 234, 212, 0))
    glow_rgba.putalpha(_alpha_from_mask(glow_mask, 0.85))
    base.alpha_composite(glow_rgba)

    # --- moon body (pale-lavender → violet vertical gradient) ---
    mfill = _vgrad((247, 245, 255), (188, 172, 255)).astype(np.uint8)
    moon_rgb = Image.fromarray(mfill, "RGB").convert("RGBA")
    moon_layer = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    moon_layer.paste(moon_rgb, (0, 0), cres)
    base.alpha_composite(moon_layer)

    # --- knowledge graph (nodes + edges) in the crescent opening ---
    nodes = [  # (x, y, r, color)
        (0.620, 0.395, 0.030, (125, 232, 255)),
        (0.720, 0.520, 0.026, (125, 232, 255)),
        (0.585, 0.620, 0.024, (125, 232, 255)),
        (0.760, 0.355, 0.020, (251, 191, 36)),   # forge ember (amber)
        (0.665, 0.700, 0.018, (167, 139, 250)),  # violet accent
    ]
    edges = [(0, 1), (0, 3), (1, 2), (1, 4), (2, 4)]
    graph = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    gd = ImageDraw.Draw(graph)
    for a, b in edges:
        x1, y1 = nodes[a][0] * W, nodes[a][1] * W
        x2, y2 = nodes[b][0] * W, nodes[b][1] * W
        gd.line([(x1, y1), (x2, y2)], fill=(125, 232, 255, 150), width=int(W * 0.006))
    for x, y, r, col in nodes:
        cxp, cyp, rp = x * W, y * W, r * W
        gd.ellipse([cxp - rp, cyp - rp, cxp + rp, cyp + rp], fill=col + (255,))
    # soft glow for the graph
    graph_glow = graph.filter(ImageFilter.GaussianBlur(W * 0.012))
    base.alpha_composite(graph_glow)
    base.alpha_composite(graph)

    # --- top gloss highlight ---
    gloss = Image.new("L", (W, W), 0)
    ImageDraw.Draw(gloss).ellipse(
        [int(W * 0.10), int(-W * 0.45), int(W * 0.90), int(W * 0.30)], fill=255
    )
    gloss = gloss.filter(ImageFilter.GaussianBlur(W * 0.03))
    gloss_rgba = Image.new("RGBA", (W, W), (255, 255, 255, 0))
    gloss_rgba.putalpha(_alpha_from_mask(gloss, 0.10))
    base.alpha_composite(gloss_rgba)

    # --- squircle (rounded-rect) clip + subtle inner border ---
    mask = Image.new("L", (W, W), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, W - 1, W - 1], radius=int(W * 0.235), fill=255
    )
    border = Image.new("RGBA", (W, W), (0, 0, 0, 0))
    ImageDraw.Draw(border).rounded_rectangle(
        [int(W * 0.006), int(W * 0.006), int(W * 0.994), int(W * 0.994)],
        radius=int(W * 0.230), outline=(255, 255, 255, 28), width=int(W * 0.004)
    )
    base.alpha_composite(border)
    base.putalpha(mask)
    return base


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    full = build().resize((BASE, BASE), Image.LANCZOS)
    full.save(os.path.join(ASSETS, "icon_1024.png"))
    full.resize((512, 512), Image.LANCZOS).save(os.path.join(ASSETS, "icon.png"))
    full.save(
        os.path.join(ASSETS, "icon.ico"),
        sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)],
    )
    try:
        full.save(os.path.join(ASSETS, "icon.icns"))
    except Exception as exc:  # noqa: BLE001
        print(f"icns skipped: {exc}")
    print(f"icons written to {ASSETS}")


if __name__ == "__main__":
    main()
