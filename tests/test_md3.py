# SPDX-License-Identifier: AGPL-3.0-or-later
"""Material Design 3 tokens: palette validity, type scale, component recipes, theme patch."""
from __future__ import annotations

import re
import unittest

import md3


_HEX = re.compile(r"^#[0-9A-Fa-f]{6}$")


class TestPalette(unittest.TestCase):
    def test_core_roles_are_valid_hex(self) -> None:
        for name in ("PRIMARY", "ON_PRIMARY", "PRIMARY_CONTAINER", "ON_PRIMARY_CONTAINER",
                     "SECONDARY_CONTAINER", "TERTIARY_CONTAINER", "SURFACE", "SURFACE_CONTAINER",
                     "SURFACE_CONTAINER_HIGH", "ON_SURFACE", "ON_SURFACE_VARIANT",
                     "OUTLINE", "OUTLINE_VARIANT", "ERROR", "ERROR_CONTAINER",
                     "SUCCESS", "WARNING", "PRIMARY_HOVER"):
            val = getattr(md3, name)
            self.assertRegex(val, _HEX, f"{name}={val!r} is not #RRGGBB")

    def test_surface_ladder_distinct(self) -> None:
        ladder = [md3.SURFACE_CONTAINER_LOWEST, md3.SURFACE_CONTAINER_LOW, md3.SURFACE_CONTAINER,
                  md3.SURFACE_CONTAINER_HIGH, md3.SURFACE_CONTAINER_HIGHEST]
        self.assertEqual(len(set(ladder)), len(ladder))  # each elevation level differs

    def test_shape_scale_monotonic(self) -> None:
        self.assertTrue(md3.SHAPE_XS < md3.SHAPE_SM < md3.SHAPE_MD < md3.SHAPE_LG < md3.SHAPE_XL)
        self.assertGreaterEqual(md3.RADIUS_BUTTON, md3.SHAPE_MD)


class TestType(unittest.TestCase):
    def test_type_spec_shape(self) -> None:
        size, weight = md3.type_spec("headline")
        self.assertIsInstance(size, int)
        self.assertIn(weight, ("normal", "bold"))

    def test_type_spec_fallback(self) -> None:
        self.assertEqual(md3.type_spec("does-not-exist"), md3.TYPE["body"])

    def test_type_scale_descends(self) -> None:
        self.assertGreater(md3.type_spec("display")[0], md3.type_spec("body")[0])
        self.assertGreater(md3.type_spec("title")[0], md3.type_spec("label-sm")[0])


class TestRecipes(unittest.TestCase):
    def test_filled_has_dark_text_on_light_primary(self) -> None:
        r = md3.button_filled()
        self.assertEqual(r["fg_color"], md3.PRIMARY)
        self.assertEqual(r["text_color"], md3.ON_PRIMARY)  # contrast: dark text on light fill
        self.assertEqual(r["corner_radius"], md3.RADIUS_BUTTON)

    def test_outlined_is_transparent_with_border(self) -> None:
        r = md3.button_outlined()
        self.assertEqual(r["fg_color"], "transparent")
        self.assertEqual(r["border_width"], 1)
        self.assertEqual(r["text_color"], md3.ON_SURFACE)  # light text stays readable

    def test_all_recipes_return_corner_radius(self) -> None:
        for fn in (md3.button_filled, md3.button_tonal, md3.button_outlined,
                   md3.button_text, md3.button_danger):
            self.assertIn("corner_radius", fn())


class TestApply(unittest.TestCase):
    def test_apply_patches_theme_without_error(self) -> None:
        try:
            import customtkinter as ctk
        except Exception:
            self.skipTest("customtkinter not installed")
        md3.apply()
        t = ctk.ThemeManager.theme
        # Button corner radius is now the MD3 token, surfaces are MD3 roles.
        self.assertEqual(t["CTkButton"]["corner_radius"], md3.RADIUS_BUTTON)
        self.assertEqual(t["CTkFrame"]["fg_color"], [md3.SURFACE_CONTAINER, md3.SURFACE_CONTAINER])
        # Default button text stays light → transparent/outlined buttons stay readable.
        self.assertEqual(t["CTkButton"]["text_color"], [md3.ON_SURFACE, md3.ON_SURFACE])


if __name__ == "__main__":
    unittest.main()
