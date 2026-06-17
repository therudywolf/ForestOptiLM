# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
"""Regression tests for LM Studio URL normalization and UI runtime state."""
from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import pytest

import lmstudio_config
from lmstudio_config import (
    _candidate_paths,
    load_ui_runtime_state,
    lmstudio_root_url,
    normalize_lmstudio_base_url,
    save_ui_runtime_state,
)


class TestConfigCandidatePaths(unittest.TestCase):
    def test_frozen_looks_next_to_exe(self) -> None:
        # Регрессия: упакованный .exe должен читать lmstudio.json РЯДОМ с собой,
        # а не только внутри read-only бандла (_internal) — иначе сервер/ключ
        # пользователя не подхватываются.
        with patch.object(lmstudio_config.sys, "frozen", True, create=True), \
             patch.object(lmstudio_config.sys, "executable", r"C:\app\NocturneDataForge.exe"):
            paths = [str(p) for p in _candidate_paths()]
        self.assertTrue(any(p.endswith(r"\app\lmstudio.json") for p in paths), paths)

    def test_non_frozen_uses_repo_local(self) -> None:
        with patch.object(lmstudio_config.sys, "frozen", False, create=True):
            paths = [str(p) for p in _candidate_paths()]
        self.assertTrue(any(p.endswith("lmstudio.json") for p in paths))


class TestLMStudioUrlNormalization(unittest.TestCase):
    def test_root_url_gets_openai_base_suffix(self) -> None:
        self.assertEqual(
            normalize_lmstudio_base_url("http://127.0.0.1:1234"),
            "http://127.0.0.1:1234/v1",
        )

    def test_native_base_url_maps_to_openai_base(self) -> None:
        self.assertEqual(
            normalize_lmstudio_base_url("http://127.0.0.1:1234/api/v1"),
            "http://127.0.0.1:1234/v1",
        )

    def test_copied_endpoint_url_maps_to_root(self) -> None:
        self.assertEqual(
            lmstudio_root_url("http://127.0.0.1:1234/v1/chat/completions"),
            "http://127.0.0.1:1234",
        )
        self.assertEqual(
            lmstudio_root_url("http://127.0.0.1:1234/api/v1/models/load"),
            "http://127.0.0.1:1234",
        )
        self.assertEqual(
            lmstudio_root_url("http://127.0.0.1:1234/api/v1/models"),
            "http://127.0.0.1:1234",
        )


@pytest.fixture()
def _runtime_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the UI runtime state file into a temp directory."""
    target = tmp_path / ".local" / "ui_runtime.json"
    monkeypatch.setattr(lmstudio_config, "_runtime_ui_path", lambda: target)
    return target


class TestScoutSettingsRoundTrip:
    def test_scout_settings_persist_with_numeric_threshold(
        self, _runtime_path: Path
    ) -> None:
        save_ui_runtime_state(
            {
                "scout_mode": True,
                "scout_threshold": 0.5,
                "selected_scout_model": "some-model",
            }
        )
        loaded = load_ui_runtime_state()
        assert loaded["scout_mode"] is True
        assert isinstance(loaded["scout_threshold"], float)
        assert loaded["scout_threshold"] == 0.5
        assert loaded["selected_scout_model"] == "some-model"

    def test_scout_threshold_persists_when_passed_as_string(
        self, _runtime_path: Path
    ) -> None:
        # gui.py's _collect_runtime_state() emits scout_threshold as a string.
        save_ui_runtime_state(
            {
                "scout_mode": True,
                "scout_threshold": "0.5",
                "selected_scout_model": "scout-x",
            }
        )
        loaded = load_ui_runtime_state()
        assert loaded["scout_mode"] is True
        assert isinstance(loaded["scout_threshold"], float)
        assert loaded["scout_threshold"] == 0.5
        assert loaded["selected_scout_model"] == "scout-x"

    def test_scout_defaults_when_file_missing(self, _runtime_path: Path) -> None:
        loaded = load_ui_runtime_state()
        assert loaded["scout_mode"] is False
        assert loaded["scout_threshold"] == 0.35
        assert loaded["selected_scout_model"] == ""

    def test_scout_threshold_clamped_to_range(self, _runtime_path: Path) -> None:
        save_ui_runtime_state({"scout_threshold": "5.0"})
        assert load_ui_runtime_state()["scout_threshold"] == 1.0
        save_ui_runtime_state({"scout_threshold": -1.0})
        assert load_ui_runtime_state()["scout_threshold"] == 0.0

    def test_scout_threshold_falls_back_on_garbage(self, _runtime_path: Path) -> None:
        save_ui_runtime_state({"scout_threshold": "not-a-number"})
        assert load_ui_runtime_state()["scout_threshold"] == 0.35


if __name__ == "__main__":
    unittest.main()
