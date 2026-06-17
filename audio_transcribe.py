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
Локальная транскрипция аудио через faster-whisper (опциональная зависимость).

Полностью офлайн: распознаёт речь из аудиофайла в текст с таймкодами, после чего
он индексируется как обычный источник. Синтез речи (TTS) намеренно НЕ делается.

faster-whisper не входит в основные зависимости (тянет ctranslate2 и модель).
Если он не установлен — аудиофайлы просто пропускаются с понятным сообщением.
Установка: ``pip install faster-whisper``. Модель — env ``NOCTURNE_WHISPER_MODEL``
(по умолчанию ``base``); язык — ``NOCTURNE_WHISPER_LANG`` (по умолчанию авто).
"""
from __future__ import annotations

import importlib.util
import logging
import os

logger = logging.getLogger("nocturne")

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".opus", ".aac", ".wma", ".webm"}

_model_cache: dict[str, object] = {}


def is_available() -> bool:
    """Установлен ли faster-whisper."""
    return importlib.util.find_spec("faster_whisper") is not None


def _fmt_ts(seconds: float) -> str:
    s = int(seconds or 0)
    return f"{s // 60:02d}:{s % 60:02d}"


def transcribe(path, model_size: str | None = None, language: str | None = None) -> str:
    """Распознать речь из аудиофайла → текст с таймкодами «[mm:ss] ...».

    Бросает RuntimeError с инструкцией, если faster-whisper не установлен.
    """
    if not is_available():
        raise RuntimeError(
            "Для аудио нужна faster-whisper: pip install faster-whisper "
            "(распознавание локальное, офлайн)."
        )
    from faster_whisper import WhisperModel  # type: ignore[import-not-found]

    size = (model_size or os.getenv("NOCTURNE_WHISPER_MODEL", "base")).strip() or "base"
    lang = language if language is not None else (os.getenv("NOCTURNE_WHISPER_LANG", "").strip() or None)

    model = _model_cache.get(size)
    if model is None:
        compute_type = os.getenv("NOCTURNE_WHISPER_COMPUTE", "int8").strip() or "int8"
        model = WhisperModel(size, device="cpu", compute_type=compute_type)
        _model_cache[size] = model

    segments, info = model.transcribe(str(path), language=lang, vad_filter=True)
    lines: list[str] = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            lines.append(f"[{_fmt_ts(seg.start)}] {text}")
    detected = getattr(info, "language", "") or lang or "?"
    logger.info("audio transcribe: %s lang=%s segments=%s", os.path.basename(str(path)), detected, len(lines))
    return "\n".join(lines)
