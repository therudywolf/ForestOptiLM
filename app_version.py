# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025 therudywolf <https://github.com/therudywolf>
#
# This file is part of ForestOptiLM / Nocturne Data Forge.
"""Единый источник версии для РАНТАЙМА приложения (GUI/exe).

Держим строку здесь, в корневом модуле, который точно попадает в сборку
(main.py → gui.py → app_version). Пакетные метаданные (forestoptilm/__init__.py,
pyproject.toml) и Windows-ресурс (scripts/version_info.txt) держим синхронно —
при бампе версии правим все, а UI теперь показывает версию, так что рассинхрон
сразу виден на экране.
"""
from __future__ import annotations

APP_VERSION = "0.7.0-beta.31"
