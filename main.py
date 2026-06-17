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
Nocturne Data Forge (ForestOptiLM) — точка входа GUI.

Массовая асинхронная обработка файлов и папок через локальные LLM (LM Studio):
текст, PDF, Office, архивы, код, изображения (vision), таблицы, RAG.
"""
import logging
import os
import sys
from pathlib import Path


def _frozen_bootstrap() -> None:
    """Настройка путей при запуске из собранного .exe (PyInstaller).

    - кэш токенайзера tiktoken берём из бандла (offline, без скачивания);
    - кэш/индексы/логи пишем рядом с .exe (в NocturneData), а не внутрь бандла.
    """
    if not getattr(sys, "frozen", False):
        return
    exe_dir = Path(sys.executable).resolve().parent
    data_dir = exe_dir / "NocturneData"
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("NOCTURNE_CACHE_DIR", str(data_dir / ".nocturne_cache"))
    except Exception:
        pass
    # В оконном .exe sys.stdout/stderr = None → любой трейсбэк/print падает в пустоту,
    # а необработанное исключение может уронить процесс. Заворачиваем вывод в файл
    # лога и включаем faulthandler, чтобы даже нативный креш (segfault) оставил след.
    try:
        import faulthandler

        log_path = data_dir / "app.log"
        _log_fh = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")  # noqa: SIM115
        if sys.stdout is None:
            sys.stdout = _log_fh
        if sys.stderr is None:
            sys.stderr = _log_fh
        faulthandler.enable(file=_log_fh)
        # logging.basicConfig при импорте повесил StreamHandler на sys.stdout, который
        # в оконном .exe = None → каждый logger.* падает с "NoneType has no write".
        # Выкидываем сломанные хендлеры и пишем логи в app.log.
        root = logging.getLogger()
        for _h in list(root.handlers):
            root.removeHandler(_h)
        _fh_handler = logging.StreamHandler(_log_fh)
        _fh_handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        root.addHandler(_fh_handler)
        root.setLevel(logging.INFO)
    except Exception:
        pass
    meipass = Path(getattr(sys, "_MEIPASS", exe_dir))
    tk_cache = meipass / "tiktoken_cache"
    if tk_cache.is_dir():
        os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(tk_cache))


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)


def main() -> None:
    _frozen_bootstrap()

    from lmstudio_config import get_connection_defaults, mask_secret

    bu, ak, src = get_connection_defaults()
    logging.getLogger("nocturne").info(
        "LM Studio config: base_url=%s api_key=%s source=%s",
        bu,
        mask_secret(ak),
        src,
    )
    from gui import NocturneApp

    app = NocturneApp()
    app.mainloop()


if __name__ == "__main__":
    main()
