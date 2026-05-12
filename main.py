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
Nocturne Data Forge — точка входа.
Массовая асинхронная обработка больших файлов (TXT, PDF, DOCX, CSV, XLSX) через локальные LLM.
"""
import logging
import sys

from lmstudio_config import get_connection_defaults, mask_secret

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)


def main() -> None:
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
