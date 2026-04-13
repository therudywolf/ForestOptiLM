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
