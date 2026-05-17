# Third-Party Notices

ForestOptiLM / Nocturne Data Forge bundles **no vendored source** of these
dependencies; they are installed via `pip` at runtime. This file documents
their typical SPDX/OSI licenses for AGPL compliance and attribution.

| Package | Typical license | Role |
|---------|-----------------|------|
| customtkinter | MIT | GUI toolkit |
| httpx | BSD-3-Clause | HTTP client (LM Studio API) |
| pandas | BSD-3-Clause | Tables / batch export |
| openpyxl | MIT | Excel read/write |
| pdfplumber | MIT | PDF text extraction |
| python-docx | MIT | Word documents |
| tiktoken | MIT | Token counting |
| beautifulsoup4 | MIT | HTML parsing |
| pyyaml | MIT | Run profiles |
| odfpy | Apache-2.0 / GPL (dual) | ODT documents |
| ebooklib | AGPL-3.0 | EPUB (dependency of this project is AGPL-aligned) |
| xlrd | BSD-3-Clause | Legacy XLS |
| striprtf | BSD-3-Clause | RTF |
| numpy | BSD-3-Clause | Numerics (FAISS) |
| faiss-cpu | MIT | Vector index (Meta FAISS) |
| pytest (dev) | MIT | Tests |
| ruff (dev) | MIT | Lint |

Verify exact license strings on your machine:

```bash
pip install pip-licenses
pip-licenses --format=markdown
```

When you distribute a modified version of ForestOptiLM, you must comply with
**AGPL-3.0-or-later** for this project and respect the licenses of dependencies
above.
