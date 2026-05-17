# 🐺 ForestOptiLM — Nocturne Data Forge

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)

Desktop application for bulk asynchronous processing of large files and
folders (including multi‑hundred‑thousand‑token corpora) through local LLMs
(LM Studio and other OpenAI-compatible APIs). Uses Map‑Reduce with an optional
**scout pass** (fast relevance filter) and a **composer** model for merge/reduce.

AGPL v3 Copyleft applies to reuse, modification, and network deployment of derived versions.

Десктопное приложение на Python для массовой асинхронной обработки больших файлов
через локальные LLM (LM Studio и другие OpenAI-совместимые API).

## Features

- **Map-Reduce pipeline** for text documents with structured JSON evidence.
- **Scout pass** — quick relevance scoring before full MAP (for huge folders).
- **Large corpus preset** — scout + smaller chunks + composer in one click.
- **Vision support** for image analysis via multimodal models.
- **RAG** — local FAISS-based retrieval-augmented generation.
- **Batch table processing** for CSV/XLSX with JSON responses.
- **Low VRAM mode** — sequential model loading for constrained hardware.
- **Composer model** — optional separate model for merge/reduce/refine phases.
- **SQLite cache** for MAP checkpoint resumption.
- **Dark theme GUI** (CustomTkinter).

## Installation

```bash
git clone https://github.com/therudywolf/ForestOptiLM.git
cd ForestOptiLM
pip install -r requirements.txt
```

### Docker

```bash
docker build -t forestoptilm .
docker run --rm -it forestoptilm
```

## Quick Start

```bash
python main.py
```

### Windows: one-click launch

Double-click **`start.bat`** in the project directory. The script will:
- create a `.venv` virtual environment if it doesn't exist;
- recreate `.venv` if it points to a removed Python installation;
- install dependencies from `requirements.txt`;
- launch the application.

## Configuration

### LM Studio connection (recommended)

Copy the template and fill in your values:

```bash
cp config/lmstudio.example.json .local/lmstudio.json
```

```json
{
  "base_url": "http://127.0.0.1:1234/v1",
  "api_key": "your-api-key",
  "timeout": 600,
  "default_model": ""
}
```

The `.local/` directory is git-ignored — secrets never leak to the repository.

Alternative: set the environment variable `NOCTURNE_LMSTUDIO_CONFIG` to a full
path to the JSON config file.

### Defaults

If no local config exists, built-in values from `lmstudio_config.py` are used
(e.g. `http://localhost:1234/v1`, API key `forest`).

### API Mode

The GUI provides a choice between:

| Mode     | Endpoints |
|----------|-----------|
| `native` | LM Studio REST (`/api/v1/*`) |
| `openai` | OpenAI-compatible (`/v1/*`) |

The URL field accepts LM Studio root URLs and copied API URLs such as
`http://127.0.0.1:1234`, `http://127.0.0.1:1234/v1`, and
`http://127.0.0.1:1234/api/v1`.

## Token Budget

Context is read automatically from LM Studio metadata:

1. `loaded_context_length` (runtime)
2. `context_length` / `max_context_length` from metadata
3. Fallback: `8096`

```
response_reserve = clamp(0.2 * effective_context, 1024, 4096)
chunk_size = effective_context - system_prompt - user_query - reserve
```

A guardrail cap (`NOCTURNE_MAX_CHUNK_TOKENS`, default `6000`, `0` = off)
limits the maximum chunk size.

## Map-Reduce Pipeline

1. **MAP** — each chunk produces structured JSON with `findings[]`,
   `evidence_refs[]` (`file`, `chunk`, `quote`), and `recommendations[]`.
2. **Merge** — hierarchical JSON merge for large document sets.
3. **REDUCE** — final markdown report with required sections:
   *Executive Summary*, *Comprehensive Findings*, *Evidence Matrix*, *Action Plan*.
4. **Refine** — second pass if the report is too short or missing sections.
5. **Validation** — warnings appended for short text, missing sections, or low
   evidence density.

Critical/high findings without `file + quote` in `evidence_refs` are
automatically downgraded to **medium**.

## RAG

1. Select an embedding model in the GUI.
2. Choose a file or folder, open the **RAG** tab, click **Build Index**.
3. Enter a question and click **Ask**.

## Project Structure

| File | Purpose |
|------|---------|
| `main.py` | Entry point |
| `gui.py` | CustomTkinter interface (dark theme) |
| `processor.py` | LLM calls, Map-Reduce, batching, backoff |
| `parser.py` | File parsing, tiktoken, chunking |
| `file_extractors.py` | Format routing, archive extraction |
| `cache.py` | SQLite MAP checkpoint cache |
| `embeddings.py` | LM Studio `/v1/embeddings` client |
| `retrieval.py` | Local FAISS index and search |
| `pipeline.py` | Ingestion / index / query pipeline |
| `models.py` | Dataclass models for chunks and retrieval |
| `lmstudio_config.py` | Local JSON config loading, secret masking |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NOCTURNE_LMSTUDIO_CONFIG` | — | Path to custom JSON config |
| `NOCTURNE_LMSTUDIO_NATIVE_API` | `1` | Set `0` to disable native LM Studio API |
| `NOCTURNE_MAX_CHUNK_TOKENS` | `6000` | Max tokens per chunk (`0` = no cap) |
| `NOCTURNE_MEGA_FILE_TOKEN_THRESHOLD` | `80000` | Mega-file part splitting threshold |
| `NOCTURNE_MEGA_PART_FACTOR` | `6` | Coarse part multiplier for mega files |
| `NOCTURNE_CACHE_TTL_DAYS` | `7` | MAP cache TTL in days (`0` = no expiry) |
| `NOCTURNE_CONTEXT_SAFETY_MARGIN` | *(adaptive)* | Fixed token margin; default is ~15% of model context (512–8192) |
| `NOCTURNE_SCOUT_THRESHOLD` | `0.35` | Default relevance threshold when scout is enabled from env |

## Troubleshooting

- **Models duplicated or won't unload** — enable *Low VRAM Sequential* mode,
  toggle API mode and retry.
- **Frequent 400 errors** — check API mode compatibility; look at Logs tab for
  the `400` classifier (`payload_mismatch`, `unsupported_option`, `context_limit`).
- **Reset GUI settings** — delete `.local/ui_runtime.json` and restart.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Copyright (C) 2025 [therudywolf](https://github.com/therudywolf)

This program is free software: you can redistribute it and/or modify it under
the terms of the **GNU Affero General Public License** as published by the Free
Software Foundation, either version 3 of the License, or (at your option) any
later version.

See [LICENSE](LICENSE) for the full text.
