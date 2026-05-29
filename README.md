# 🐺 ForestOptiLM — Nocturne Data Forge

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-brightgreen.svg)](https://www.python.org/)

**ForestOptiLM** is the repository name; the product UI is **Nocturne Data Forge** —
a desktop app for bulk asynchronous processing of large files and folders
(including multi‑hundred‑thousand‑token corpora) through **local LLMs**
(LM Studio REST API v1 and OpenAI-compatible endpoints).

**Core idea:** drop one huge file or a pile of large mixed-format files, type
what you need in plain language, get a precise result — optimized. The engine is
**domain-agnostic**: it understands the task, derives *what to extract* per
fragment, runs Map‑Reduce over the corpus on local models, and assembles the
answer in the form the task implies. No built-in assumptions about any domain
(logs, code, contracts, reports — all the same to the core).

Built on: **query understanding** (intent + per-task extraction schema),
**Map‑Reduce** with evidence-grounding, optional **scout pass** (fast relevance
filter), **reasoning-model control** (`reasoning: off` for qwen3 / deepseek-r1),
an optional **composer** model for merge/reduce, **hybrid retrieval**
(BM25 + vectors), and **deterministic aggregation** (counts/grouping computed in
code, not hallucinated).

**Target workloads (by design):** multi‑GB logs, huge line-oriented dumps, ZIP/TAR
trees with thousands of source files, structured JSON/XML/CSV reports, and folder
corpora in the **~1M+ token** range with scout + hierarchical merge — not
“one tiny PDF only”.

Licensed under **AGPL-3.0-or-later** — see [LICENSE](LICENSE), [NOTICE](NOTICE),
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Десктопное приложение на Python: массовая обработка файлов через локальные LLM
без отправки данных в облако (при использовании локального LM Studio).

## Features

- **Query-adaptive extraction** — the MAP schema is derived from your task, not fixed.
- **Map-Reduce pipeline** with evidence-grounding and faithful merge.
- **Record-aware ingestion** — JSON/JSONL/XML reports split by *records* (one record never split across chunks), no per-format parsers.
- **Deterministic aggregation** — counts/grouping/dedup computed in code and handed to the report as ground truth.
- **Hybrid retrieval (RAG)** — BM25 + local FAISS vectors fused with RRF for exact + semantic search.
- **Run memory** — diff a corpus against its previous run (added/removed/unchanged).
- **Scout pass** — quick relevance scoring before full MAP (for huge folders).
- **Large corpus preset** — scout + smaller chunks + composer in one click.
- **Vision support** for image analysis via multimodal models.
- **Batch table processing** for CSV/XLSX.
- **Low VRAM mode** — sequential model loading for constrained hardware.
- **Composer model** — optional separate model for merge/reduce/refine phases.
- **SQLite cache** for MAP checkpoint resumption + cooperative stop/resume.
- **Dark theme GUI** (CustomTkinter).
- **Reasoning models** — auto-detected from LM Studio catalog; CoT disabled for MAP/REDUCE.
- **Headless CLI** — `python -m forestoptilm.cli analyze`.

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

### GUI

```bash
python main.py
```

### CLI (headless)

```bash
python -m forestoptilm.cli analyze ./docs -q "Summarize security findings"
python -m forestoptilm.cli analyze ./docs -q "..." --profile large_corpus -o report.md
```

### Windows: one-click launch

Double-click **`start.bat`** in the project directory. The script will:
- create a `.venv` virtual environment if it doesn't exist;
- recreate `.venv` if it points to a removed Python installation;
- install dependencies from `requirements.txt`;
- launch the application.

### Windows: build a standalone .exe

No Python needed on the target machine. From a checkout with deps installed:

```powershell
pip install -r requirements.txt pyinstaller
pwsh -File scripts/build_exe.ps1      # or double-click scripts\build_exe.bat
```

The script prefetches the tokenizer (so the app works **offline**) and runs
PyInstaller against [`nocturne.spec`](nocturne.spec). Output is a one-dir app:

```
dist/NocturneDataForge/NocturneDataForge.exe   (+ _internal/)
```

Ship the **whole `dist/NocturneDataForge` folder** (zip it). On first run the app
creates a `NocturneData/` folder next to the `.exe` for cache and indexes. Build
artifacts (`build/`, `dist/`, `.build/`) are git-ignored.

## Configuration

### LM Studio connection (recommended)

Copy the template and fill in your values:

```bash
cp config/lmstudio.example.json .local/lmstudio.json
```

```json
{
  "base_url": "http://127.0.0.1:1234",
  "api_key": "sk-lm-xxxxxxxx:yyyyyyyy",
  "timeout": 600,
  "default_model": "",
  "api_mode": "native"
}
```

The `.local/` directory is git-ignored — secrets never leak to the repository.

Alternative: set the environment variable `NOCTURNE_LMSTUDIO_CONFIG` to a full
path to the JSON config file.

### Defaults

If no local config exists, defaults to `http://127.0.0.1:1234` with no API key.
Create `.local/lmstudio.json` from the template (never commit it).

### LM Studio 0.4+ REST API v1

With **Server Settings → Require authentication** enabled, create a token under
**Manage Tokens** (`sk-lm-...`) and put it in `.local/lmstudio.json`. The app uses:

| Endpoint | Use |
|----------|-----|
| `GET /api/v1/models` | Model list |
| `POST /api/v1/chat` | MAP / REDUCE / scout |
| `POST /api/v1/models/load` | Low VRAM model load (`context_length` supported) |
| `POST /api/v1/models/unload` | Unload instance |

Native mode is default (`api_mode: native` in local config or GUI).

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

## Large files and archives

| Input | Behavior |
|-------|----------|
| **Plain text/code/log ≥ 50 MiB** (`NOCTURNE_STREAMING_FILE_BYTES`) | Line streaming into MAP chunks — file is **not** loaded entirely into RAM |
| **ZIP / TAR / `.tar.gz`** | Extracted to a temp dir; each file keeps its **path** (same as choosing a folder) |
| **Folder** | Every supported file → chunks → one Map-Reduce job (scout recommended) |

Limits are practical, not magical: each MAP/SCOUT chunk still calls your local LLM.
A **trillion-line** file is supported in the sense of **streaming input**; total runtime
depends on LM Studio throughput, scout threshold, and hardware. Use **scout** +
`large_corpus` profile for huge corpora.

| Variable | Default | Meaning |
|----------|---------|---------|
| `NOCTURNE_STREAMING_FILE_BYTES` | `52428800` (50 MiB) | Stream plain files at or above this size (`0` = always load whole file) |
| `NOCTURNE_MAX_ARCHIVE_BYTES` | `8589934592` (8 GiB) | Refuse to extract larger (compressed) archives |
| `NOCTURNE_MAX_UNCOMPRESSED_BYTES` | `8589934592` (8 GiB) | Refuse to extract if total **uncompressed** size exceeds this (zip/tar/gz bomb guard; `0` = off) |
| `NOCTURNE_RECORD_AWARE` | `1` | Structure-preserving ingestion for JSON/JSONL/XML reports — records never split across chunks (`0` = legacy text/table path) |
| `NOCTURNE_RUN_MEMORY` | `1` | Persist extracted items per run and show a diff (added/removed/unchanged) vs the previous run of the same source (`0` = off; alias: `NOCTURNE_FINDINGS_MEMORY`) |
| `NOCTURNE_RECORD_AWARE_MAX_BYTES` | `209715200` (200 MiB) | Above this, large JSON/XML falls back to streaming text (JSONL still streamed per record) |
| `NOCTURNE_MAX_CHUNKS_IN_RAM` | `12000` | Spill MAP chunks to SQLite above this count |
| `NOCTURNE_MAP_BATCH_SIZE` | `workers × 4` | MAP concurrency batch size (limits peak in-flight tasks) |
| `NOCTURNE_MAP_NORMALIZE_SPILL` | `2500` | Spill normalized MAP JSON to SQLite before merge (`0` = keep all in RAM) |
| `NOCTURNE_DUAL_MAP_RESOLVE` | `0` | When `1` and dual-instance pool has 2+ IDs, run MAP on both and merge via `conflict_resolve` |

### GUI (no manual tuning)

- **Быстро / Глубоко / 1M+** — run profiles in one click.
- **Оценка (dry-run)** — chunk/file estimates and rough ETA without LLM calls.
- **История** — recent runs from metrics DB.
- **Продолжить** — resume an interrupted MAP job from SQLite cache. Resume matches
  on file/folder + query **and** run parameters (chunk size, MAP model, composer):
  changing the model or profile starts a fresh job rather than reusing stale chunks.
- Large file / ZIP / folder → **auto** large_corpus preset on Start.

MAP checkpoints and job metadata live under `.nocturne_cache/` (or `NOCTURNE_CACHE_DIR`).
The last incomplete job pointer is stored in `.local/last_job.json` next to your LM Studio config.

## Pipeline

1. **Understand** — the query is parsed into a `QueryPlan` (intent, key terms,
   group-by axis, output style) and a **per-task extraction schema** (what to pull
   from each fragment). Domain-agnostic — security severity is just one optional
   field, used only when the task is about issues/quality/risk.
2. **MAP** — each chunk produces structured JSON: extracted items with
   `type`, content, optional task-specific `fields`, and `evidence_refs`
   (`file`, `chunk`, `quote`) for grounding.
3. **Merge** — hierarchical, deterministic JSON merge for large corpora.
4. **Aggregate** — counts / grouping / dedup computed **in code** (by the
   QueryPlan key) and handed to REDUCE as ground truth, so numbers are correct
   at any volume.
5. **REDUCE** — final Markdown answer; an intent-driven output contract shapes
   the form (table / ranked list / comparison / summary / report).
6. **Refine** — second pass if the answer is too short or incomplete.
7. **Validation** — warnings appended for short output, missing structure, or
   low evidence density.

Items claimed at high importance without `file + quote` evidence are
automatically downgraded — conclusions must be grounded in the source.

Extracted items are deduplicated (by a query-derived key) and capped per chunk
and per merge level — see `NOCTURNE_MAX_FINDINGS_PER_CHUNK` and
`NOCTURNE_MERGE_FINDINGS_CAP` to raise the limits for very large corpora.

## RAG

1. Select an embedding model in the GUI.
2. Choose a file or folder, open the **RAG** tab, click **Build Index**.
3. Enter a question and click **Ask**.

Retrieval is **hybrid**: dense vectors (FAISS) for semantics + BM25 for exact
tokens (`CVE-2024-3094`, hostnames, `pkg@version`), fused with Reciprocal Rank
Fusion. If the embedding model/server is unavailable, BM25-only search still
works — exact lookups don't depend on the embedder.

## Project Structure

| File / dir | Purpose |
|------------|---------|
| `main.py` | GUI entry point |
| `gui.py` | CustomTkinter interface (dark theme) |
| `forestoptilm/cli.py` | Headless `analyze` command |
| `query_plan.py` | Query understanding: intent, entities, per-task extraction schema |
| `processor.py` | LLM calls, query-adaptive Map-Reduce, scout, batching |
| `record_chunking.py` | Format-agnostic record-aware ingestion (JSON/JSONL/XML) |
| `aggregate.py` | Deterministic neutral aggregation (counts/grouping/dedup) |
| `run_memory.py` | Persistent run memory + cross-run diff |
| `bm25.py` | Lexical BM25 index + Reciprocal Rank Fusion (hybrid search) |
| `large_corpus_io.py` | Streaming huge text files; archive → folder expansion |
| `corpus_planner.py` | Dry-run estimates; file-level relevance heuristic |
| `chunk_store.py` | On-disk MAP chunk spill (bounded RAM) |
| `map_result_store.py` | On-disk normalized MAP JSON before merge (large jobs) |
| `conflict_resolve.py` | Pick richer MAP JSON when dual-instance resolve is on |
| `parser.py`, `chunking.py`, `file_extractors.py` | Parsing and chunking |
| `cache.py` | SQLite MAP checkpoint cache (resume) |
| `pipeline.py`, `embeddings.py`, `retrieval.py` | Hybrid retrieval (BM25 + FAISS) |
| `lmstudio_config.py`, `lm_client.py`, `lm_studio_api.py` | LM Studio connection |
| `reasoning_models.py` | Reasoning / thinking model detection |
| `merge_hierarchy.py` | Hierarchical merge helpers |
| `run_config.py` | Immutable per-run configuration |
| `config/run_profiles.yaml` | Presets: `large_corpus`, `quick_scan`, `deep_audit` |
| `tests/` | Unit tests (`pytest`; integration opt-in) |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NOCTURNE_LMSTUDIO_CONFIG` | — | Path to custom JSON config |
| `NOCTURNE_LMSTUDIO_NATIVE_API` | `1` | Set `0` to disable native LM Studio API |
| `NOCTURNE_MAX_CHUNK_TOKENS` | `6000` | Max tokens per chunk (`0` = no cap) |
| `NOCTURNE_MEGA_FILE_TOKEN_THRESHOLD` | `80000` | Mega-file part splitting threshold |
| `NOCTURNE_MEGA_PART_FACTOR` | `6` | Coarse part multiplier for mega files |
| `NOCTURNE_CACHE_TTL_DAYS` | `7` | MAP cache TTL in days (`0` = no expiry) |
| `NOCTURNE_MAX_FINDINGS_PER_CHUNK` | `60` | Cap on deduplicated findings kept per MAP chunk |
| `NOCTURNE_MERGE_FINDINGS_CAP` | `1000` | Cap on deduplicated findings per merge level (file/dir/corpus) |
| `NOCTURNE_CONTEXT_SAFETY_MARGIN` | *(adaptive)* | Fixed token margin; default is ~15% of model context (512–8192) |
| `NOCTURNE_SCOUT_THRESHOLD` | `0.35` | Default relevance threshold when scout is enabled from env |
| `NOCTURNE_RUN_INTEGRATION` | — | Set `1` to run live LM Studio integration tests |
| `NOCTURNE_SKIP_INTEGRATION` | — | Documented alias; CI uses `-m "not integration"` |
| `NOCTURNE_CACHE_DIR` | `.nocturne_cache` | Override MAP SQLite cache directory |
| `NOCTURNE_MAP_BATCH_SIZE` | `workers × 4` | Max parallel MAP chunks per batch |
| `NOCTURNE_MAP_NORMALIZE_SPILL` | `2500` | Spill normalized MAP results before merge |
| `NOCTURNE_DUAL_MAP_RESOLVE` | `0` | Dual-instance MAP + conflict resolution per chunk |
| `NOCTURNE_SMOKE_CHAT_MODEL` | — | Override chat model for integration smoke |
| `NOCTURNE_SMOKE_EMBED_MODEL` | — | Override embedding model for integration smoke |

### Tests

```bash
# Unit tests only (CI default)
python -m pytest tests/ -q -m "not integration"

# Live LM Studio (server must be running; uses .local/lmstudio.json)
set NOCTURNE_RUN_INTEGRATION=1
python -m pytest tests/test_lmstudio_integration.py -m integration -q
```

GUI **Проверить LM Studio** runs full smoke when a chat model is selected.

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

See [LICENSE](LICENSE) for the full text. Third-party dependency licenses:
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Project attribution: [NOTICE](NOTICE).
