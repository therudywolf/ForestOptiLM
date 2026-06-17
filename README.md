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

- **Notebooks (NotebookLM-style)** — persistent named source collections with a
  **grounded chat** that answers *only* from that notebook's sources, with
  clickable `[N]` citations (open the file / see the cited passage) and an honest
  *"not in the sources"* refusal. Plus a **Studio** panel that generates study
  guides, FAQs, timelines, briefings and flashcards over the corpus. See
  [Notebooks](#notebooks).
- **Smart import** — recognized chat exports are normalized to clean,
  LLM-friendly `[date] sender: text` blocks *before* indexing instead of being
  dumped as raw markup. Shipped importers: **Telegram Desktop HTML**
  (`messages*.html`), **WhatsApp** (`_chat.txt`), **Slack** and **Discord** JSON
  exports — with grouped-message author carry-over, service-message skipping and
  `[медиа: …]` markers. Works everywhere (Map-Reduce, RAG, Notebooks) and is
  pluggable for more formats. See [Smart import](#smart-import).
- **Multiple backends** — one-click **provider presets** for LM Studio (native or
  OpenAI), **Ollama**, and any OpenAI-compatible server (vLLM, llama.cpp, LocalAI);
  models picked from a live API list. See [Provider presets](#provider-presets-lm-studio--ollama--openai-compatible).
- **Cross-platform builds** — standalone apps for **Windows, macOS and Linux** via
  a CI matrix and per-OS scripts. See [Desktop builds](#desktop-builds-windows--macos--linux).
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

# Point at any server (here: Ollama) without editing config files
python -m forestoptilm.cli analyze ./docs -q "..." \
  --base-url http://127.0.0.1:11434 --api-mode openai -m qwen2.5 --profile ollama_local
```

Flags: `--base-url`, `--api-key`, `--api-mode {native,openai}`, `--model/-m`,
`--profile`, `--workers`, `--composer`, `--scout-model`, `--output/-o`.

### Windows: one-click launch

Double-click **`start.bat`** in the project directory. The script will:
- create a `.venv` virtual environment if it doesn't exist;
- recreate `.venv` if it points to a removed Python installation;
- install dependencies from `requirements.txt`;
- launch the application.

## Desktop builds (Windows / macOS / Linux)

Standalone builds need no Python on the target machine. PyInstaller does **not**
cross-compile, so each OS is built on its own — locally or via CI.

### Download a release

Tagged releases publish a per-OS archive (built by the
[Release workflow](.github/workflows/release.yml)):

| OS | Asset | Run |
|----|-------|-----|
| Windows | `NocturneDataForge-Windows-*.zip` | unzip → `NocturneDataForge.exe` |
| macOS | `NocturneDataForge-macOS-*.tar.gz` | `tar xzf …` → open `NocturneDataForge.app` |
| Linux | `NocturneDataForge-Linux-*.tar.gz` | `tar xzf …` → `./NocturneDataForge/NocturneDataForge` (or `run.sh`) |
| Fedora | `NocturneDataForge-Fedora-*.tar.gz` | `tar xzf …` → `./NocturneDataForge/NocturneDataForge` (or `run.sh`) |

Linux/macOS archives are `.tar.gz` so the executable bit and symlinks survive (a
plain zip drops them and the binary won't launch). The **Fedora** build is made
inside a `fedora:latest` container for native-library compatibility; the generic
**Linux** build is made on Ubuntu and runs on most recent distros.

On Linux, run `./install-desktop.sh` from inside the unpacked folder to add
**Nocturne Data Forge** (with its icon) to your application menu; it installs a
per-user `~/.local/share/applications/nocturnedataforge.desktop`.

Every push also builds all targets on CI ([Build workflow](.github/workflows/build.yml))
so artifacts are downloadable from the Actions run.

### Build it yourself

From a checkout with deps installed (`pip install -r requirements.txt pyinstaller`):

```powershell
# Windows
pwsh -File scripts/build_exe.ps1      # or double-click scripts\build_exe.bat
```

```bash
# macOS / Linux
bash scripts/build.sh
```

Both prefetch the tokenizer (so the app works **offline**) and run PyInstaller
against [`nocturne.spec`](nocturne.spec). Output is a one-dir app under
`dist/NocturneDataForge/` (plus `dist/NocturneDataForge.app` on macOS). Package it
for release with `python scripts/package_dist.py` (`.zip` on Windows, `.tar.gz`
elsewhere). On first run the app creates a `NocturneData/` folder next to the
binary for cache and indexes. Build artifacts (`build/`, `dist/`, `.build/`) are
git-ignored.

### Code signing

The Windows `.exe` carries author/product metadata (publisher **therudywolf**,
via [`scripts/version_info.txt`](scripts/version_info.txt)) and the macOS bundle
carries its copyright, but the binaries are **not code-signed** — so Windows
SmartScreen and macOS Gatekeeper will warn on first run (Windows: *More info →
Run anyway*; macOS: right-click → *Open*, or `xattr -d com.apple.quarantine`).
Real signing needs a paid certificate — an **Authenticode** cert (Windows) and an
**Apple Developer ID** (macOS, for notarization). Once you have them, sign the
build with `signtool` / `codesign`+`notarytool` (wire it into the Release workflow
behind repo secrets); no certificate ships in this repo.

## Configuration

### Provider presets (LM Studio / Ollama / OpenAI-compatible)

Pick a **Провайдер (сервер LLM)** in the sidebar and the Base URL + API mode are
filled for you — no need to remember ports or paths:

| Preset | Base URL | API mode | Notes |
|--------|----------|----------|-------|
| **LM Studio (REST v1)** | `http://127.0.0.1:1234` | `native` | reasoning:off for thinking models, auto load/unload, context from metadata |
| **LM Studio (OpenAI API)** | `http://127.0.0.1:1234` | `openai` | same server via `/v1` — use if native is finicky |
| **Ollama** | `http://127.0.0.1:11434` | `openai` | OpenAI-compatible `/v1`; no API key |
| **OpenAI-compatible** | *(you enter)* | `openai` | vLLM, llama.cpp server, LocalAI, … |
| **Вручную** | *(you enter)* | *(you choose)* | manual |

Then click **Обновить модели** to list the models the server actually exposes and
pick the chat / embedding / vision model from the dropdowns. The preset is
auto-detected from your saved URL on next launch.

#### Ollama quickstart

```bash
ollama serve                      # start the server (:11434)
ollama pull qwen2.5               # a chat model
ollama pull nomic-embed-text      # an embedding model for RAG / Notebooks
```

In the app: Провайдер → **Ollama**, **Обновить модели**, select `qwen2.5` as the
LLM and `nomic-embed-text` as the embedding model. CLI: `--base-url
http://127.0.0.1:11434 --api-mode openai`. (Embeddings need an Ollama build with
the OpenAI-compatible `/v1/embeddings` endpoint — any recent release.)

### LM Studio connection file (optional)

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

### Optimization presets (run profiles)

[`config/run_profiles.yaml`](config/run_profiles.yaml) bundles engine settings
(scout filter, chunk size, workers, composer merge, low-VRAM) into named presets
applied with `--profile <name>` (CLI) or the GUI buttons:

| Profile | For | Highlights |
|---------|-----|-----------|
| `quick_scan` | fast first pass | scout on, small chunks |
| `balanced` | mid-size corpora | no scout, 5k chunks |
| `precise` | maximum fidelity | small chunks, composer merge |
| `deep_audit` | careful review | no scout, large chunks |
| `large_corpus` | 1M+ tokens | scout + composer + low-VRAM |
| `ollama_local` | Ollama / small context | scout, 3k chunks |

The actual extraction prompt is **derived from your query** (intent + per-task
schema), so presets tune *throughput and context*, not the wording — the wording
adapts itself. Context size is read automatically from the server; cap it with
`NOCTURNE_MAX_CHUNK_TOKENS` or the **Дополнительно** panel.

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

## Smart import

Raw exports index poorly: a Telegram HTML dump fed through naive `get_text()`
becomes a soup of navigation, dates, reactions and markup. **Smart import**
detects known export formats and rewrites them into clean, per-message text
*before* chunking — the same shape people hand-prepare for NotebookLM.

- **Telegram Desktop HTML export** — point the app (or a notebook source) at a
  `messages*.html` file or the export folder. Each message becomes
  `[<date+timezone>] <sender>:\n<text>`; consecutive grouped messages inherit the
  previous sender (Telegram omits it), `message service` separators are dropped,
  and photos/files/stickers get a `[медиа: …]` marker instead of vanishing.
- **WhatsApp** (`_chat.txt`), **Slack** and **Discord** JSON exports — same clean
  `[date] sender: text` normalization (`whatsapp_txt`, `slack_json`,
  `discord_json` importers in `smart_import.py`).

Detection and conversion happen inside
[`file_extractors.extract_content`](file_extractors.py), so smart import benefits
the main Map-Reduce flow, RAG and Notebooks alike — no special button. Anything
not recognized falls back to the normal `html→text` path. Adding a new format is a
single class in [`smart_import.py`](smart_import.py).

## Notebooks

A **Notebook** is a persistent, named collection of sources you can chat with and
study — the local-first take on NotebookLM. Each notebook is a self-contained
folder; everything stays on your machine.

The **Блокноты** tab opens on a **research archive** — a gallery of notebook cards
(cover, description, source/chunk counts, index status, search) you return to,
reopen, rename, describe and manage over time. Opening a card switches to a
three-pane **workspace** — **Источники · Чат · Studio** — mirroring NotebookLM.

1. In the archive, click **＋ Новое исследование** to create a notebook (then
   **Изменить** to set its description/icon; the cover colour is automatic).
2. Add sources: **Файл** / **Папка** (indexed in place — huge dumps are not
   copied) or **URL** (the page text is fetched and saved into the notebook).
3. Click **🔨 Построить индекс** — builds the notebook's own hybrid index
   (BM25 + FAISS) using the embedding model selected in the sidebar.
4. **Chat:** ask in plain language. Answers are **grounded** — retrieval runs
   *only* over this notebook, every claim carries a `[N]` citation, and clicking
   a citation chip shows the exact passage with a button to open the source. If
   the answer isn't in the sources, the assistant says so instead of inventing it.
5. **Studio:** one click generates a **study guide**, **FAQ**, **timeline**,
   **briefing/synopsis** or **flashcards** over the corpus; results are saved as
   notes inside the notebook.

Notebooks live in `NocturneData/notebooks/` next to the packaged `.exe`, or in
`.local/notebooks/` when running from source (override with
`NOCTURNE_NOTEBOOKS_DIR`). Layout per notebook: `notebook.json` (metadata),
`index/` (FAISS + BM25), `sources/` (URL/derived text), `notes/` (Studio output),
`chat.jsonl` (chat history). Adding a source rebuilds the index (FAISS is
immutable). Audio files (`.mp3/.wav/.m4a/.ogg/.flac/.opus/.aac/.wma`) are
transcribed locally via the optional `faster-whisper` dependency and become text
sources; only TTS / audio-video "overviews" are intentionally out of scope.

## Project Structure

| File / dir | Purpose |
|------------|---------|
| `main.py` | GUI entry point |
| `notebook_store.py` | Notebooks: persistent source collections, CRUD, per-notebook index/chat/notes |
| `notebook_chat.py` | Grounded chat over a notebook with `[N]` citations + honest refusal |
| `notebook_studio.py` | Studio: study guide / FAQ / timeline / briefing / flashcards generation |
| `notebook_gui.py` | «Блокноты» tab (NotebookUIMixin) — sources, chat with citation chips, Studio |
| `url_ingest.py` | Fetch a web page and extract its text as a notebook source |
| `smart_import.py` | Smart import: recognize exports (Telegram HTML…) and normalize to clean text |
| `connection_presets.py` | Provider presets (LM Studio / Ollama / OpenAI-compatible) + auto-detection |
| `scripts/build.sh`, `build_exe.ps1` | Per-OS PyInstaller builds (Linux/macOS, Windows) |
| `scripts/prefetch_tiktoken.py`, `package_dist.py` | Offline tokenizer cache; zip the build per OS |
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
| `NOCTURNE_NOTEBOOKS_DIR` | *(auto)* | Override where Notebooks are stored (default: next to cache dir / `.local/notebooks`) |
| `NOCTURNE_URL_MAX_BYTES` | `26214400` (25 MiB) | Max response size when adding a URL source |

### Tests

```bash
# Unit tests only (CI default)
python -m pytest tests/ -q -m "not integration"

# Live LM Studio (server must be running; uses .local/lmstudio.json)
set NOCTURNE_RUN_INTEGRATION=1
python -m pytest tests/test_lmstudio_integration.py -m integration -q
```

GUI **Проверить подключение** runs full smoke when a chat model is selected.

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
