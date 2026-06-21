# ForestOptiLM Roadmap

## 0. Recovery

- Fix local launch on Windows when `.venv` points to a removed Python installation.
- Keep LM Studio transport compatible with current REST v1 (`/api/v1/*`) and OpenAI-compatible (`/v1/*`) endpoints.
- Make URL handling tolerant of `host`, `host/v1`, `host/api/v1`, and copied endpoint URLs.
- Add a smoke test that checks model listing, one short chat request, and embeddings against a running LM Studio server.

## 1. Stabilization

- Split provider transport code from `processor.py` into a small client module with typed request/response adapters.
- Add mocked tests for model catalog parsing, context detection, reasoning fallback, load/unload lifecycle, and embeddings.
- Replace global runtime flags with an explicit immutable run configuration passed through GUI and processing layers.
- Add structured error messages for: server unavailable, model missing, unsupported payload option, context overflow, and embedding model mismatch.

## 2. Usability

- Auto-refresh models on startup, but keep the UI responsive and show a clear status when LM Studio is offline.
- Filter model dropdowns by model type: chat, vision, embedding.
- Add a preflight panel that shows selected model availability, loaded context, API mode, and expected chunk size before starting.
- Persist successful runtime context per model and invalidate it when LM Studio reports a different loaded instance config.

## 3. Quality

- Improve final report validation with checks for evidence coverage, duplicate findings, unsupported critical/high severity, and missing recommendations.
- Add deterministic fixtures for MAP/REDUCE prompts so regressions can be tested without a live LLM.
- Add export formats for Markdown, JSON evidence, and CSV evidence matrix.
- Track run metrics in SQLite: duration, retries, failed chunks, model, context, and quality warnings.

## 4. Architecture

- Move GUI, processing, extraction, retrieval, and provider clients into a package layout (`forestoptilm/`).
- Add a CLI entry point for headless batch runs.
- Introduce a plugin-like provider interface for LM Studio, Ollama, llama.cpp server, and generic OpenAI-compatible APIs.
- Add CI for lint, type-check, unit tests, and packaging.

## 5. Product Finish

- Create a first-run setup flow for LM Studio URL, API key, default models, and smoke test.
- Add resumable jobs with a visible run history.
- Add a project/session concept so indexes, cache, logs, and outputs are grouped by user task.
- Ship signed Windows release artifacts once the recovery and stabilization phases are complete.

## 6. Notebooks (NotebookLM-style)

- [x] Persistent named source collections (file / folder / URL) with a per-notebook hybrid index.
- [x] Grounded multi-turn chat: retrieval scoped to the notebook, `[N]` citations with click-to-passage, honest "not in sources" refusal.
- [x] Studio panel: study guide, FAQ, timeline, briefing, flashcards generated over the corpus.
- [x] Research-archive UI: notebook gallery (covers, descriptions, search, index status) ↔ three-pane workspace (Sources · Chat · Studio) with rename/describe/manage.
- [x] Local audio transcription (faster-whisper, optional dep) — audio files become text sources; no TTS (audio-overview script generation deferred).
- [x] Page (PDF) / line citation granularity — chunks carry page/line, surfaced in the citation chip and popup («стр. N» / «строка N»).
- [x] Incremental index updates (append new sources without a full rebuild; rebuild on remove / embedding-model / chunk-size change).
- [x] Retrieval-sized index chunks (~512 tokens, `notebook_index_chunk_tokens`) instead of the chat-model MAP size. Fixes grounded chat always answering «нет ответа»: 6000-token chunks (~12–14k chars) exceeded the embedding model's window, so vectors were dominated by identical `[FILE_PATH]` headers and retrieval was near-random. chunk_size is stored in `index_info.json`; legacy huge-chunk indexes migrate on the next «Построить/обновить».
- [x] nomic task prefixes (`search_document:` at index time, `search_query:` at query time) — `nomic-embed-text-v1.5` is prefix-conditioned; embedding raw degrades recall. `prefix_scheme` is stored in `index_info.json` and a mismatch forces a full rebuild (must use the same scheme for docs and queries or the spaces don't match). Index chat top_k 8→16, context budget 12000.
- [x] Embedding/index-build resilience: `EmbeddingClient._embed_batch` retries 5xx / «model is loading» with backoff + Retry-After (one 500 no longer discards a 200-file build); per-batch progress so a long embed isn't read as a freeze.
- [x] Grounded chat keeps `reasoning:off` (verified live: gemma-4-12b with reasoning ON parrots the system-prompt constraints / thinks out loud into the answer; with reasoning OFF it answers cleanly). `prefer_reasoning_off` is a threaded param (default True for chat) and call_llm still auto-escalates off→on on empty output.
- [x] Vision at index time (`vision_index.py`): image sources (schemas/diagrams/screenshots) are described by a vision model during indexing and the description is stored as the chunk text, so "what's on this diagram / in this table" becomes answerable. Opt-in checkbox in the notebook UI; `has_vision` in `index_info.json` forces a rebuild when toggled; descriptions cached per image content.
- [x] Embedding signal cleanup: strip `[FILE_PATH]/[FILE_TITLE]/[Файл:…]` header boilerplate from the text sent to the embedder (kept in the stored chunk for citations) — scheme bumped to `nomic-v2`, legacy indexes migrate.
- [x] Friendlier failures: notebook chat shows a server-health message for 5xx/timeout instead of raw `Ошибка: Server error 500`; first-run marker moved next to the exe (NocturneData) so the wizard stops reappearing after each rebuild.
- [x] Document understanding (beta.7): DOCX tables are extracted as **Markdown** tables (header + separator) instead of tab-joined lines, so the structure survives chunking and retrieval; **all** worksheets of an `.xlsx`/`.xls` are read (not just the first) with a `__sheet__` label column; images **embedded inside** a `.docx` (diagrams/schemas in the document body) are described by the vision model at index time and appended to the document text (raster only — EMF/WMF vector art is skipped, no rasteriser). Images **embedded inside a PDF** (beta.8) are extracted per-image with **PyMuPDF** (`extract_pdf_images`: `page.get_images` + `doc.extract_image`, skips sub-64px icons, caps at 40/file, lazy `fitz` import) and described the same way; PyMuPDF is bundled into the frozen exe via `nocturne.spec`.
- [x] Retrieval quality, inspired by Karpathy's LLM-Wiki pattern + qmd (beta.9, opt-in «🎯 Точный поиск»): **query expansion** (the chat LLM rewrites the question into 1-2 paraphrases, all are retrieved and the candidate sets merged/deduped → higher recall, fewer «нет ответа») and **listwise LLM re-ranking** (one LLM call orders the merged top-N candidates by real relevance; relevant-first, RRF tail kept so recall isn't lost → higher precision, less noise in context). Pure helpers in `retrieval_enhance.py` (prompts/parse/merge/rerank, all unit-tested); `notebook_chat.answer_question(enhanced=...)` orchestrates the LLM calls, with graceful fallback to plain RRF on any bad/empty output. Off by default (adds ~2 LLM calls/question). Also: a **«🔍 Проверить блокнот» (Lint)** Studio action — Karpathy's wiki health-check — surfaces contradictions / stale-or-duplicate facts / gaps as a tiered Markdown report. And markdown **heading-aware chunk boundaries** (a `## section` starts a new segment so unrelated sections don't merge into one chunk).
- [x] Hardening (beta.8, from an adversarial multi-agent review of the beta.7/8 diff — 9 confirmed findings, all fixed): the per-`index_dir` `LocalFaissStore` cache signature is **content-aware** (`mtime + file sizes`) and the cache is **evicted on rebuild/append**, so a same-mtime rebuild on a coarse-mtime volume (FAT32/exFAT/USB/network — where the user's data dir can live) no longer serves a stale corpus; `_store_for` is lock-guarded; BM25 is stamped with the signature of the meta it was actually built from; retrieval returns empty (not `FileNotFoundError`) if the notebook is deleted mid-query; the chat **«Стоп»** button only shows for the chat op (not index build / URL add / material generation); a cancelled chat turn is no longer persisted into history; multi-sheet xlsx survives a `__sheet__` header collision and keeps integer columns as integers (no `int→float`/`NaN` pollution across heterogeneous sheets); PDF embedded-image descriptions don't get a misleading last-page citation.
- [x] Grounded-chat polish (beta.7): the refusal prompt is **loosened** — the model gives a partial grounded answer when sources only partly cover the question, and refuses only when nothing on-topic was retrieved; on a refusal the UI still shows the **retrieved candidate fragments** («Найдено по теме») so the user can verify. Hybrid retrieval gained a conservative **RRF score floor** (drop the tail below 0.15× the top fused score) to cut noise from the chat context, and the `LocalFaissStore` is now **reused per index dir** (a process-wide cache, thread-safe via a lock) so the FAISS index and BM25 are not rebuilt on every question. A **«Стоп»** button cooperatively cancels an in-flight chat request (the async task is cancelled → the httpx request is torn down).

## 7. Providers & distribution

- [x] Connection presets: LM Studio (native/OpenAI), Ollama, OpenAI-compatible, manual — one-click Base URL + API mode (`connection_presets.py`), with auto-detection and a GUI provider selector.
- [x] CLI `--base-url` / `--api-key` / `--api-mode` so any server (incl. Ollama) works headless.
- [x] Reasoning-model adaptation in `call_llm`: auto-escalate reasoning:off→on on empty output, strip inline `<think>` blocks, salvage from the reasoning channel; heuristic covers gemma-4+, gpt-oss, glm-4.6, o4. Verified live on gemma-4-e2b + gemma-12b (small MAP + big composer).
- [x] Heavy-model load resilience: a `400 "Failed to load model … Operation canceled"` (a big model still loading, ~15s) is now classified as `model_loading` and retried with a backoff wait instead of failing the request — so a slow/contended LM Studio gets time to load the model.
- [x] Optimization run-profiles: `balanced`, `precise`, `ollama_local` alongside the existing presets.
- [x] Cross-platform PyInstaller builds (Windows/macOS/Linux) via CI matrix + per-OS scripts; release artifacts attached to GitHub Releases on tag.
- [x] Name-based vision-model detection (llava / *-vl / qwen2.5-vl / minicpm-v / pixtral / gemma-3/4 …) so Ollama / OpenAI-compatible servers expose vision; the image_url path works on both transports.
- [x] Windows .exe / macOS bundle carry author metadata (therudywolf). Actual code-signing/notarization still needs a paid certificate (scaffolding documented).
- [x] Full LM Studio REST v1 coverage: model download (`POST /api/v1/models/download`) + progress polling (`GET /api/v1/models/download/status/:job_id`) — client helpers in `lm_studio_api.py` and a «Скачать модель…» GUI dialog.
- [x] Model-instance hygiene: every app-triggered load (chat / embedding / vision / composer / scout / context-probe) is tracked, MAP keeps a single target instance, and loaded models are best-effort unloaded on app close (`NOCTURNE_UNLOAD_ON_CLOSE`, default on) so instances stop piling up. Clean shutdown cancels all `after()` timers and moves model-status polling off the UI thread. EmbeddingClient now loads the embedding model at most once per process (skips if already loaded) — fixes LM Studio spawning a new `text-embedding-…:N` instance on every chat/index request.
- [ ] Provider-native context-length detection for Ollama / OpenAI-compatible servers.
- [x] NotebookLM-first UI: app opens on the «📓 Блокноты» tab (Map-Reduce controls hidden there for a clean notebook view), brand violet-indigo accent matching the icon, larger default window. Fixed a hard crash when opening the Notebooks tab with existing notebooks (re-layout was done inside the gallery `<Configure>` event → debounced).
- [x] Packaged exe reads `lmstudio.json` placed NEXT TO the binary (editable without a rebuild), and writes logs / faulthandler traces to `NocturneData/app.log` (windowed exe otherwise has no stderr).

## 8. Smart import (intelligent ingestion)

- [x] Pluggable importer registry that normalizes recognized exports to clean text before chunking (`smart_import.py`), wired into `file_extractors.extract_content`.
- [x] Telegram Desktop HTML export: per-message dialogue, grouped-sender carry-over, service-message skip, `[медиа: …]` markers.
- [x] More formats: WhatsApp `_chat.txt`, Slack/Discord JSON exports (`smart_import.py`: `whatsapp_txt`, `slack_json`, `discord_json`).
- [ ] Generic chat-log heuristic importer (unstructured `name: message` logs).
- [ ] Preserve reply/thread structure and forwarded-from attribution in the cleaned output.
