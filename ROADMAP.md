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
- [x] Incremental index updates (append new sources without a full rebuild; rebuild on remove / embedding-model change).

## 7. Providers & distribution

- [x] Connection presets: LM Studio (native/OpenAI), Ollama, OpenAI-compatible, manual — one-click Base URL + API mode (`connection_presets.py`), with auto-detection and a GUI provider selector.
- [x] CLI `--base-url` / `--api-key` / `--api-mode` so any server (incl. Ollama) works headless.
- [x] Reasoning-model adaptation in `call_llm`: auto-escalate reasoning:off→on on empty output, strip inline `<think>` blocks, salvage from the reasoning channel; heuristic covers gemma-4+, gpt-oss, glm-4.6, o4. Verified live on gemma-4-e2b + gemma-12b (small MAP + big composer).
- [x] Optimization run-profiles: `balanced`, `precise`, `ollama_local` alongside the existing presets.
- [x] Cross-platform PyInstaller builds (Windows/macOS/Linux) via CI matrix + per-OS scripts; release artifacts attached to GitHub Releases on tag.
- [x] Name-based vision-model detection (llava / *-vl / qwen2.5-vl / minicpm-v / pixtral / gemma-3/4 …) so Ollama / OpenAI-compatible servers expose vision; the image_url path works on both transports.
- [x] Windows .exe / macOS bundle carry author metadata (therudywolf). Actual code-signing/notarization still needs a paid certificate (scaffolding documented).
- [x] Full LM Studio REST v1 coverage: model download (`POST /api/v1/models/download`) + progress polling (`GET /api/v1/models/download/status/:job_id`) — client helpers in `lm_studio_api.py` and a «Скачать модель…» GUI dialog.
- [x] Model-instance hygiene: every app-triggered load (chat / embedding / vision / composer / scout / context-probe) is tracked, MAP keeps a single target instance, and loaded models are best-effort unloaded on app close (`NOCTURNE_UNLOAD_ON_CLOSE`, default on) so instances stop piling up. Clean shutdown cancels all `after()` timers and moves model-status polling off the UI thread.
- [ ] Provider-native context-length detection for Ollama / OpenAI-compatible servers.

## 8. Smart import (intelligent ingestion)

- [x] Pluggable importer registry that normalizes recognized exports to clean text before chunking (`smart_import.py`), wired into `file_extractors.extract_content`.
- [x] Telegram Desktop HTML export: per-message dialogue, grouped-sender carry-over, service-message skip, `[медиа: …]` markers.
- [x] More formats: WhatsApp `_chat.txt`, Slack/Discord JSON exports (`smart_import.py`: `whatsapp_txt`, `slack_json`, `discord_json`).
- [ ] Generic chat-log heuristic importer (unstructured `name: message` logs).
- [ ] Preserve reply/thread structure and forwarded-from attribution in the cleaned output.
