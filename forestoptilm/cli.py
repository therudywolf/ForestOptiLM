# SPDX-License-Identifier: AGPL-3.0-or-later
"""Headless CLI for batch analysis."""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# Repo root on path when invoked as module
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lmstudio_config import get_connection_defaults
from parser import compute_dynamic_chunk_size, parse_file
from pipeline import _iter_files
from processor import SYSTEM_PROMPT_MAP, compute_job_id, run_map_reduce
from run_profiles import get_profile
from chunking import build_document_chunks, chunks_to_map_strings


def _collect_chunks(target: Path, chunk_size: int) -> list[str]:
    if target.is_file():
        kind, payload, _ = parse_file(target, chunk_size, root_dir=target.parent)
        if kind == "text":
            return payload  # type: ignore[return-value]
        if kind == "vision":
            return payload  # type: ignore[return-value]
        return []
    files = _iter_files([target])
    out: list[str] = []
    for fp in files:
        for dc in build_document_chunks(fp, chunk_size, root_dir=target):
            out.append(dc.text)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="forestoptilm", description="Nocturne Data Forge CLI")
    sub = p.add_subparsers(dest="command", required=True)
    analyze = sub.add_parser("analyze", help="Run Map-Reduce on file or folder")
    analyze.add_argument("path", type=Path)
    analyze.add_argument("--query", "-q", required=True)
    analyze.add_argument("--model", "-m", default="")
    analyze.add_argument("--composer", default="")
    analyze.add_argument("--scout-model", default="")
    analyze.add_argument("--profile", default="", help="run_profiles.yaml key")
    analyze.add_argument("--workers", type=int, default=0)
    analyze.add_argument("--output", "-o", type=Path, default=None)
    args = p.parse_args(argv)

    if args.command != "analyze":
        return 1

    base_url, api_key, _ = get_connection_defaults()
    profile = get_profile(args.profile) if args.profile else {}
    scout_mode = bool(profile.get("scout_mode", False))
    scout_threshold = float(profile.get("scout_threshold", 0.35))
    workers = int(args.workers or profile.get("workers", 3))
    max_chunk = int(profile.get("max_chunk_tokens", 6000))
    model = args.model.strip() or None
    if not model:
        from processor import fetch_models
        models = fetch_models(base_url, api_key)
        if not models:
            print("No models available", file=sys.stderr)
            return 2
        model = next((m for m in models if "embed" not in m.lower()), models[0])

    chunk_size = compute_dynamic_chunk_size(8096, SYSTEM_PROMPT_MAP, args.query)
    if max_chunk:
        import os
        os.environ["NOCTURNE_MAX_CHUNK_TOKENS"] = str(max_chunk)

    chunks = _collect_chunks(args.path, chunk_size)
    if not chunks:
        print("No chunks extracted", file=sys.stderr)
        return 3

    job_id = compute_job_id(
        args.path,
        args.query,
        file_paths=_iter_files([args.path]) if args.path.is_dir() else None,
    )
    composer = args.composer.strip() or (model if profile.get("composer_enabled") else None)
    scout_m = args.scout_model.strip() or None

    result = asyncio.run(
        run_map_reduce(
            chunks=chunks,
            user_query=args.query,
            base_url=base_url,
            api_key=api_key,
            model=model,
            workers=workers,
            dynamic_chunk_size=chunk_size,
            job_id=job_id,
            composer_model=composer,
            scout_mode=scout_mode,
            scout_relevance_threshold=scout_threshold,
            scout_model=scout_m,
        )
    )
    out_path = args.output or Path("report.md")
    out_path.write_text(result, encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
