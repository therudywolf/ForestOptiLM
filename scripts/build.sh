#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Build Nocturne Data Forge with PyInstaller on Linux or macOS.
#
#   bash scripts/build.sh
#
# Output (one-dir):
#   Linux: dist/NocturneDataForge/NocturneDataForge
#   macOS: dist/NocturneDataForge/NocturneDataForge  +  dist/NocturneDataForge.app
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

echo "==> Ensuring build tools (pyinstaller)..."
"$PY" -m pip install --quiet --upgrade pyinstaller

echo "==> Prefetching tiktoken encoding for offline use..."
"$PY" scripts/prefetch_tiktoken.py || echo "   (prefetch skipped)"

echo "==> Running PyInstaller..."
"$PY" -m PyInstaller --noconfirm --clean nocturne.spec

BIN="dist/NocturneDataForge/NocturneDataForge"
if [ -f "$BIN" ]; then
  echo "==> Built: $BIN"
  if [ -d "dist/NocturneDataForge.app" ]; then
    echo "    macOS bundle: dist/NocturneDataForge.app"
  fi
  echo "    Ship the whole 'dist/NocturneDataForge' folder (zip it), or run scripts/package_dist.py."
else
  echo "Build finished but binary not found at $BIN" >&2
  exit 1
fi
