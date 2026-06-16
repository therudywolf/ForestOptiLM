# SPDX-License-Identifier: AGPL-3.0-or-later
"""Zip the PyInstaller one-dir build into dist/NocturneDataForge-<os>-<arch>.zip.

Cross-platform packaging for release artifacts. Run after PyInstaller.
"""
from __future__ import annotations

import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"


def _os_tag() -> str:
    return {"win32": "Windows", "darwin": "macOS", "linux": "Linux"}.get(sys.platform, sys.platform)


def main() -> int:
    # macOS: предпочитаем готовый .app-бандл (двойной клик в Finder), иначе one-dir.
    app_bundle = DIST / "NocturneDataForge.app"
    if sys.platform == "darwin" and app_bundle.is_dir():
        base_dir = "NocturneDataForge.app"
    else:
        base_dir = "NocturneDataForge"
    build_dir = DIST / base_dir
    if not build_dir.is_dir():
        print(f"error: {build_dir} not found — run PyInstaller first", file=sys.stderr)
        return 1
    arch = platform.machine() or "x64"
    base = DIST / f"NocturneDataForge-{_os_tag()}-{arch}"
    archive = shutil.make_archive(str(base), "zip", root_dir=str(DIST), base_dir=base_dir)
    size_mb = Path(archive).stat().st_size / (1024 * 1024)
    print(f"packaged: {archive} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
