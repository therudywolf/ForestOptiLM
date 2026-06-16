# SPDX-License-Identifier: AGPL-3.0-or-later
"""Package the PyInstaller one-dir build for release.

- Windows -> .zip
- macOS / Linux / Fedora -> .tar.gz  (preserves the exec bit and symlinks that
  PyInstaller one-dir builds rely on; a plain .zip would drop them and the binary
  would not run).

Output name: dist/NocturneDataForge-<os>-<arch>.{zip,tar.gz}. Override the OS tag
with NOCTURNE_DIST_OS (e.g. "Fedora") to distinguish distro-specific Linux builds.
"""
from __future__ import annotations

import os
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
ASSETS = ROOT / "assets" / "linux"


def _os_tag() -> str:
    override = os.getenv("NOCTURNE_DIST_OS", "").strip()
    if override:
        return override
    return {"win32": "Windows", "darwin": "macOS", "linux": "Linux"}.get(sys.platform, sys.platform)


def main() -> int:
    # macOS: prefer the ready-made .app bundle (double-click in Finder).
    app_bundle = DIST / "NocturneDataForge.app"
    if sys.platform == "darwin" and app_bundle.is_dir():
        base_dir = "NocturneDataForge.app"
    else:
        base_dir = "NocturneDataForge"
    build_dir = DIST / base_dir
    if not build_dir.is_dir():
        print(f"error: {build_dir} not found — run PyInstaller first", file=sys.stderr)
        return 1

    # Convenience launcher for Linux/Fedora next to the binary.
    if sys.platform.startswith("linux") and base_dir == "NocturneDataForge":
        launcher = ASSETS / "run.sh"
        if launcher.is_file():
            dest = build_dir / "run.sh"
            shutil.copy2(launcher, dest)
            os.chmod(dest, 0o755)

    fmt = "zip" if sys.platform == "win32" else "gztar"
    arch = platform.machine() or "x64"
    base = DIST / f"NocturneDataForge-{_os_tag()}-{arch}"
    archive = shutil.make_archive(str(base), fmt, root_dir=str(DIST), base_dir=base_dir)
    size_mb = Path(archive).stat().st_size / (1024 * 1024)
    print(f"packaged: {archive} ({size_mb:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
