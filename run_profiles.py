# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load YAML run profiles."""
from __future__ import annotations

from pathlib import Path
from typing import Any

_PROFILES_PATH = Path(__file__).resolve().parent / "config" / "run_profiles.yaml"


def load_profiles() -> dict[str, dict[str, Any]]:
    if not _PROFILES_PATH.is_file():
        return {}
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        return _load_profiles_minimal()
    data = yaml.safe_load(_PROFILES_PATH.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def _load_profiles_minimal() -> dict[str, dict[str, Any]]:
    """Fallback without PyYAML."""
    out: dict[str, dict[str, Any]] = {}
    current: str | None = None
    for line in _PROFILES_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.endswith(":") and not line.startswith(" "):
            current = line[:-1]
            out[current] = {}
        elif ":" in line and current:
            k, v = line.split(":", 1)
            v = v.strip()
            if v.lower() in ("true", "false"):
                out[current][k.strip()] = v.lower() == "true"
            else:
                try:
                    out[current][k.strip()] = int(v) if "." not in v else float(v)
                except ValueError:
                    out[current][k.strip()] = v
    return out


def get_profile(name: str) -> dict[str, Any]:
    return dict(load_profiles().get(name, {}))
