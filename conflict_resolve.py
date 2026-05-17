# SPDX-License-Identifier: AGPL-3.0-or-later
"""Resolve conflicting MAP JSON from two worker models."""
from __future__ import annotations

import json


def pick_findings_from_dual_worker(
    result_a: str,
    result_b: str,
) -> str:
    """Простое правило: больше findings с evidence → победитель; иначе A."""
    def score(raw: str) -> int:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            return 0
        if not isinstance(obj, dict):
            return 0
        n = 0
        for f in obj.get("findings") or []:
            if isinstance(f, dict) and f.get("evidence_refs"):
                n += 1
        return n

    sa, sb = score(result_a), score(result_b)
    return result_b if sb > sa else result_a
