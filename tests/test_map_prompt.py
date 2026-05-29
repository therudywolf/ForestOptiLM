# SPDX-License-Identifier: AGPL-3.0-or-later
"""Query-adaptive MAP system prompt — neutral by default, task-driven with a plan."""
from __future__ import annotations

import unittest

from processor import build_map_system_prompt
from query_plan import build_query_plan


class TestMapPrompt(unittest.TestCase):
    def test_default_is_neutral(self) -> None:
        p = build_map_system_prompt(None)
        low = p.lower()
        # Stable envelope keys must remain for downstream merge/aggregate.
        self.assertIn("findings", low)
        self.assertIn("evidence_refs", low)
        self.assertIn("no_relevant_data", low)
        # No security framing in the neutral default.
        self.assertNotIn("vulnerab", low)
        # severity must be optional, not mandated.
        self.assertIn("optional", low)

    def test_adaptive_includes_task_and_fields(self) -> None:
        plan = build_query_plan("извлеки контрагентов и суммы из договоров")
        p = build_map_system_prompt(plan)
        # The derived extraction directive is injected.
        self.assertIn("TASK", p)
        # Derived fields surface in the prompt.
        for f in plan.extraction_fields:
            self.assertIn(f, p)

    def test_explain_task_schema(self) -> None:
        plan = build_query_plan("объясни почему упала выручка")
        p = build_map_system_prompt(plan)
        self.assertIn("claim", p)  # explain intent fields


if __name__ == "__main__":
    unittest.main()
