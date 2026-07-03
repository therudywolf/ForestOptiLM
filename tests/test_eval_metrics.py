# SPDX-License-Identifier: AGPL-3.0-or-later
"""IR-метрики eval-харнеса (tools/eval/ir_metrics.py) — чистые, без корпуса/сервера.

Грузим через importlib под уникальным именем, БЕЗ sys.path.insert: иначе каталог
tools/eval попал бы в sys.path на весь процесс pytest и `import metrics` в других
тестах поднял бы наш модуль вместо КОРНЕВОГО metrics.py (SQLite-метрики) → падение
сборки. Имя файла ir_metrics и так уникально, но изоляция важнее удобства."""
from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

_p = Path(__file__).resolve().parent.parent / "tools" / "eval" / "ir_metrics.py"
_spec = importlib.util.spec_from_file_location("eval_ir_metrics", _p)
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)

_pa = Path(__file__).resolve().parent.parent / "tools" / "eval" / "aggregate_judges.py"
_spa = importlib.util.spec_from_file_location("eval_aggregate_judges", _pa)
aj = importlib.util.module_from_spec(_spa)
_spa.loader.exec_module(aj)


class TestIRMetrics(unittest.TestCase):
    def test_recall_at_k(self) -> None:
        ranked = ["a", "b", "c", "d"]
        self.assertEqual(metrics.recall_at_k(ranked, {"b", "d"}, 4), 1.0)
        self.assertEqual(metrics.recall_at_k(ranked, {"b", "d"}, 2), 0.5)  # only b in top-2
        self.assertEqual(metrics.recall_at_k(ranked, {"z"}, 4), 0.0)
        self.assertEqual(metrics.recall_at_k(ranked, set(), 4), 0.0)  # no gold → 0

    def test_precision_at_k(self) -> None:
        ranked = ["a", "b", "c"]
        self.assertAlmostEqual(metrics.precision_at_k(ranked, {"a", "b"}, 3), 2 / 3)
        self.assertEqual(metrics.precision_at_k(ranked, {"a"}, 0), 0.0)

    def test_reciprocal_rank(self) -> None:
        self.assertEqual(metrics.reciprocal_rank(["a", "b", "c"], {"a"}), 1.0)
        self.assertEqual(metrics.reciprocal_rank(["a", "b", "c"], {"b"}), 0.5)
        self.assertAlmostEqual(metrics.reciprocal_rank(["a", "b", "c"], {"c"}), 1 / 3)
        self.assertEqual(metrics.reciprocal_rank(["a", "b"], {"z"}), 0.0)

    def test_ndcg_rewards_higher_rank(self) -> None:
        # Тот же набор попаданий, но выше в списке → больше NDCG.
        top = metrics.ndcg_at_k(["a", "x", "y"], {"a"}, 3)
        low = metrics.ndcg_at_k(["x", "y", "a"], {"a"}, 3)
        self.assertGreater(top, low)
        self.assertEqual(top, 1.0)  # единственный релевантный на позиции 1 = идеал
        self.assertAlmostEqual(low, 1.0 / math.log2(4))

    def test_ndcg_perfect_and_empty(self) -> None:
        self.assertEqual(metrics.ndcg_at_k(["a", "b"], {"a", "b"}, 2), 1.0)
        self.assertEqual(metrics.ndcg_at_k(["a"], set(), 2), 0.0)

    def test_score_ranking_bundle(self) -> None:
        s = metrics.score_ranking(["a", "b", "c"], {"a", "c"}, ks=(1, 3))
        self.assertEqual(s["recall@1"], 0.5)
        self.assertEqual(s["recall@3"], 1.0)
        self.assertEqual(s["mrr"], 1.0)
        self.assertEqual(s["n_gold"], 2.0)

    def test_aggregate_skips_ungolded(self) -> None:
        per_q = [
            metrics.score_ranking(["a", "b"], {"a"}, ks=(2,)),
            metrics.score_ranking(["x", "y"], set(), ks=(2,)),  # no gold — skipped
        ]
        agg = metrics.aggregate(per_q)
        self.assertEqual(agg["n_questions"], 2.0)
        self.assertEqual(agg["n_scored"], 1.0)
        self.assertEqual(agg["recall@2"], 1.0)  # only the golded question counts


class TestMultiJudgeAggregate(unittest.TestCase):
    def _rec(self, qid, beats, p, b, hall=False):
        trip = lambda x: {"completeness": x, "correctness": x, "grounding": x}
        return {"qid": qid, "project_beats_baseline": beats,
                "project": trip(p), "baseline": trip(b), "project_hallucination": hall}

    def test_majority_verdict_and_means(self) -> None:
        # 3 судьи по q1: 2×no + 1×yes → мажоритарно no, agree=False.
        recs = [self._rec("q1", "no", 2, 5), self._rec("q1", "no", 3, 5),
                self._rec("q1", "yes", 4, 4)]
        agg = aj.aggregate(recs)["q1"]
        self.assertEqual(agg["beats"], "no")
        self.assertFalse(agg["agree"])                 # судьи разошлись
        self.assertEqual(agg["beats_votes"], {"no": 2, "yes": 1})
        self.assertAlmostEqual(agg["project_mean"], 3.0)   # (2+3+4)/3
        self.assertAlmostEqual(agg["baseline_mean"], 4.67)  # (5+5+4)/3, rounded 2 dp

    def test_unanimous_and_hallucination_majority(self) -> None:
        recs = [self._rec("q2", "yes", 5, 3, hall=True), self._rec("q2", "yes", 5, 3, hall=True),
                self._rec("q2", "yes", 5, 3, hall=False)]
        agg = aj.aggregate(recs)["q2"]
        self.assertTrue(agg["agree"])
        self.assertEqual(agg["beats"], "yes")
        self.assertTrue(agg["hallucination"])          # 2/3 → большинство
        self.assertEqual(agg["hallucination_votes"], 2)

    def test_tally(self) -> None:
        agg = aj.aggregate([
            self._rec("q1", "yes", 4, 3), self._rec("q1", "yes", 4, 3),
            self._rec("q2", "no", 2, 5), self._rec("q2", "no", 2, 5),
            self._rec("q3", "no", 3, 3), self._rec("q3", "tie", 3, 3),  # split
        ])
        t = aj.tally(agg)
        self.assertEqual(t["n_questions"], 3)
        self.assertEqual(t["wins"], 1)
        self.assertEqual(t["losses"], 2)   # q3 majority no (tie-break by most_common order)
        self.assertEqual(t["split_verdicts"], 1)  # q3 judges disagreed


if __name__ == "__main__":
    unittest.main()
