# Eval harness — measurement loop for retrieval + answer quality

Reusable rig to make "did this change make answers *better*?" a number instead of
a vibe. The **code here is committed**; the **data it runs on is not** — question
sets and run outputs contain real corpus content (names, handles, paths) and live
in gitignored `eval_data/` and `eval_runs/`.

## Pieces

- `ir_metrics.py` — pure IR metrics (Recall@K, Precision@K, MRR, NDCG@K) over a
  ranked list of chunk-ids vs a gold set. No corpus, no server. Unit-tested in
  `tests/test_eval_metrics.py`.
- `schema.py` — `Question` (JSONL): `id`, `task_type` (enumeration | causal |
  portrait | factoid), `question`, `gold_chunk_ids`, `gold_sources`,
  `gold_answer_points`, `notes`.
- `build_cards.py` — assemble judge cards (baseline answer + project answer + the
  project's **own** retrieved contexts + question) for the multi-judge gate.
- `aggregate_judges.py` — fold K independent judge verdicts per question into a
  stable result (majority verdict + mean + `agree` split-flag). Unit-tested.
- `runner.py` — runs the **real** pipeline over a question set:
  - `dual_rankings(store, q, qvec, cand)` → the two ranked lists (vector, BM25)
    **before** fusion. Compute once; then sweep fusion params in memory.
  - `fuse_to_ids(...)` → apply `fuse_rankings(k, score_weight)` + `candidate_k` +
    `min_score_ratio` offline → ranked chunk-ids. No server.
  - `embed_query(...)` → embed one question via the server (the only server call
    on the retrieval path). Cache the vector; all ablations are then offline.
  - `generate(...)` → full grounded answer (`enhanced` / `deep_mode` /
    `deep_depth`). **Requires a live LM Studio server.**

## Offline vs server

| Step | Server? |
|---|---|
| BM25-only ranking, fusion-param sweep, all IR metrics | no |
| Embedding a question to a query vector (once, then cached) | yes |
| Generating a grounded answer / deep analysis | yes |

## Usage sketch

```python
import sys; sys.path.insert(0, "tools/eval")
from runner import open_store_and_notebook, embed_query, dual_rankings, fuse_to_ids
from schema import load_questions
import ir_metrics as metrics

nb, store = open_store_and_notebook("Telegram_...", Path("dist/.../NocturneData/notebooks"))
qs = load_questions("eval_data/questions.jsonl")

per_q = []
for q in qs:
    qvec = embed_query(q.question, BASE_URL, API_KEY, nb.embedding_model)  # once
    vr, br = dual_rankings(store, q.question, qvec, cand=500)              # once
    ranked = fuse_to_ids(vr, br, k=60, score_weight=1.0, top_k=20)        # sweep me
    per_q.append(metrics.score_ranking(ranked, q.gold_chunk_ids))
print(metrics.aggregate(per_q))
```

Run scripts with the repo root on `PYTHONPATH` (the runner adds it, but the
project modules must be importable). The judge of answer quality is **Claude**
(via subagents over saved `generate()` outputs), never the local model grading
itself.
