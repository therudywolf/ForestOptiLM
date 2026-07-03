# Eval run — findings (2026-07-03)

First full pass of the measurement loop against a large real-world corpus
(~455k chunks, mixed chat + notification data). All corpus content stays local
and gitignored; this report is deliberately data-agnostic (no entities, no text).

## Setup

- **20 questions**, 5 per task type: `factoid`, `enumeration`, `causal`, `portrait`.
- **Retrieval gold**: anchor-based (relevant = chunk mentions the anchor). Clean
  only for entity/portrait questions (a generic topic word matches 35–52% of this
  corpus → useless as gold). So entity questions drive IR-metric tuning; all 20
  drive answer-quality judging.
- **E2E sweep**: 3 models (`gemma-4-e2b`, `-12b-qat`, `-26b-a4b-qat`) ×
  {`simple`, `enhanced`, `deep-fast`, `deep-full`} = **86 generations, 0 errors**.
- **Baseline ("me")**: per question, a manual BM25 grep + synthesis — the bar the
  project must clear.
- **Judge**: Claude (never the model grading itself), one judge per question,
  scoring every variant on completeness / correctness / grounding (1–5) + a
  hallucination flag, grounded against a reference context.

## ⚠️ Methodology caveat (read before the numbers)

The judge grounded each answer against the **baseline's own** BM25 reference. The
baseline was written *from* that reference, so it scores near-perfect by
construction, while `deep` mode draws from a **much wider** gather — its
real-but-outside-the-reference facts get flagged as "unsupported / hallucinated."
So the head-to-head is **biased toward the baseline**, and deep-mode scores are
**under-measured**. The fix is already applied to the harness (`generate()` now
saves each answer's own retrieved contexts) so the next iteration judges grounding
fairly. Treat the raw "project loses 19/20" as an upper bound on the gap, not the
truth.

## Offline retrieval ablation (entity questions, no server)

- `vector-only` **P@10 0.975** > `fused` 0.925 > `bm25-only` 0.30.
- Same handle, **terse** query → BM25 P@10 **1.00**; **verbose** natural-language
  query → **0.28**. BM25's exact-match power is real but destroyed by verbose
  queries (common words dominate its score sum).
- `score_weight=0` (pure RRF) collapses to 0.71 → **score-aware fusion is
  essential** (confirms the earlier change).
- A tried-and-**failed** fix: top-IDF term selection for the BM25 arm → 0.02. In
  this corpus the entity is high-frequency (low IDF); scaffold words are rare
  (high IDF), so IDF picks the wrong tokens. The entity needs **detection**, not
  IDF — which `deep`/`enhanced` already do (entity extraction → terse retrieval).
- `candidate_k` floor 50→**200**: ~+0.016 nDCG@10 at negligible cost. **Applied.**

## E2E findings

Raw judge aggregates (reference-biased — see caveat):

| variant | mean(compl/corr/grnd) | hallucination |
|---|---|---|
| baseline (me) | 4.92 | 0/20 |
| 12b / simple | 2.58 | 9/20 |
| 12b / enhanced | 2.48 | 14/20 |
| 12b / deep-fast | 2.04 | 16/16 |
| e2b / * | ~1.8 | high |

**Eyeball-verified, real (not artifact):**
1. **`e2b` leaks chain-of-thought into the answer** ("Пользователь просит… Я должен
   проанализировать… Просмотрю источники: [1]…") *and* fabricates source
   descriptions. e2b is unusable for this pipeline as-is — recommend 12b/26b.
2. **Over-refusal compounded by weak retrieval.** On refused factoids, hybrid
   top-16 surfaced only 1–4 relevant contexts (baseline's BM25-30 + terse entity
   queries found more), and the model then refused even with 3–4 relevant. One
   refusal (a term genuinely absent) was **correct** — so not all refusals are bugs.
3. **`enhanced` occasionally adds unsupported claims** with a fabricated citation.
   `enhanced` did not beat `simple` on the refused factoids and scored slightly
   lower with more hallucination.

**Under-measured but good (artifact of the biased judge):**
4. **`deep` mode never refuses** (0 across all deep configs vs ~20% for
   simple/enhanced) and produces coherent, structured portraits/aggregations —
   including the specific portrait case that historically failed. Its low judge
   scores are largely the reference bias.

**Cross-model:** honesty/quality ranks `26b > 12b > e2b` (bigger = less
hallucination, more willing to refuse cleanly); e2b is the clear outlier (CoT leak).

**Latency ladder (12b):** simple ~14s · enhanced ~25s · deep-fast ~177s ·
deep-full ~463s. deep-full's extra ~5 min over deep-fast did not clearly win.

## Applied this pass

- `retrieval.py`: `candidate_k` floor 50→200 (offline-proven).
- `tools/eval/runner.py`: `generate()` saves each answer's retrieved contexts →
  fair grounding for the next iteration.

## Validated next targets (each needs a fair-eval A/B before shipping)

1. **Entity-aware BM25 arm** — detect `@handle` / quoted / CVE-like tokens and give
   BM25 a terse entity query (keeps exact-match power, stops verbose dilution).
   Must not regress factoid/exact.
2. **Wider / better grounding context** for the simple path (top_k, dedup) so fewer
   spurious refusals when relevant chunks exist.
3. **Refusal-guard tuning** — extract partial/related facts instead of a bare "нет
   ответа" when some relevant contexts are present, *without* increasing
   hallucination. Risky → A/B first.
4. **e2b handling** — strip leaked reasoning or de-recommend e2b.
5. **Fair-eval v2** — re-judge grounding against each answer's own saved contexts
   (harness now supports this) to remove the reference bias and re-measure deep mode.
