# Eval run — findings (2026-07-03)

First full pass of the measurement loop against a large real-world corpus
(~455k chunks, mixed chat + notification data). All corpus content stays local
and gitignored; this report is deliberately data-agnostic (no entities, no text).

## ⭐ Sprint result (authoritative — K=3 multi-judge, apples-to-apples)

The fix sprint (T1 over-refusal, T3 portrait attribution, T4 leaked-CoT detect,
T5/B1 entity-aware BM25) measured before/after with the **same** K=3 multi-judge:

| metric | before (pre-sprint) | after (shipped) |
|---|---|---|
| project **wins** vs manual-grep | 4 | **8** (doubled) |
| losses / ties | 15 / 1 | 12 / 0 |
| project mean | 3.35 | **3.75** (+0.40) |
| hallucinations (majority) | 4/20 | **3/20** |

Reliability: generation is **deterministic** (4 runs of a question → byte-identical),
so the only eval noise is the judge; the baseline (identical answers both runs)
wobbled just 0.16 = the judge-noise floor, so the project's +0.40 is real. 17/20
verdicts unanimous across 3 judges. New wins came from T1+B1 (f1 found the literal
label the grep missed) and T3 (the historically-failing portrait case now wins).

**Still-real losses** (unanimous): enum term-drift (e3 «домены», e4), causal
completeness (c1/c3/c4), and 2 genuine fabrications (e2, p2). **B2** (deep enum
anti-drift prompt) was tried and **reverted** — with deterministic generation its
effect is measurable and 4 judges confirmed it didn't fix the drift.

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

## Two passes: biased → fair (read this before the numbers)

**Pass 1 (biased):** the judge grounded each answer against the *baseline's own*
narrow BM25 reference → baseline scores near-perfect by construction; `deep` mode's
wider-gather facts get flagged as "hallucinated." Result: project loses 19/20,
deep-fast hallucinates 16/16. **This is an artifact — do not trust it.**

**Pass 2 (fair, authoritative):** re-ran 12b in each question's best mode with its
**own** retrieved contexts saved, and re-judged grounding against those (deep-mode
facts consistent with the domain + own contexts are NOT counted as fabrication).

**Fair project-vs-me: 6 wins / 13 losses / 1 tie.** Mean 3.38 (project) vs 4.80
(baseline). Hallucinations 3/20 (all portrait) — down from the biased 16/16.

| task type | project wins | project mean | baseline mean | pattern |
|---|---|---|---|---|
| causal | 2 / 5 | **3.80** | 4.87 | strongest — deep reasoning competitive |
| portrait | 2 (+1 tie) | 3.22 | 4.83 | wins incl. the historically-failing case; 3/6 fabricate traits |
| enumeration | 1 / 5 | 3.40 | 4.93 | deep **drifts** — misreads terms, answers an adjacent question |
| factoid | 1 / 4 | 3.08 | 4.50 | **over-refusal** is the biggest loss driver |

The project genuinely **wins 6** — including a factoid where deep retrieval found a
threshold-breach fact the manual grep missed, and the portrait case that used to
fail. It **loses 13** on three precise, real weaknesses (below). Careful manual
analysis still edges it overall (~1.4 pts, not the ~2.3 the biased pass implied).

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

## E2E findings (from the fair pass)

Three real, precisely-characterized weaknesses drive the 13 losses:

1. **Factoid over-refusal** (biggest loss driver; project 3.08 vs 4.50). On
   f1/f3/f4 the model returns a bare "нет ответа" even though its *own* contexts
   held directly relevant facts (multi-account spray, ЕСИА/urent traffic, WAF spam
   rules). The baseline "won" by being *helpful* — reconstructing the related facts
   and honestly noting the exact label doesn't exist. One refusal (f2-adjacent, a
   term genuinely absent) is legitimate — not all refusals are bugs. Note: f2
   itself the project **won**, finding a threshold-breach fact the grep missed.
2. **Deep-mode topic drift on enumeration** (project 3.40 vs 4.93). Deep gathers
   widely but sometimes answers an *adjacent* question: misread "домены" as subject
   areas instead of DNS/FQDNs (e3), substituted «голосование»→«согласование» (e5),
   misattributed chat @handles as compromised accounts (e2). Coverage is broad but
   off-target — a precision/grounding problem, not lack of recall.
3. **Portrait fabrication** (3/6 portraits flagged: p1/p2/p4). Deep portraits
   invent traits/attributions beyond the contexts. Yet it also **wins** portraits
   (p3 — the historically-failing portrait case — and p5; p6 tie), so deep portraits
   are high-variance: excellent or fabricating.

**Strength:** `causal` is the project's best type (3.80, 2 wins) — deep multi-hop
reasoning is genuinely competitive with careful manual analysis. And `deep` mode
**never refuses** (0/16 vs ~20% for simple/enhanced).

**`e2b` is unusable** (separate from the above): it leaks chain-of-thought into the
answer ("Пользователь просит… Я должен… Просмотрю источники: [1]…") and fabricates
source descriptions. Cross-model honesty ranks `26b > 12b > e2b`.

**Latency ladder (12b):** simple ~14s · enhanced ~25s · deep-fast ~177s ·
deep-full ~463s. deep-full's extra ~5 min over deep-fast did not clearly win.

## Applied this pass

- `retrieval.py`: `candidate_k` floor 50→200 (offline-proven, +0.016 nDCG@10).
- `tools/eval/runner.py`: `generate()` saves each answer's retrieved contexts →
  enabled the fair pass; keep for all future eval.

## Next targets, ranked by fair-pass impact (each needs an A/B before shipping)

The harness now makes each of these a ~30-min measure-change-remeasure loop.

1. **Refusal-guard tuning (biggest win).** When the exact thing isn't in sources but
   *related* facts are, present them + note the gap instead of a bare "нет ответа".
   This alone would flip several factoid losses. Risk: hallucination ↑ — A/B first.
2. **Deep-mode question fidelity.** Deep drifts to adjacent topics on enumeration
   (misreads domain terms, substitutes concepts). Tighten the map/reduce prompt to
   re-anchor on the literal question; consider a "does this answer the asked
   question?" check in reduce.
3. **Portrait anti-fabrication.** Deep portraits are high-variance (3/6 fabricate).
   Strengthen the reduce guard: only assert traits with ≥N supporting fragments;
   attribute by identifier, not invented ФИО (the existing anti-conflation rule
   needs teeth).
4. **Entity-aware BM25 arm** — detect `@handle`/quoted/CVE tokens → terse BM25 query
   (keeps exact-match power, stops verbose dilution). Must not regress factoid/exact.
5. **De-recommend / fix `e2b`** — it leaks reasoning into the answer; recommend
   12b/26b, or strip leaked CoT.

Done this run: candidate_k tuning, fair-eval methodology (bias found + corrected).
