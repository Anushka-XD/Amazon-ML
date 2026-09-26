## Why key-based blocking saturated, and what replaces it

Measured on real missed truth pairs:

- **61.6%** of missed pairs were produced by the blocking join and then discarded by a cap;
  **0.0%** were signal-absent.
- Recall saturates at **0.884** across per-S1 cap 200 -> 3200 (candidates/S1 flattens at ~235),
  so the per-S1 cap is the binding constraint, not the key set.
- **Romanization is not the answer**: across 6 schemes, mean fuzz 61-68, only **9.9%** of
  cross-script pairs share a romanized token. The address is the real bridge (**97.5%** of
  cross-script pairs share an address token).
- **Dense retrieval is.** `intfloat/multilingual-e5-small` (384-dim, 118M, MIT): true
  cross-script pairs mean cosine **0.929** (p10 0.890) against unrelated **0.822** (p90 0.850).
  Of the pairs blocking *currently misses*, **100%** clear cosine 0.85 and 96.3% clear 0.90.

So the residual recall gap is a **similarity-measure** gap, not absent signal — which is what
makes >= 0.99 plausible in principle, and why embeddings rather than more key passes are the
remaining lever.

# Results

Authoritative metric: macro F_0.5 (beta = 0.5), per Source 1 entity, singletons included.
The only true score is produced by the challenge portal from `output/matching_results.tsv`.

**Evaluation protocol (binding):** every macro F0.5 below is computed on the **complete
held-out candidate set** for grouped-split Source 1 entities, using the exact inference-time
decision rule. The 4:1 sampled training distribution is never used to report a score.

## Status: no new F0.5 measured yet

Work in progress targets **macro F0.5 >= 0.99 including France**. What has been measured this
session is **blocking recall**, not F0.5:

| Quantity | Value |
|---|---|
| Blocking recall, baseline (cap=200) | 0.8142 |
| Blocking recall, cap=400 + `house_number` fix | 0.8714 |
| Blocking recall, + address passes 11-14 | **0.8836** |
| Oracle macro F0.5 at recall 0.8836 (perfect matcher) | ~0.975 |
| **Last measured full-candidate macro F0.5** | **0.8488** |
| Last official leaderboard score | 0.811 |
| Target | 0.99 |

Recall is a ceiling, not a score, and 0.8836 is **unvalidated** — no model has been retrained on
the new candidate set, so it is unknown whether the larger candidate set raises F0.5 or dilutes
precision. Macro F0.5 weights precision double, so this must be measured, not assumed.

## Ceiling analysis (why 0.814 recall capped F0.5 at ~0.95)

Inverting `1.25R/(0.25+R) = 0.99` gives R ~= 0.952 per entity **at perfect precision**. Because
misses are entity-correlated rather than pair-independent, the pair recall actually needed for
0.99 is ~0.97+. Two further terms are arithmetic, not tunable:

- **Singletons**: 5.59% of Source 1 entities are singletons, worth 1.0 for predicting empty and
  0.0 for a false merge. False-merging every one caps macro F0.5 at **0.944**.
- **France** is 14.98% of test Source 1 and has no labels. Leave-one-country-out measured
  0.668 (US->India, *and* a script change) and 0.804 (India->US, same script). France is
  Latin-script like US, so **0.804 is the relevant proxy**; and both LOO runs achieved only
  ~92% *of their own candidate-set ceiling* (0.869 / 0.732), so the LOO figure is itself
  recall-limited and pessimistic.

Decomposing the official 0.811 against the 0.8488 US+India held-out estimate implies France is
currently scoring roughly **0.60** — worse than the LOO proxy, consistent with the address-key
defects below. France is therefore the single largest identified source of lost score, not a
15% rounding error.

## Official leaderboard result

- **Public leaderboard macro F0.5 = 0.811** (submitted 26 Sep 2026, 02:43 PM IST; status: Evaluated).
- This is the real Portal score on the public test split, computed from `output/matching_results.tsv`
  (one-to-one, threshold 0.925).
- It lands inside the predicted **0.80–0.85** band and slightly below the held-out full-candidate
  estimate (0.8488). The gap is consistent with the ~15% unseen-France slice and public/private
  split differences.

## Headline

| Evaluation | macro F0.5 | Notes |
|---|---|---|
| **Official leaderboard (public, Portal)** | **0.811** | real score, 26 Sep 2026 |
| 4:1 sampled split (train, grouped) | 0.9807 | **optimistic, not leaderboard-comparable** |
| **Full candidates, held-out S1 (test-like)** | **0.8488** | 95% CI 0.8481–0.8496; over-estimates by ~0.04 |
| Unseen-country proxy (train US -> India) | 0.6684 | France proxy, **lower bound** — confounds a script change |
| Unseen-country proxy (train India -> US) | 0.8041 | France proxy, **same script — the relevant estimate** |
| Full model, US val entities | 0.8905 | in-domain |
| Full model, India val entities | 0.7885 | in-domain |

Realistic leaderboard expectation (now confirmed): **~0.80–0.85**. France is ~15% of the test set,
has no labels, and an unseen country costs 0.09–0.12 F0.5 (LOO). Candidate recall ceiling on the
held-out set is **0.8142**, which bounds the maximum achievable score.


## Full-candidate held-out details (`DATA/reports/eval_full_candidates.json`)

- 438,499 held-out Source 1 entities; 60,912,676 candidate pairs; 1,524,017 truth pairs of which
  1,240,887 found (ceiling 0.8142).
- Threshold-only: 0.8475 @ 0.925. One-to-one: **0.8488 @ 0.925** (chosen).
- Per country: US 0.889, India 0.788. Singletons in val: 23,182.

## 4:1 sampled details (`DATA/reports/eval_marks.json`)

- Threshold-only 0.98033 @ 0.700; one-to-one 0.98075 @ 0.675.
- Precision 0.988, recall 0.977, 40,967 singleton entities, 1,352 false merges on singletons.
- Included only to show the optimism gap versus the full-candidate number.

## Leave-one-country-out (`DATA/reports/eval_loo.json`)

- train US -> validate India: 0.6684 (ceiling 0.7322).
- train India -> validate US: 0.8041 (ceiling 0.8690).
- Full model: US 0.8905, India 0.7885.
- Interpretation: cross-country transfer loses 0.086–0.121 F0.5.

## Test submission (`output/`)

- `matching_results.tsv`: 1,732,544 rows total, 200,982 empty (singletons), 1,533,562 non-empty.
- `candidate_pairs.tsv`: 1,732,544 rows, 554 empty, 1,731,990 non-empty.
- 5,071,867 candidate pairs scored above threshold 0.925.
- Method: one-to-one post-process, threshold 0.925.
- Official validator: **PASS**.

## Reproduce

```
ber.cli prepare
ber.cli block --split both
ber.cli audit --split train
ber.cli features --split train --workers 8 --combine
ber.cli train
ber.cli features --split test --workers 8
ber.cli predict --split test --one-to-one --threshold 0.925
ber.cli evaluate        # 4:1 marks
ber.cli validation --workers 8   # believable full-candidate mark
ber.cli loo             # unseen-country proxy
```

## Leaderboard upload

Per PROBLEM_STATEMENT.md §13, the leaderboard submission is **only** `output/matching_results.tsv`:

- Tab-separated, header exactly `source1_entity_id<TAB>matched_entity_ids`.
- One row per test Source 1 entity (1,732,544 rows), empty `matched_entity_ids` for singletons.
- This is the file uploaded in the Portal; public and private leaderboards are computed from it
  (public = a subset, private = the remainder; final rankings use the private split).
- A byte-identical upload copy is staged at `dist/leaderboard_upload/matching_results.tsv`.
- `candidate_pairs.tsv` is not scored on the leaderboard; it belongs to the final submission zip.
- **Submitted 26 Sep 2026, 02:43 PM IST — result: macro F0.5 = 0.811 (Evaluated).**

## Next milestone (M2) — target

Plan: `docs/superpowers/plans/2026-09-26-precision-colab.md`. Headroom analysis: the blocking ceiling
(0.814 pairs) is worth ~0.95 in macro F0.5, so ~0.10 of the current score is matcher precision/recall;
India (0.788) has the most headroom. Workstream: local diagnostic + char n-gram features + per-country
thresholds/singleton calibration, then Colab T4 multilingual embedding cosine (+ optional
cross-encoder rerank), conditionally MinHash-LSH re-blocking. Target held-out > 0.90, leaderboard > 0.85.


