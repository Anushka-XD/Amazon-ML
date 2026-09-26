# AGENTS.md — working notes for this repository

Amazon ML Challenge 2026: Business Entity Resolution. Read `RULES.md` (binding rules), `PROBLEM_STATEMENT.md` (full spec), and `DATA/student_resource/dataset/DATASET.md` (measured data facts) before changing code.

**Current objective:** macro F0.5 >= 0.99 including France. See `docs/RESULTS.md` for what is and is not
measured, and note the binding evaluation rule below.

Keep documentation and logs current as you work:
- `docs/PROJECT_LOG.md` — chronological run log (commands, timings, numbers).
- `docs/DECISIONS.md` — architecture decisions and reasoning.
- `docs/FAILURES_AND_FIXES.md` — failures and resolutions; add an entry for every non-trivial bug.
- `docs/RESULTS.md` — metrics; update when a new believable evaluation is produced.

## Evaluation rule (binding)

**Every reported macro F0.5 must come from the full held-out candidate set**, using the exact
inference-time decision rule, with grouped splitting on Source 1 entity. The 4:1 sampled training
distribution is for training only and must never be quoted as a score. Report
`F(US) / F(India) / F(France) / weighted total` separately so an open-set country cannot be
averaged away.

## Environment

- ML code runs on **Python 3.12** via the project venv. This checkout is **macOS**: use
  `.venv/bin/python` (the `Scripts\python.exe` paths in older docs are Windows-only).
- **This host has 8 GB RAM and 8 cores**, not the 23 GB Windows box the older docs assume. All
  DuckDB limits are config-driven in `config.json` under `db` (`memory_limit`, `threads`,
  `max_temp`, `block_shards`). Do not hardcode them again — hardcoded 12 GB limits made every
  stage OOM here.
- LightGBM and XGBoost need `libomp.dylib`, which macOS does not ship. Resolved by symlinking
  PyTorch's LLVM OpenMP: `ln -sf $(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib','libomp.dylib'))") /opt/homebrew/opt/libomp/lib/libomp.dylib`.
- Disk is the binding resource (~25 GB free). Regenerate rather than keep: `test_*` processed
  parquet costs 2.2 GB and is rebuilt by `prepare` in ~25 min.
- Tests: `.venv/bin/python -m pytest -q` from the repo root (63+ tests). `conftest.py` puts `src/` on `sys.path`.
- CLI: `PYTHONPATH=code/business_entity_resolution/src .venv/bin/python -m ber.cli <command> --config code/business_entity_resolution/config.json`.
  Commands: `prepare`, `block`, `ann`, `union`, `recall`, `audit`, `features`, `train`, `tune`,
  `predict`, `outputs`, `evaluate`, `validation`, `loo`, `diagnose`, `all`.

## Blocking contracts

- Passes: 1 exact normalized name; 3 rare-token exact; 4 `house_no|<rarest street token>[:5]`;
  5 `postal|first name token`; 6 fallback `state|name[:3]`; 7 name-token prefix-5;
  8 rare street-token prefix-5; 9 token pairs over the 4 rarest tokens; 10 token triples over the
  3 rarest; **11 `h{house_no}`**; **12 exact street token**; **13 `postal|<street>`**;
  **14 accent-folded name prefix-5** (France: `name_norm` keeps accents, so `Café`/`Cafe` never joined).
- **Street keys select the rarest street token by IDF, never `street_tokens[0]`.** The first token
  is a grammatical particle in French (`Rue de la Paix` → `de`) and a filler in many Indian
  addresses, so first-token keys collide across a huge share of records and are always dropped.
- Per-S1 selection is `ORDER BY block_score DESC, n_passes DESC` — **not** `pass_id`, and
  `block_score` must lead. `n_passes` first lets a fuzzy multi-pass pair outrank an exact-name
  match and push it outside the per-S1 cap (see F19).
- Per-pass caps live in `config.json` under `pass_caps`; per-S1 `cap` is the top-level key.
  These are coupled: widening block caps *lowers* recall unless the ranking is quality-aware.
- **The shard predicate must sit on the Source 1 side, before the join** (`F18`), and
  `_prepare_cand_keys` pre-filters candidate keys to probed blocks once, globally.

## Dense retrieval contracts (new)

- Model: `intfloat/multilingual-e5-small`, 384-dim, 118M params, **MIT** (satisfies `RULES.md`
  §1.2: ≤8B, MIT/Apache). Pinned in `tools/run_embed.py`.
- `DATA/emb/{split}/shard_*.npy` (fp16, 250k rows each) + a parallel `ids.txt`. Resumable:
  existing shards are skipped. ~9 h for the 12.5M train records on this host at ~380 texts/s.
- `tools/run_ann_retrieve.py` / `ber.ann_retrieve.py` produce
  `DATA/candidates/{split}_ann_candidates.parquet` (`s1_id, cand_id, emb_cos`).
- `ber/union_candidates.py` unions the key and retrieval sets into
  `{split}_union_candidates.parquet`, **preserving provenance** (`pass_id=0` means
  retrieval-only, `emb_cos=-1` means key-only). The two sources fail differently and their
  agreement is a precision signal, so do not merge them into one opaque list.
- `ber.cli recall --split train` measures recall for whichever candidate set exists, and reports
  the oracle macro F0.5 upper bound.
- **Why this exists:** key blocking saturates at 0.884 and romanization recovers almost nothing
  (9.9% shared tokens). Of the truth pairs key blocking misses, 100% clear cosine 0.85 under this
  model (mean 0.965) versus 0.822 for unrelated pairs — the gap is a similarity-measure gap, not
  absent signal.

## Pipeline data contracts (all on disk, parquet/JSON)

- `DATA/processed/{split}_source{1,2,3}.parquet` — normalized records from `prepare`. Key columns: `entity_id`, `name_norm`, `name_fold`, `name_roman`, `name_tokens`, `name_idf_tokens`, `name_script`, `name_stripped`, `suffix_class`, `addr_norm`, `addr_raw_missing`, `house_no`, `street_tokens`, `postal`, `state_key`, `landmark_flag`.
- `DATA/processed/train_ground_truth.parquet` — official columns `source1_entity_id`, `matched_entity_ids`.
- `DATA/reports/{split}_token_idf.json` — `{"n", "min_idf", "idf": {token: idf}}` for every token (used for pass-3 rarity and top-4 rarest ranking).
- `DATA/keys/{split}_s1_keys.parquet`, `{split}_source{2,3}_keys.parquet` — blocking keys `(entity_id, pass_id, key, block_score)`.
- `DATA/candidates/{split}_candidates.parquet` — `(s1_id, cand_id, pass_id, block_score, is_s2)`, one row per final candidate pair after dedup and the per-S1 cap. This is exactly the set the matcher scores.

## Blocking contracts

- Passes: 1 exact normalized name; 3 rare-token exact; 4 `house_no|street[0][:5]`; 5 `postal|first name token`; 6 fallback `state|name[:3]`; 7 name-token prefix-5; 8 street-token prefix-5; 9 token pairs over the 4 rarest tokens; 10 token triples over the 3 rarest tokens.
- Per-pass candidate-block caps live in `config.json` under `pass_caps` (keys are strings). Per-S1 cap is `cap` (200). `max_block` is not used by the current join; `_valid_sql` uses the pass caps.
- Tuning loop: edit `pass_caps` → `ber.cli block --split <train|test>` (keys are cached; deleting `DATA/keys/` forces regeneration, ~20 min for train) → `ber.cli audit --split train` (~50 s) → read recall. Current operating point: train recall **0.8115** (US 0.868, India 0.727), 304.8M candidates, reduction 29.5.
- Never run `block` and `audit` concurrently with another heavy job (RAM/disk contention; DuckDB uses up to 12 GB and a 50-60 GiB temp directory).

## Audit contract

- Entry point: `ber.audit.run_audit(cfg, split)`, DuckDB end-to-end (never materializes candidate tuples). `audit_candidates()` is a small-scale pandas reference used only by tests; do not delete it and do not overwrite `run_audit`.
- Report path: `DATA/reports/{split}_blocking_audit.json`. Keys: `recall`, `recall_by_country`, `reduction_ratio`, `candidates_per_s1_mean`, `singleton_candidates_mean`, `per_pass_recall`, `truth_pairs`, `found_pairs`.
- `test_run_audit_matches_pandas_reference` asserts the DuckDB path equals the pandas reference field-by-field.

## Gotchas already learned

- Parquet list columns come back from `iter_batches` as numpy arrays, not Python lists.
- `pd.factorize` returns `(codes, uniques)` — the codes are the *first* element, not the second.
- `SEMI JOIN ... USING (...)` fails to bind in this DuckDB build; use an explicit `ON`, or
  `WHERE (a, b) IN (SELECT ...)` for tuple membership.
- List lambdas (`list_transform(x, v -> left(v, 5))`) do not bind here; explode to a token table
  and equi-join instead.
- Per-token Metaphone blocking explodes (75 GiB temp); it was removed. Do not reintroduce it.
- Commas are data: always `sep="\t"`, `keep_default_na=False`.
- Never quote a 4:1-sampled number as a score. It is training-only.
- A ceiling figure must state its predicate: "shares a token" is not "shares a *usable* key"
  (`F17`). Blocks over the per-pass cap are always dropped, so common tokens recover nothing.
- Generated datasets/keys/candidates/embeddings are git-ignored; never commit them.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
