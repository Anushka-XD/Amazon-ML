# Failures and Fixes

Every non-trivial failure encountered while building the pipeline, with root cause and resolution.
Kept for future reference so the same dead ends are not re-entered.

## F1 — Per-token Metaphone blocking exploded the temp directory
- **Symptom:** `OutOfMemoryException ... 75.6 GiB/75.6 GiB used`; the run never completed.
- **Cause:** per-token phonetic codes collide at scale (a single generic code was shared by >1.2M
  candidate records), so the pass-2 join produced billions of rows.
- **Fix:** removed the Metaphone blocking pass entirely (`RULES`/`DECISIONS` note: do not
  reintroduce). Recall was recovered with name-token prefix-5 and rare-token pair/triple passes.

## F2 — 64-bucket join loop took >3 hours
- **Symptom:** a per-bucket loop over 64 hash buckets ran for the full 3 h tool timeout.
- **Cause:** each bucket re-scanned the entire key tables (64 full scans) instead of one join.
- **Fix:** single DuckDB join with pass-specific block caps (`_valid_sql`); join size fell from
  819M to 38.5M rows in the first tuning step and candidates were produced in minutes.

## F3 — Pandas audit could not handle 98.8M candidate rows
- **Symptom:** `ber.cli audit` hung/OOMed after blocking grew to ~100M candidates.
- **Cause:** the audit materialized candidate tuples into Python sets/dicts.
- **Fix:** rewrote `run_audit` to be DuckDB end-to-end (read_parquet + joins/group-bys) with
  streaming; ~50 s for the full train audit. `audit_candidates` kept only as a pandas test reference.

## F4 — Parquet list columns came back as numpy arrays
- **Symptom:** `ValueError: The truth value of an array with more than one element is ambiguous`
  in `block_keys`.
- **Cause:** `ParquetFile.iter_batches(...).to_pandas()` returns list columns as `numpy.ndarray`,
  not Python lists; `if house and street:` then fails.
- **Fix:** coerce list columns with `list(x)` before use in `block_keys`.

## F5 — Feature pipeline needed a different metadata split
- **Symptom:** validation pairs lived in `pairs/valfull_pairs.parquet` but metadata/processed
  parquet are named for `train`.
- **Fix:** added an optional `meta_split` parameter to `run_features`/`_phase1_merged`.

## F6 — Prediction TSV aggregation OOM
- **Symptom:** `_duckdb.OutOfMemoryException (7.4 GiB/7.4 GiB used)` while writing
  `candidate_pairs.tsv`.
- **Cause:** a single `string_agg` over 250.6M pairs grouped into 1.73M rows exceeded the memory
  limit.
- **Fix:** bucket the candidate pairs by `hash(s1_id) % 64` once, aggregate each bucket separately
  (bounded memory), then byte-concatenate the parts with a single header.

## F7 — Empty match lists written as `""`, validator FAIL
- **Symptom:** `matched_entity_ids contains IDs without an S2-/S3- prefix: ""` (validator exit 1).
- **Cause:** DuckDB's CSV writer quotes empty strings as `""`; the validator then parses a literal
  empty ID.
- **Fix:** add `QUOTE ''` to both CSV `COPY` options so empty lists are written as true empty fields.

## F8 — `data` vs `DATA` path collision on Windows
- **Symptom:** intermediates landed under `DATA/` even though `config.json` said `data`.
- **Cause:** NTFS is case-insensitive and the challenge directory `DATA/` already existed.
- **Fix:** documented as expected behaviour in `AGENTS.md`; the packaged `src/config.json` uses
  `DATA` explicitly.

## F9 — Normalization stripped Indic combining marks
- **Symptom:** test `normalize_name("राम मार्केटिंग")` returned only consonants.
- **Cause:** the regex `[^\w\s]` does not treat combining marks (category Mn/Mc) as word characters.
- **Fix:** replaced regex cleanup with a `unicodedata`-based `_clean_chars` that keeps letters,
  digits, and marks.

## F10 — Suffix class overwritten by stacked suffixes
- **Symptom:** `"Private Ltd"` yielded suffix class `pvt` instead of `ltd`.
- **Cause:** the pop loop kept overwriting the class.
- **Fix:** keep the first (outermost) suffix class while still stripping all suffix tokens;
  `strip_legal_suffix` normalizes internally (uses `fold_name`).

## F11 — `grouped_split` was O(n·m) (43 h)
- **Symptom:** validation split took ~43 h on the full pair set.
- **Cause:** `np.isin(groups, val_groups)` over every group id.
- **Fix (by prior session):** `pd.factorize` + boolean group mask; ~12 s.

## F12 — Threshold tuned on the wrong distribution
- **Symptom:** deployed threshold 0.675, chosen on the 4:1 sample, produced many false merges when
  scored against the full candidate distribution.
- **Cause:** sampled negatives under-represent the confusable candidate mass present at inference.
- **Fix:** re-tuned on full candidates for held-out Source 1 groups -> **0.925**; outputs regenerated.

## F13 — Blocking recall below target
- **Symptom:** initial blocking recall 0.52, then 0.61, then 0.70, then 0.80.
- **Cause:** tight pass caps dropped common-token blocks; only exact/rare-token passes existed.
- **Fix:** added name-token prefix-5, street-token prefix-5, and rare-token pair/triple passes and
  widened caps; final train recall **0.8115**. India (0.727) remains the weak spot; raising recall
  further (embedding/LSH fuzzy blocking) is the top future work item.

## F14 — Global injective assignment is provably a no-op (dead end, do not retry)
- **Hypothesis:** greedy one-to-one resolves candidate conflicts by local score order, so a
  min-cost-flow / max-weight-matching solve over the whole (S1 x candidate) graph should convert
  into macro F_0.5, because the ground truth is injective on the candidate side.
- **Measured:** built the solver, then validated it against **exhaustive enumeration** on small
  random instances. Greedy one-to-one matched the brute-force optimum in **450/450 trials**
  (300 at 3 S1 x 5 candidates, 150 at 4-6 S1 x 5-8 candidates), 0 mismatches.
- **Why:** the only constraint is that each S2/S3 id is claimed by at most one S1. An S1 entity
  has **no capacity limit** (matches average 3.46, max 11). With no coupling between candidates,
  the objective decomposes per candidate, so "take the highest-probability pair for each
  candidate independently" **is** the exact global optimum. There is no global structure to
  exploit.
- **Also:** the first solver implementation was itself buggy - rerouting an incumbent left its
  old candidate occupied, so a pair could be double-counted and injectivity was violated
  (caught by `test_beats_greedy_on_conflicting_swaps`, which reported weight *above* the
  brute-force optimum - impossible). Recursion also needed a `visited` set to terminate.
- **Resolution:** deleted `ber/assign.py` and its tests. `ber/postprocess.py::one_to_one` is kept
  and is already optimal. Effort redirected to recall, which is the actual binding constraint.
- **When it would matter:** only if S1 entities gained a capacity constraint. They do not have one.

## F15 — ITRANS romanization is a weak cross-script bridge (measured, not assumed)
- **Hypothesis:** blocking keys ignore `name_roman`, so Indic candidate names can never join
  ASCII Source 1 keys. A romanized prefix-5 pass should be worth +0.10-0.13 recall.
- **Measured** on 614 real cross-script truth pairs (`tools/exp_romanization.py`): every scheme
  lands in the same band - itrans/casefold 67.3 mean fuzz, optitrans 67.8, hk 67.2, wx 67.0,
  iso 61.5, slp1 62.6. Only **9.9%** of cross-script pairs share a full romanized token and
  **36.5%** share a 5-char prefix.
- **Conclusion:** scheme choice is not the lever, and the achievable gain is well under half the
  hypothesis. The real bridge is the **address**: 97.5% of cross-script pairs share an address
  token and 96.8% share an address 5-char prefix - statistically indistinguishable from Latin
  pairs (95.4% / 94.7%) - because addresses are frequently ASCII even when the name is not
  (e.g. `स्टार प्रॉपर्टीज प्राइवेट लिमिटेड` vs `Star Properties Private Limited`, address
  token_set=100).
- **Real defect found instead:** `parse_address` extracted `house_no` only when `tokens[0]`
  began with a digit, so `No.107/2, ...`, `E - 100, ...`, `G-26, ...` and `Plot 149 ...` all
  yielded no house number and silently disabled pass 4. Fixed by `ber/address.py::house_number`
  (marker-aware, 12 formats covered by tests) plus `street_key_tokens` (drops street-type words
  so `Lane`/`Ln` and `Boulevard`/`Bd` agree).
- **Side effect:** casefolding ITRANS output (`sharmA` -> `sharma`) was a genuine small win -
  it lets the shared name token actually match. Kept in `ber/translit.py`.

## F16 — Blocking never read `name_roman`, so the "cross-script blind spot" was misdiagnosed
- **Symptom:** India recall 0.727 vs US 0.868, with ~15% of S2/S3 names non-Latin and 100% of S1
  train names ASCII. Looked like a structural name-matching dead end.
- **Cause of the misdiagnosis:** assumed romanization was the bridge. F15 shows it is weak, and
  the address is the real bridge. `fold_name` does not help either - NFKD plus stripping `Mn`
  deletes Devanagari vowel signs, leaving bare consonants (this is the original F9).
- **Actual defect:** `blocking.py::block_keys` builds every pass from `name_norm` /
  `name_idf_tokens` and never reads `name_roman`, so `roman_ratio` only reaches the model *after*
  a candidate exists - too late to help recall.
- **Resolution:** `ber/diagnostic.py` classifies every missed truth pair by cause
  (shared-token / shared-prefix / cap-limited / both-addresses-missing / script-mismatch) and
  reports `recoverable_if_roman_bridge`, so whether a romanized pass is worth adding is decided
  from measurement rather than assumption.

## F17 — The "0.989 address-key ceiling" was measured on the wrong predicate
- **Symptom:** `exp_addr_ceiling.py` reported that an identical street token recovers 91.4% of
  missed truth pairs, implying recall 0.989, and predicted 1.0000 for an exhaustive address pass.
  Actual recall after adding those passes was only **0.8836**.
- **Cause:** the test counted a pair as recoverable whenever *any* address token was shared -
  including `rue`, `main`, `street`, which occur in thousands of records. A block that large is
  always over the per-pass cap and is **always dropped**, so those pairs were never reachable by
  key-based blocking at any cap setting. The metric measured string overlap, not key usability.
- **Fix:** distinguish *shared* from *shared and usable* (block size <= the per-pass cap) before
  quoting a ceiling. Any future ceiling claim must state which predicate it used.
- **Consequence:** the honest statement became "key-based blocking saturates near 0.884", which
  redirected the work to dense retrieval (D18) rather than further key engineering.

## F18 — Sharding was a no-op: the predicate was applied after the join
- **Symptom:** the block join OOMed at 5.5 GB on every shard, including shard 0 of 16.
- **Cause:** `_join_sql` put the `hash(s1_id) % n` filter on the `dedup` CTE, i.e. *after* the
  join. Every shard therefore joined the full Source 1 key set, so sharding reduced nothing.
  A second, related fault: the `valid` block-size aggregate scanned all ~50M candidate keys per
  shard instead of only the keys the shard's Source 1 entities actually probe.
- **Fix:** shard predicate moved onto the Source 1 side before the join, and `candk_f`/`validb`
  pre-filtered globally (145M probed candidate keys -> 5.7M valid blocks in 27 s).
- **Lesson:** a shard filter must reduce the *input* to the expensive operator, not filter its
  output. Measure per-shard row counts, not just total.

## F19 — `n_passes DESC` ranking crowded out exact-name matches
- **Symptom:** missed truth pairs included records with byte-identical names (cosine 1.000 under
  the embedding probe) that pass 1 should have matched.
- **Cause:** ranking by number of agreeing passes let a fuzzy pair proposed by five weak passes
  outrank an exact-name match, which then fell outside the per-S1 cap of 400.
- **Fix:** `ORDER BY block_score DESC, n_passes DESC` so exact names (block_score 1.0) always
  rank first. Recorded as D19.
- **Lesson:** a "clever" secondary ranking key can be worse than the obvious one; agreement
  counts are evidence *given* equal strength, not a replacement for it.
