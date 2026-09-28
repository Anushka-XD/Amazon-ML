# Graph Report - Amazon-gunjan   (2026-09-26)

## Corpus Check
- 57 files · ~42,565 words
- Verdict: corpus is large enough that graph structure adds value.
- Unclassified: 3 file(s) not represented in the graph (top: (none) 2, .ini 1)

## Summary
- 296 nodes · 690 edges · 13 communities (12 shown, 1 thin omitted)
- Extraction: 94% EXTRACTED · 6% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.87)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `84a20887`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- Decisions and Reasoning (DECISIONS.md)
- prepare.py
- validation.py
- features.py
- pandas
- test_score.py
- blocking.py
- cli.py
- bench_gbdt.py
- eda_dataset.py
- validate_submission.py
- graphify.js

## God Nodes (most connected - your core abstractions)
1. `Decisions and Reasoning (DECISIONS.md)` - 19 edges
2. `M2 Precision + Colab Plan (10 Tasks)` - 18 edges
3. `main()` - 15 edges
4. `prepare_frame()` - 13 edges
5. `AGENTS.md Working Notes` - 12 edges
6. `Repo README` - 12 edges
7. `M2 Precision + Colab Design Spec` - 12 edges
8. `normalize_name()` - 11 edges
9. `evaluate_loo()` - 11 edges
10. `parse_address()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Results Metrics` --references--> `Milestone Version Tags (1.4.0-2.0.0)`  [INFERRED]
  docs/RESULTS.md → RULES.md
- `Pinned ML Stack (pandas 3.0.6, lightgbm 4.7.0, duckdb 1.5.5)` --conceptually_related_to--> `LightGBM Matcher (D5)`  [INFERRED]
  code/business_entity_resolution/requirements.txt → docs/DECISIONS.md
- `candidate_pairs.tsv Contract` --shares_data_with--> `Blocking + Classifier Approach (D1)`  [INFERRED]
  PROBLEM_STATEMENT.md → docs/DECISIONS.md
- `ber.cli Staged Pipeline (prepare/block/audit/features/train/predict)` --references--> `Blocking + Classifier Approach (D1)`  [INFERRED]
  code/business_entity_resolution/README.md → docs/DECISIONS.md
- `Char N-Gram TF-IDF Cosine Features (D14)` --conceptually_related_to--> `33-36 Pairwise Feature Set`  [INFERRED]
  docs/DECISIONS.md → Documentation_template.md

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Blocking to Recall Ceiling Flow** — ten_blocking_passes, blocking_cap, train_recall_08115, blocking_recall_ceiling_0814, candidate_pairs_contract [EXTRACTED 0.80]
- **M2 Precision + Colab Flow** — char_ngram_features_d14, per_country_thresholds_d15, singleton_calibration, per_pair_cosine_interface_d13, adoption_gates [EXTRACTED 0.80]
- **Submission Output and Validation Flow** — matching_results_contract, candidate_pairs_contract, submission_validator, submission_tree, official_leaderboard_0811 [EXTRACTED 0.85]

## Communities (13 total, 1 thin omitted)

### Community 0 - "Decisions and Reasoning (DECISIONS.md)"
Cohesion: 0.09
Nodes (64): M2 Adoption Gates (numeric comparisons vs 0.8488), AGENTS.md Working Notes, Per-S1 Candidate Cap 200 + Per-Pass pass_caps, Blocking + Classifier Approach (D1), Blocking Recall Ceiling 0.814, candidate_pairs.tsv Contract, Char N-Gram TF-IDF Cosine Features (D14), Pinned Requirements (+56 more)

### Community 1 - "prepare.py"
Cohesion: 0.08
Nodes (42): AddressParts, parse_address(), _postal_re(), _clean_chars(), detect_script(), fold_name(), name_tokens(), normalize_name() (+34 more)

### Community 2 - "validation.py"
Cohesion: 0.14
Nodes (29): build_training_pairs(), _explode_truth(), grouped_split(), DataFrame, _feature_sources(), _load_booster(), predict_parts(), run_predict() (+21 more)

### Community 3 - "features.py"
Cohesion: 0.17
Nodes (17): attach_country(), _combine_feature_parts(), compute_features(), _country_map(), _feature_block(), _pair_set_metrics(), _pairs_has_label(), _phase1_merged() (+9 more)

### Community 4 - "pandas"
Cohesion: 0.15
Nodes (18): audit_candidates(), _connect(), _explode_truth(), load_country_map(), _Progress, DataFrame, Reference (in-memory pandas) implementation. Kept for tests/cross-checks., run_audit() (+10 more)

### Community 5 - "test_score.py"
Cohesion: 0.18
Nodes (14): evaluate_marks(), macro_f05(), _one_to_one_mask(), one_to_one(), DataFrame, entity_f05(), _entity_f05_from_codes(), sweep_threshold() (+6 more)

### Community 6 - "blocking.py"
Cohesion: 0.28
Nodes (13): block_keys(), compute_token_idf(), generate_candidates(), _join_sql(), load_token_idf(), normalize_keys(), _pass_caps(), run_block() (+5 more)

### Community 7 - "cli.py"
Cohesion: 0.14
Nodes (19): _default_config(), main(), Config, write_inference_pairs(), write_training_pairs(), run_prepare(), train_model(), cfg() (+11 more)

### Community 8 - "bench_gbdt.py"
Cohesion: 0.25
Nodes (14): rapidfuzz, sklearn_metrics, best_entity_f05(), build_pairs(), compute_features(), entity_macro_f05(), load_positives(), load_s1_sample() (+6 more)

### Community 9 - "eda_dataset.py"
Cohesion: 0.43
Nodes (6): cached(), check_membership(), main(), pct(), scan_ground_truth(), scan_source()

### Community 10 - "validate_submission.py"
Cohesion: 0.18
Nodes (15): argparse, examples(), load_match_targets(), main(), ML Challenge 2026 — Submission Validator Run this BEFORE submitting. It checks…, Validate the submission output(s); return ``(errors, warnings)`` lists.…, Return the set of first-column entity IDs from a source TSV. The header row is…, Return a short, human-readable sample of ``items`` for an error message. (+7 more)

### Community 12 - "graphify.js"
Cohesion: 0.40
Nodes (3): IMPORTANT: keep the reminder string free of backticks and $(...) constructs., ref_fs, ref_path

## Knowledge Gaps
- **4 isolated node(s):** `Per-S1 Candidate Cap 200 + Per-Pass pass_caps`, `Full-Candidate Held-Out Validation (D9)`, `4:1 Negative Sampling (D6)`, `Challenge Documentation Template`
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 36 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `cli.py` to `validation.py`, `features.py`, `pandas`, `test_score.py`, `blocking.py`?**
  _High betweenness centrality (0.015) - this node is a cross-community bridge._
- **What connects `Per-S1 Candidate Cap 200 + Per-Pass pass_caps`, `Full-Candidate Held-Out Validation (D9)`, `4:1 Negative Sampling (D6)` to the rest of the system?**
  _4 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `Decisions and Reasoning (DECISIONS.md)` be split into smaller, more focused modules?**
  _Cohesion score 0.08630952380952381 - nodes in this community are weakly interconnected._
- **Should `prepare.py` be split into smaller, more focused modules?**
  _Cohesion score 0.0783673469387755 - nodes in this community are weakly interconnected._
- **Should `validation.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13903743315508021 - nodes in this community are weakly interconnected._
- **Should `cli.py` be split into smaller, more focused modules?**
  _Cohesion score 0.13675213675213677 - nodes in this community are weakly interconnected._