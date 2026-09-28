# Amazon ML Challenge 2026 — Business Entity Resolution

Scalable entity resolution pipeline matching noisy business records from **Source 2** and **Source 3** to deduplicated reference **Source 1**. Evaluated by **macro F_0.5** (precision-weighted, computed per Source 1 entity, singletons included).

- **Official Public Leaderboard:** **0.811** macro F0.5 (26 Sep 2026)
- **Local Held-out Full-Candidate Validation:** **0.8488** (95% CI 0.8481–0.8496)
- **Target:** macro F0.5 ≥ 0.99 including zero-shot France

---

## Problem & Key Invariants

Across 2.2M Source 1 reference records and ~10.3M candidate records (Source 2 + Source 3), the pairwise search space is ~10¹³ pairs.

- **Injective Ground Truth:** Each S2/S3 record belongs to at most one S1 entity. Exploited via provably optimal greedy 1-to-1 matching.
- **Open-Set Generalization:** Train set contains `US` and `India`; Test adds `France` (~15%) with zero training labels.
- **Singletons:** 5.585% of S1 entities have zero matches. False merging any singleton scores 0.0 for that entity.
- **Cross-Script Matching:** ~15% of Indian S2 records use non-Latin Indic scripts against ASCII S1 records. Addresses serve as the primary bridge (97.5% shared tokens).

---

## Pipeline Architecture

```
Raw TSVs (S1, S2, S3)
  │
  ├──► prepare (NFKC normalize, legal suffix strip, address/house-no parsing, token IDF)
  │
  ├──► Candidate Generation (Hybrid)
  │      ├── Key Blocking (14 passes in DuckDB, sharded out-of-core, per-pass & per-S1 caps)
  │      └── Dense Retrieval (intfloat/multilingual-e5-small, FAISS top-50, min cosine 0.80)
  │
  ├──► union (Combine blocking + ANN candidates while preserving provenance)
  │
  ├──► features (41 pairwise Rapidfuzz string, token Jaccard, address, and embedding features)
  │
  ├──► train (LightGBM binary classifier, grouped 80/20 entity split, early stopping)
  │
  ├──► predict & postprocess (Score pairs, apply threshold 0.925, enforce injective 1-to-1)
  │
  └──► output/ (matching_results.tsv, candidate_pairs.tsv verified via validate_submission.py)
```

---

## Results & Benchmarks

> **Binding Evaluation Rule:** All reported macro F0.5 scores are evaluated on the **full held-out candidate set** using exact inference rules. The 4:1 training distribution is never quoted as a score.

| Evaluation Mode | Macro F0.5 | Notes |
|---|---|---|
| **Official Leaderboard (Portal)** | **0.811** | Public test split (`matching_results.tsv`, threshold 0.925) |
| **Full Held-Out Candidate Set** | **0.8488** | 438k held-out S1 entities, test-like inference (US 0.889 / India 0.788) |
| **Unseen-Country Proxy (LOO)** | **0.804** | India → US (same Latin script, proxy for France) |
| *4:1 Sampled Set (Optimistic)* | *0.9807* | *Training/validation split only; not comparable to leaderboard* |

**Candidate Recall Progression:** 0.8142 (baseline cap=200) → 0.8714 (cap=400 + house-no fix) → **0.8836** (+ passes 11–14). Dense retrieval (e5-small) bridges the remaining ceiling, with 100% of missed pairs having cosine ≥ 0.85 (mean 0.965).

---

## Setup & Requirements

- **Runtime:** Python 3.12 (tested on macOS Apple Silicon & Linux).
- **Memory:** Host-friendly (8 GB RAM / 8 cores); all DuckDB memory limits are dynamically managed via `config.json`.
- **Dependencies:** LightGBM, DuckDB, Rapidfuzz, PyArrow, FAISS, Sentence-Transformers, PyTorch.

```bash
# 1. Create and activate venv
uv venv .venv --python 3.12
source .venv/bin/activate

# 2. Install dependencies
uv pip install -r code/business_entity_resolution/requirements.txt
uv pip install sentence-transformers faiss-cpu torch

# 3. (macOS only) Symlink OpenMP library for LightGBM / XGBoost
ln -sf $(python -c "import torch,os;print(os.path.join(os.path.dirname(torch.__file__),'lib','libomp.dylib'))") \
  /opt/homebrew/opt/libomp/lib/libomp.dylib
```

---

## Running the Pipeline

Set the Python path and CLI alias:

```bash
export PYTHONPATH=code/business_entity_resolution/src
CLI=".venv/bin/python -m ber.cli --config code/business_entity_resolution/config.json"
```

### End-to-End Workflow

```bash
# Step 1: Preprocess records and parse addresses
$CLI prepare

# Step 2: Dense retrieval embeddings & ANN candidate search
.venv/bin/python tools/run_embed.py --split train --threads 8
$CLI ann --split train --topk 50 --min-cos 0.80

# Step 3: Key-based blocking and candidate union
$CLI block --split train
$CLI union --split train
$CLI recall --split train                       # Audits recall and oracle ceiling

# Step 4: Extract 41 pairwise features and train LightGBM
$CLI features --split train --workers 8 --combine
$CLI train

# Step 5: Full-candidate evaluation and unseen-country validation
$CLI validation --workers 8                     # Unbiased full-candidate score (0.8488)
$CLI loo                                        # Leave-one-out France proxy

# Step 6: Inference on test set & generate submissions
$CLI features --split test --workers 8
$CLI predict --split test --one-to-one --threshold 0.925

# Step 7: Official submission validation
python DATA/student_resource/utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir DATA/student_resource/dataset/test
```

---

## Repository Structure

```
├── code/business_entity_resolution/
│   ├── config.json              # Resource limits, pass caps, thresholds
│   └── src/ber/
│       ├── prepare.py           # Text normalization and address parsing
│       ├── blocking.py          # 14-pass DuckDB candidate blocking
│       ├── ann_retrieve.py      # Multilingual embedding retrieval
│       ├── union_candidates.py  # Provenance-preserving candidate union
│       ├── features.py          # Parallel 41 pairwise similarity features
│       ├── train.py             # LightGBM training & grouped split
│       ├── predict.py           # Thresholding and injective 1-to-1 matching
│       ├── audit.py             # DuckDB recall & reduction audit
│       └── validation.py        # Full-candidate bootstrap evaluation
├── docs/                        # RESULTS.md, DECISIONS.md, PROJECT_LOG.md
├── tools/                       # Embedding generators and diagnostic scripts
└── tests/                       # Pytest test suite (68 tests passing)
```

Run tests at any time with:
```bash
PYTHONPATH=code/business_entity_resolution/src .venv/bin/python -m pytest -q
```

