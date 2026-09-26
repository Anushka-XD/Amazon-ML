#!/usr/bin/env bash
# End-to-end recall + score pipeline, ordered so the cheap gate comes first.
#
# Order matters: the recall gate is the whole point of the embedding work, and it costs
# minutes once the vectors exist. Training and the full-candidate validation only run
# after recall clears, because a model trained on a candidate set that is about to be
# rebuilt is wasted compute.
#
# Every stage is independently resumable; re-running skips completed work.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=code/business_entity_resolution/src
PY=.venv/bin/python
CLI="$PY -m ber.cli --config code/business_entity_resolution/config.json"
SPLIT="${1:-train}"
RECALL_GATE="${RECALL_GATE:-0.975}"

log() { printf '\n=== %s ===\n' "$1"; }

log "stage 0: environment"
$PY -c "import duckdb, lightgbm, sentence_transformers, faiss; print('deps OK')"

log "stage 1: embeddings ($SPLIT)"
if compgen -G "DATA/emb/$SPLIT/shard_*.npy" > /dev/null; then
  echo "shards present: $(ls DATA/emb/$SPLIT/shard_*.npy | wc -l | tr -d ' ')"
else
  $PY code/business_entity_resolution/tools/run_embed.py --split "$SPLIT" --threads 8
fi

log "stage 2: dense retrieval"
$CLI ann --split "$SPLIT" --topk 50 --min-cos 0.80

log "stage 3: key-based candidates"
if [ ! -f "DATA/candidates/${SPLIT}_candidates.parquet" ]; then
  $CLI block --split "$SPLIT"
else
  echo "reusing DATA/candidates/${SPLIT}_candidates.parquet"
fi

log "stage 4: union key + retrieval"
$CLI union --split "$SPLIT"

log "stage 5: RECALL GATE"
GATE_JSON="$($CLI recall --split "$SPLIT")"
echo "$GATE_JSON"
RECALL="$($PY -c "import json,sys; print(json.loads(sys.stdin.read().split('=== ')[0])['recall'])" <<< "$GATE_JSON" 2>/dev/null || echo 0)"
echo "recall = $RECALL (gate $RECALL_GATE)"
if [ "$SPLIT" = "train" ]; then
  $PY - "$RECALL" "$RECALL_GATE" <<'EOF'
import sys
r, gate = float(sys.argv[1]), float(sys.argv[2])
if r < gate:
    print(f"GATE FAILED: recall {r:.4f} < {gate}. Do not train - fix recall first.")
    raise SystemExit(1)
print(f"GATE PASSED: recall {r:.4f} >= {gate}")
EOF
fi

log "stage 6: training pairs + features"
$CLI features --split "$SPLIT" --workers 4 --combine

log "stage 7: train"
$CLI train --split "$SPLIT"

log "stage 8: FULL-CANDIDATE held-out validation (never the 4:1 sample)"
$CLI validation --workers 4

log "done"
echo "Read DATA/reports/eval_full_candidates.json for the authoritative macro F0.5."
