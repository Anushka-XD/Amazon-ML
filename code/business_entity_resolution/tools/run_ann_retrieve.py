"""ANN retrieval over the multilingual embeddings, unioned with key-based candidates.

Rationale, all measured: key-based blocking saturates at 0.884 pair recall and no cap
tuning moves it, because the residual pairs share no usable key. Romanization is a
weak bridge (9.9% of cross-script pairs share a romanized token; mean fuzz ~67). Dense
retrieval separates cleanly instead - true cross-script pairs score cosine 0.929
(p10 0.890) against 0.822 (p90 0.850) for unrelated pairs, and 84.9% of true pairs
clear 0.90 where 0% of unrelated pairs do.

Retrieval is a *recall* stage: it proposes pairs, and the matcher decides. The two
candidate sources are unioned rather than merged so the agreement between them is
itself available as a feature.

Usage:
    python tools/run_ann_retrieve.py --split train --topk 50 [--min-cos 0.80]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, "code/business_entity_resolution/src")


def log(m):
    print(f"[ann] {m}", flush=True)


def load_shards(emb_dir: Path, want_prefix: str):
    """Load vectors and ids, keeping only entities matching `want_prefix` (S1/S2/S3)."""
    ids_file = emb_dir / "ids.txt"
    all_ids = [line.strip() for line in ids_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    shards = sorted(emb_dir.glob("shard_*.npy"))
    log(f"{len(all_ids):,} ids across {len(shards)} shards")
    keep_mask = np.array([i.startswith(want_prefix) for i in all_ids])
    log(f"  {want_prefix}*: {int(keep_mask.sum()):,}")
    vecs = np.empty((int(keep_mask.sum()), 384), dtype=np.float16)
    out = np.empty(int(keep_mask.sum()), dtype=object)
    pos = 0
    for k, shard in enumerate(shards):
        arr = np.load(shard)
        m = keep_mask[k * 250_000:(k + 1) * 250_000]
        idx = np.flatnonzero(m)
        if len(idx):
            vecs[pos:pos + len(idx)] = arr[idx]
            out[pos:pos + len(idx)] = [all_ids[k * 250_000 + i] for i in idx]
            pos += len(idx)
        del arr
    return vecs[:pos], out[:pos]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--emb", default=None)
    ap.add_argument("--topk", type=int, default=50)
    ap.add_argument("--min-cos", type=float, default=0.80)
    ap.add_argument("--shard", type=int, default=200_000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    emb = Path(args.emb or f"DATA/emb/{args.split}")
    out_path = Path(args.out or f"DATA/candidates/{args.split}_ann_candidates.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_dir = out_path.parent / f"{args.split}_ann_parts"
    if part_dir.exists():
        import shutil
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)

    import faiss

    log("loading S2/S3 candidate vectors")
    cand_v, cand_ids = load_shards(emb, "S2-")
    s3_v, s3_ids = load_shards(emb, "S3-")
    import numpy as _np
    cand_v = _np.concatenate([cand_v, s3_v])
    cand_ids = _np.concatenate([cand_ids, s3_ids])
    del s3_v
    log(f"candidate vectors: {cand_v.shape}")

    index = faiss.IndexFlatIP(cand_v.shape[1])  # vectors are L2-normalised
    t0 = time.time()
    index.add(cand_v.astype("float32"))
    log(f"index built ({cand_v.shape[0]:,} x {cand_v.shape[1]}) in {time.time() - t0:.0f}s")

    log("loading S1 query vectors")
    q_v, q_ids = load_shards(emb, "S1-")
    log(f"query vectors: {q_v.shape}")

    total = 0
    part = 0
    for start in range(0, len(q_ids), args.shard):
        chunk = q_v[start:start + args.shard].astype("float32")
        sims, idxs = index.search(chunk, args.topk)
        rows_s1, rows_cand, rows_cos = [], [], []
        for r in range(len(chunk)):
            for c, j in enumerate(idxs[r]):
                if j < 0:
                    continue
                cos = float(sims[r][c])
                if cos < args.min_cos:
                    break
                rows_s1.append(q_ids[start + r])
                rows_cand.append(cand_ids[j])
                rows_cos.append(cos)
        tbl = pa.table({
            "s1_id": pa.array(rows_s1, pa.string()),
            "cand_id": pa.array(rows_cand, pa.string()),
            "emb_cos": pa.array(rows_cos, pa.float32()),
        })
        pq.write_table(tbl, part_dir / f"part_{part:05d}.parquet")
        total += len(rows_s1)
        part += 1
        if part % 10 == 0:
            log(f"  queried {start + len(chunk):,}/{len(q_ids):,}  pairs={total:,}")
    log(f"DONE: {total:,} ANN pairs -> {out_path}")

    writer = None
    for p in sorted(part_dir.glob("part_*.parquet")):
        for b in pq.ParquetFile(p).iter_batches(batch_size=500_000):
            t = pa.Table.from_batches([b])
            if writer is None:
                writer = pq.ParquetWriter(out_path, t.schema)
            writer.write_table(t)
    if writer is not None:
        writer.close()
    import shutil
    shutil.rmtree(part_dir, ignore_errors=True)
    log(f"written {out_path} ({out_path.stat().st_size / 1e9:.2f} GB)")
    Path(str(out_path) + ".meta.json").write_text(json.dumps({
        "split": args.split, "topk": args.topk, "min_cos": args.min_cos,
        "pairs": total, "queries": int(len(q_ids)),
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
