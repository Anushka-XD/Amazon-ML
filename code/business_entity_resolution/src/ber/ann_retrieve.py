"""Dense retrieval over the multilingual embeddings.

Recall stage only: it proposes candidates, and the matcher decides. Kept separate from
`blocking.py` because the evidence it produces is independent — a pair found by an exact
name key *and* by embedding retrieval is far more trustworthy than one found by either
alone, and that agreement is carried through as a feature.

Measured justification: of the truth pairs key-based blocking misses, 100% clear cosine
0.85 and 96.3% clear 0.90 (mean 0.965), against 0.822 mean / 0.850 p90 for unrelated
pairs.
"""
from pathlib import Path

import numpy as np

from ber.db import connect

DIM = 384
SHARD_ROWS = 250_000


def load_shards(emb_dir: Path, prefix: str):
    """Load vectors and ids for entities whose id starts with `prefix`."""
    ids_path = emb_dir / "ids.txt"
    all_ids = [ln.strip() for ln in ids_path.read_text(encoding="utf-8").splitlines()
               if ln.strip()]
    shards = sorted(emb_dir.glob("shard_*.npy"))
    if not shards:
        raise FileNotFoundError(f"no embedding shards under {emb_dir}")
    keep = np.array([i.startswith(prefix) for i in all_ids])
    n_keep = int(keep.sum())
    vecs = np.empty((n_keep, DIM), dtype=np.float16)
    ids = np.empty(n_keep, dtype=object)
    pos = 0
    for k, shard in enumerate(shards):
        arr = np.load(shard)
        lo = k * SHARD_ROWS
        idx = np.flatnonzero(keep[lo:lo + arr.shape[0]])
        if len(idx):
            vecs[pos:pos + len(idx)] = arr[idx]
            ids[pos:pos + len(idx)] = [all_ids[lo + i] for i in idx]
            pos += len(idx)
        del arr
    return vecs[:pos], ids[:pos]


def retrieve_ann(cfg, split="train", topk=50, min_cos=0.80, emb_dir=None,
                 shard=200_000, out_path=None):
    import faiss
    import pyarrow as pa
    import pyarrow.parquet as pq

    emb = Path(emb_dir or (Path(cfg.data_dir) / "emb" / split))
    out = Path(out_path or (Path(cfg.data_dir) / "candidates"
                            / f"{split}_ann_candidates.parquet"))
    out.parent.mkdir(parents=True, exist_ok=True)
    part_dir = out.parent / f"{split}_ann_parts"
    if part_dir.exists():
        import shutil
        shutil.rmtree(part_dir)
    part_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ann] loading candidate vectors from {emb}", flush=True)
    c2, id2 = load_shards(emb, "S2-")
    c3, id3 = load_shards(emb, "S3-")
    cand_v = np.concatenate([c2, c3])
    cand_ids = np.concatenate([id2, id3])
    del c2, c3
    print(f"[ann] candidate vectors: {cand_v.shape}", flush=True)

    index = faiss.IndexFlatIP(cand_v.shape[1])
    index.add(cand_v.astype("float32"))
    del cand_v
    print(f"[ann] index built over {index.ntotal:,}", flush=True)

    q_v, q_ids = load_shards(emb, "S1-")
    print(f"[ann] queries: {q_v.shape}", flush=True)

    total, part = 0, 0
    for start in range(0, len(q_ids), shard):
        chunk = q_v[start:start + shard].astype("float32")
        sims, idxs = index.search(chunk, topk)
        s1_col, cand_col, cos_col = [], [], []
        for r in range(len(chunk)):
            sid = q_ids[start + r]
            for c, j in enumerate(idxs[r]):
                if j < 0:
                    break
                cos = float(sims[r][c])
                if cos < min_cos:
                    break  # results are sorted descending
                s1_col.append(sid)
                cand_col.append(cand_ids[j])
                cos_col.append(cos)
        pq.write_table(pa.table({
            "s1_id": pa.array(s1_col, pa.string()),
            "cand_id": pa.array(cand_col, pa.string()),
            "emb_cos": pa.array(cos_col, pa.float32()),
        }), part_dir / f"part_{part:05d}.parquet")
        total += len(s1_col)
        part += 1
        if part % 10 == 0:
            print(f"[ann]   {start + len(chunk):,}/{len(q_ids):,} queries, "
                  f"{total:,} pairs", flush=True)

    writer = None
    for p in sorted(part_dir.glob("part_*.parquet")):
        for b in pq.ParquetFile(p).iter_batches(batch_size=500_000):
            t = pa.Table.from_batches([b])
            if writer is None:
                writer = pq.ParquetWriter(out, t.schema)
            writer.write_table(t)
    if writer is not None:
        writer.close()
    import shutil
    shutil.rmtree(part_dir, ignore_errors=True)
    return {"split": split, "ann_pairs": total, "queries": int(len(q_ids)),
            "topk": topk, "min_cos": min_cos, "path": str(out)}


def audit_recall(cfg, split="train", candidates=None):
    """Pair recall for any candidate set, so key-only and union can be compared."""
    data = Path(cfg.data_dir)
    cand = Path(candidates or (data / "candidates" / f"{split}_union_candidates.parquet"))
    if not cand.exists():
        cand = data / "candidates" / f"{split}_candidates.parquet"
    gt = data / "processed" / "train_ground_truth.parquet"
    con = connect(cfg)
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=3")
    con.execute(
        "CREATE TEMP TABLE truth AS SELECT source1_entity_id AS s1_id, "
        "unnest(string_split(matched_entity_ids, ',')) AS cand_id "
        f"FROM read_parquet('{gt.as_posix()}') WHERE matched_entity_ids <> ''"
    )
    con.execute(f"CREATE TEMP TABLE found AS SELECT DISTINCT s1_id, cand_id "
                f"FROM read_parquet('{cand.as_posix()}')")
    total = con.execute("SELECT COUNT(*) FROM truth").fetchone()[0]
    hit = con.execute(
        "SELECT COUNT(*) FROM truth t SEMI JOIN found f "
        "ON f.s1_id = t.s1_id AND f.cand_id = t.cand_id"
    ).fetchone()[0]
    n_cand = con.execute("SELECT COUNT(*) FROM found").fetchone()[0]
    con.close()
    recall = hit / max(total, 1)
    return {"candidates": str(cand), "candidate_pairs": int(n_cand),
            "truth_pairs": int(total), "found_pairs": int(hit), "recall": recall,
            "oracle_macro_f05_upper_bound": _oracle_bound(recall)}


def _oracle_bound(pair_recall: float) -> float:
    """Upper bound on macro F0.5 implied by a pair-recall figure, at perfect precision.

    Uses f(R) = 1.25R/(0.25+R), which is the per-entity F0.5 when precision is 1. It
    assumes every entity sees the same recall, which is optimistic because misses are
    entity-correlated, so treat it as a ceiling rather than a forecast.
    """
    r = min(max(pair_recall, 0.0), 1.0)
    return 1.25 * r / (0.25 + r)
