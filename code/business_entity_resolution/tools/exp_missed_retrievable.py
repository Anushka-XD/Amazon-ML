"""Would embeddings actually recover the pairs blocking misses?

This is the decisive cheap test, run before committing ~9 CPU-hours to embedding all
12.5M records. It takes the truth pairs that the current candidate set does NOT
contain, embeds both sides, and measures what fraction clear a usable cosine.

If a large share of the misses clear the threshold, dense retrieval closes the recall
gap. If they do not, the whole embedding plan is not worth the compute and the honest
answer is that the recall ceiling is lower than the 0.988 target needs.

Usage:
    python tools/exp_missed_retrievable.py --n 6000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "code/business_entity_resolution/src")

MODEL = "intfloat/multilingual-e5-small"


def log(m):
    print(f"[miss] {m}", flush=True)


def build_text(name: str, addr: str) -> str:
    name = (name or "").strip()
    addr = (addr or "").strip()
    if name and addr:
        return f"passage: {name}; {addr}"
    return f"passage: {name or addr}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--max-seq", type=int, default=64)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--candidates", default="DATA/candidates/train_candidates.parquet")
    args = ap.parse_args()

    data = Path("DATA/processed")
    s1p = (data / "train_source1.parquet").as_posix()
    cp = (f"['{(data / 'train_source2.parquet').as_posix()}',"
          f"'{(data / 'train_source3.parquet').as_posix()}']")

    import duckdb
    from ber.config import Config
    from ber.db import connect

    cfg = Config.load("code/business_entity_resolution/config.json")
    con = connect(cfg)
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=3")

    cand_path = Path(args.candidates)
    if cand_path.exists():
        con.execute(f"CREATE TEMP TABLE found AS SELECT DISTINCT s1_id, cand_id "
                    f"FROM read_parquet('{cand_path.as_posix()}')")
        extra = f"AND (t.s1_id, t.cand_id) NOT IN (SELECT s1_id, cand_id FROM found)"
        log(f"using existing candidate set {cand_path.name}")
    else:
        extra = ""
        log("no candidate set found; treating all truth pairs as candidates")
    con.execute(
        "CREATE TEMP TABLE tmiss AS SELECT source1_entity_id AS s1_id, "
        "unnest(string_split(matched_entity_ids, ',')) AS cand_id "
        f"FROM read_parquet('{(data / 'train_ground_truth.parquet').as_posix()}') "
        f"WHERE matched_entity_ids <> '' {extra}"
    )
    total_pairs = con.execute("SELECT COUNT(*) FROM tmiss").fetchone()[0]
    rng = np.random.default_rng(42)
    con.execute("CREATE TEMP TABLE tsample AS SELECT * FROM tmiss USING SAMPLE "
                f"{args.n} ROWS (reservoir, 42)")
    rows = con.execute(
        f"SELECT t.s1_id, t.cand_id, s.name_norm, s.addr_norm, "
        f"       c.name_norm, c.addr_norm "
        f"FROM tsample t "
        f"JOIN read_parquet('{s1p}') s ON s.entity_id = t.s1_id "
        f"JOIN read_parquet({cp}) c ON c.entity_id = t.cand_id"
    ).fetchall()
    con.close()
    log(f"missed truth pairs (total): {total_pairs:,}; sampled {len(rows):,}")

    torch.set_num_threads(args.threads)
    model = SentenceTransformer(MODEL, device="cpu")
    model.eval()
    model.max_seq_length = args.max_seq

    texts = [build_text(r[2], r[3]) for r in rows] + [build_text(r[4], r[5]) for r in rows]
    order = np.argsort([len(t) for t in texts], kind="stable")
    t0 = time.time()
    with torch.inference_mode():
        emb = model.encode([texts[i] for i in order], batch_size=args.batch,
                           normalize_embeddings=True, show_progress_bar=False,
                           convert_to_numpy=True)
    inv = np.empty_like(order)
    inv[order] = np.arange(len(order))
    emb = emb[inv]
    n = len(rows)
    sims = (emb[:n] * emb[n:]).sum(axis=1)
    log(f"embedded in {time.time() - t0:.0f}s")

    base_recall = 0.8714
    print("\n=== of the pairs blocking MISSES, cosine similarity ===")
    for thr in (0.70, 0.75, 0.80, 0.85, 0.90):
        frac = float((sims >= thr).mean())
        new_recall = base_recall + (1 - base_recall) * frac
        log(f"cos>={thr:.2f}: {frac * 100:5.1f}% recoverable  ->  "
            f"recall could reach {new_recall:.4f}")
    log(f"mean cosine on missed pairs: {sims.mean():.3f}  "
        f"p50={np.percentile(sims, 50):.3f} p90={np.percentile(sims, 90):.3f}")

    print("\n--- lowest-scoring missed pairs (where embeddings also fail) ---")
    for i in np.argsort(sims)[:6]:
        log(f"  cos={sims[i]:.3f}  S1={rows[i][2][:38]!r}  CAND={rows[i][4][:38]!r}")
    print("\n--- highest-scoring missed pairs (what we gain) ---")
    for i in np.argsort(-sims)[:6]:
        log(f"  cos={sims[i]:.3f}  S1={rows[i][2][:38]!r}  CAND={rows[i][4][:38]!r}")


if __name__ == "__main__":
    main()
