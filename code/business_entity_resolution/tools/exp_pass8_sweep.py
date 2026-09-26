"""Confirm the configuration that actually reaches 0.99 recall.

The address-key ceiling measurement showed an identical street token recovers 91.4% of
missed truth pairs (recall 0.8714 -> 0.9890) and an identical address token recovers
94.4% (-> 0.9929). Street tokens are already emitted by pass 8, so the loss is
discarded candidates, not missing signal.

But widening block caps previously *lowered* recall, because the per-S1 selection is
ordered by `pass_id`: a flood of low-value address collisions from passes 4/5 evicts
true matches arriving via passes 7-10. So this tests pass-8 width together with a
quality-aware selection order (`n_passes DESC`) instead of pass id.

Usage:
    python tools/exp_pass8_sweep.py --n-s1 60000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.blocking import block_keys, load_token_idf
from ber.config import Config
from ber.db import connect

BASE = {1: 5000, 3: 300, 4: 3000, 5: 1000, 6: 50, 7: 600, 8: 300, 9: 200, 10: 60}


def log(m):
    print(f"[p8] {m}", flush=True)


def build_candidates(con, caps, cap, rank_expr):
    """Materialise each stage.

    Written as separate statements rather than one CTE chain: DuckDB was re-evaluating
    the `valid` aggregate and the join together and exhausting 5 GB as soon as pass 8
    widened, because the per-S1 row_number window needs the whole join resident anyway.
    """
    cases = " ".join(f"WHEN {int(p)} THEN {int(c)}" for p, c in sorted(caps.items()))
    con.execute("DROP TABLE IF EXISTS valid")
    con.execute("DROP TABLE IF EXISTS j")
    con.execute("DROP TABLE IF EXISTS agg")
    con.execute("DROP TABLE IF EXISTS cand")
    con.execute(
        "CREATE TEMP TABLE valid AS SELECT pass_id, key FROM candk WHERE key <> '' "
        f"GROUP BY pass_id, key HAVING COUNT(*) <= CASE pass_id {cases} ELSE 100000 END"
    )
    con.execute(
        "CREATE TEMP TABLE j AS "
        "SELECT s.entity_id AS s1_id, c.entity_id AS cand_id, "
        "       s.pass_id AS pass_id, s.block_score AS block_score "
        "FROM (SELECT * FROM s1k WHERE key <> '') s "
        "JOIN (SELECT * FROM candk WHERE key <> '') c "
        "  ON s.key = c.key AND s.pass_id = c.pass_id "
        "JOIN valid v ON v.pass_id = s.pass_id AND v.key = c.key"
    )
    con.execute(
        "CREATE TEMP TABLE agg AS SELECT s1_id, cand_id, MIN(pass_id) AS pass_id, "
        "MAX(block_score) AS block_score, COUNT(DISTINCT pass_id) AS n_passes "
        "FROM j GROUP BY s1_id, cand_id"
    )
    con.execute(
        f"CREATE TEMP TABLE cand AS SELECT s1_id, cand_id, pass_id, block_score, n_passes "
        f"FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY {rank_expr}) AS rn "
        f"FROM agg) WHERE rn <= {int(cap)}"
    )
    con.execute("DROP TABLE IF EXISTS j")
    con.execute("DROP TABLE IF EXISTS agg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="code/business_entity_resolution/config.json")
    ap.add_argument("--out", default="DATA/reports/pass8_sweep.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = Path(cfg.data_dir)
    processed = data / "processed"
    key_dir = data / "keys"
    idf_map, _ = load_token_idf(data / "reports" / "train_token_idf.json")
    cand_sql = "[" + ",".join(
        f"'{(key_dir / f'train_source{s}_keys.parquet').as_posix()}'" for s in (2, 3)) + "]"

    rng = np.random.default_rng(args.seed)
    gt = pd.read_parquet(processed / "train_ground_truth.parquet")
    matched = set(gt.loc[gt.matched_entity_ids != "", "source1_entity_id"])
    chosen = set(rng.choice(sorted(matched), size=min(args.n_s1, len(matched)), replace=False))
    frames = []
    for batch in pq.ParquetFile(processed / "train_source1.parquet").iter_batches(
            batch_size=250_000):
        df = batch.to_pandas()
        hit = df[df.entity_id.isin(chosen)]
        if len(hit):
            frames.append(hit)
        if sum(len(f) for f in frames) >= len(chosen):
            break
    s1 = pd.concat(frames, ignore_index=True)

    con = connect(cfg)
    con.execute("PRAGMA max_temp_directory_size='20GiB'")
    con.execute("SET memory_limit='7GB'")
    con.execute("SET threads=2")
    con.register("s1k", block_keys(s1, idf_map, cfg.idf_min))
    con.execute(f"CREATE TEMP VIEW candk AS SELECT * FROM read_parquet({cand_sql})")
    con.execute(
        "CREATE TEMP TABLE truth_ids AS SELECT source1_entity_id AS s1_id, "
        "unnest(string_split(matched_entity_ids, ',')) AS cand_id "
        f"FROM read_parquet('{(processed / 'train_ground_truth.parquet').as_posix()}') "
        "WHERE matched_entity_ids <> ''"
    )
    total = con.execute(
        "SELECT COUNT(*) FROM truth_ids WHERE s1_id IN (SELECT entity_id FROM s1)"
    ).fetchone()[0]
    log(f"{len(s1):,} S1 rows, {total:,} truth pairs")

    RANK = {
        "pass_id": "pass_id ASC, block_score DESC, cand_id ASC",
        "evidence": "n_passes DESC, block_score DESC, cand_id ASC",
    }
    configs = [
        ("tight cap=400",             400, dict(BASE), "pass_id"),
        ("p8=2k cap=400 ev",          400, {**BASE, 8: 2000}, "evidence"),
        ("p8=5k cap=400 ev",          400, {**BASE, 8: 5000}, "evidence"),
        ("p8=5k cap=1200 ev",        1200, {**BASE, 8: 5000}, "evidence"),
        ("p8=20k cap=1200 ev",       1200, {**BASE, 8: 20000}, "evidence"),
        ("p8=20k p4=10k cap=1200 ev",1200, {**BASE, 8: 20000, 4: 10000}, "evidence"),
    ]

    results = []
    print(f"\n{'config':30s} {'cands/S1':>9s} {'recall':>8s} {'sec':>6s}")
    for label, cap, caps, rank in configs:
        t0 = time.time()
        build_candidates(con, caps, cap, RANK[rank])
        n = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
        found = con.execute(
            "SELECT COUNT(*) FROM truth_ids WHERE (s1_id, cand_id) IN "
            "(SELECT s1_id, cand_id FROM (SELECT DISTINCT s1_id, cand_id FROM cand))"
        ).fetchone()[0]
        recall = found / max(total, 1)
        dt = time.time() - t0
        results.append({"label": label, "cap": cap, "pass_caps": caps, "ranking": rank,
                        "candidates": n, "candidates_per_s1": n / max(len(s1), 1),
                        "found": found, "recall": recall, "seconds": dt})
        print(f"{label:30s} {n / max(len(s1), 1):9.1f} {recall:8.4f} {dt:6.0f}", flush=True)
        if recall >= 0.99:
            log(f"*** 0.99 REACHED: {label} ***")
    con.close()
    Path(args.out).write_text(json.dumps({"truth_pairs": total, "results": results}, indent=2),
                              encoding="utf-8")
    log(f"written {args.out}")


if __name__ == "__main__":
    main()
