"""Street-token rarity gate: coverage vs block size.

The address-key ceiling said an identical street token recovers 91.4% of missed truth
pairs, but gating street tokens on high IDF (to keep blocks small) dropped recall to
0.8832 - because the *common* street tokens are the ones that recover most misses,
and they are exactly the ones the gate removed.

This sweeps the gate to find where coverage and block size balance, and reports the
projected full-train pair count so a configuration that cannot be built at full scale
is visibly rejected rather than discovered later.

Usage:
    python tools/exp_street_gate.py --n-s1 40000
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
from ber.blocking import _join_sql, _prepare_cand_keys, block_keys, load_token_idf
from ber.config import Config
from ber.db import connect

BASE = {1: 5000, 3: 300, 4: 3000, 5: 1000, 6: 50, 7: 800, 8: 3000, 9: 200, 10: 60,
        11: 2000, 12: 3000, 13: 2000, 14: 500}
MAX_PAIRS = 450_000_000  # ~11 GB of parquet; the disk budget on this host


def log(m):
    print(f"[gate] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--config", default="code/business_entity_resolution/config.json")
    ap.add_argument("--out", default="DATA/reports/street_gate.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = Path(cfg.data_dir)
    processed = data / "processed"
    key_dir = data / "keys"
    idf_map, _ = load_token_idf(data / "reports" / "train_token_idf.json")
    cand_sql = ("read_parquet("
                f"['{(key_dir / 'train_source2_keys.parquet').as_posix()}',"
                f"'{(key_dir / 'train_source3_keys.parquet').as_posix()}'])")

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

    # (label, per-S1 cap, street IDF gate, pass-12 cap)
    configs = [
        ("gate4.0 p12=3000 cap400", 400, 4.0, 3000),
        ("gate2.0 p12=3000 cap400", 400, 2.0, 3000),
        ("gate0   p12=3000 cap400", 400, 0.0, 3000),
        ("gate0   p12=8000 cap400", 400, 0.0, 8000),
        ("gate0   p12=20000 cap400", 400, 0.0, 20000),
        ("gate0   p12=8000 cap250", 250, 0.0, 8000),
    ]

    results = []
    print(f"\n{'config':26s} {'cands/S1':>9s} {'recall':>8s} {'proj pairs':>12s} {'fits':>5s} {'sec':>6s}")
    for label, cap, s_gate, p12 in configs:
        t0 = time.time()
        caps = {**BASE, 12: p12, 8: p12}
        s1k = block_keys(s1, idf_map, cfg.idf_min, street_min_idf=s_gate)
        con = connect(cfg)
        con.execute("SET memory_limit='6GB'")
        con.execute("SET threads=2")
        con.register("s1k", s1k)
        try:
            _prepare_cand_keys(con, cand_sql, "s1k", caps)
            con.execute("DROP TABLE IF EXISTS cand_all")
            con.execute("CREATE TEMP TABLE cand_all AS SELECT s1_id, cand_id FROM ("
                        + " UNION ALL ".join(
                            f"SELECT s1_id, cand_id FROM ({_join_sql('s1k', cap, shard=(sh, args.shards))})"
                            for sh in range(args.shards)) + ")")
            n_c = con.execute("SELECT COUNT(*) FROM cand_all").fetchone()[0]
            con.execute(
                "CREATE TEMP TABLE truth_ids AS SELECT source1_entity_id AS s1_id, "
                "unnest(string_split(matched_entity_ids, ',')) AS cand_id "
                f"FROM read_parquet('{(processed / 'train_ground_truth.parquet').as_posix()}') "
                "WHERE matched_entity_ids <> ''"
            )
            total = con.execute(
                "SELECT COUNT(*) FROM truth_ids WHERE s1_id IN (SELECT entity_id FROM s1)"
            ).fetchone()[0]
            found = con.execute(
                "SELECT COUNT(*) FROM truth_ids t SEMI JOIN "
                "(SELECT DISTINCT s1_id, cand_id FROM cand_all) c "
                "ON c.s1_id = t.s1_id AND c.cand_id = t.cand_id"
            ).fetchone()[0]
        except Exception as exc:
            con.close()
            print(f"{label:26s} {'-':>9s} {'-':>8s} {'-':>12s} {'OOM':>5s} "
                  f"{time.time() - t0:6.0f}  ({type(exc).__name__})", flush=True)
            results.append({"label": label, "cap": cap, "street_min_idf": s_gate,
                            "p12": p12, "error": type(exc).__name__})
            continue
        con.close()
        per_s1 = n_c / max(len(s1), 1)
        projected = per_s1 * 2_206_821
        recall = found / max(total, 1)
        fits = projected <= MAX_PAIRS
        results.append({"label": label, "cap": cap, "street_min_idf": s_gate, "p12": p12,
                        "candidates": n_c, "candidates_per_s1": per_s1,
                        "projected_full_train_pairs": projected,
                        "found": found, "truth_pairs": total, "recall": recall,
                        "fits_disk_budget": fits})
        print(f"{label:26s} {per_s1:9.1f} {recall:8.4f} {projected / 1e6:11.0f}M "
              f"{str(fits):>5s} {time.time() - t0:6.0f}", flush=True)
        if recall >= 0.975:
            log(f"*** 0.975 REACHED: {label} ***")

    Path(args.out).write_text(json.dumps({"n_s1": len(s1), "results": results}, indent=2),
                              encoding="utf-8")
    log(f"written {args.out}")


if __name__ == "__main__":
    main()
