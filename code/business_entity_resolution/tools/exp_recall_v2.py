"""Measure recall of the NEW address-pass blocking on a sample, cheaply.

A full-train block at cap=400 with the address passes produces ~793M pairs, needs
~20 GB and ~6 h on this host - unaffordable for tuning. The sample harness answers the
same question in about a minute per configuration, because the keys are cached and
only the Source 1 side is sharded down.

Reports recall against the ceiling target and the candidate cost, so a configuration
can be chosen that is both high-recall and small enough to actually build at full
scale.

Usage:
    python tools/exp_recall_v2.py --n-s1 60000 --configs quick
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


def log(m):
    print(f"[v2] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--config", default="code/business_entity_resolution/config.json")
    ap.add_argument("--out", default="DATA/reports/recall_v2.json")
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
    s1k = block_keys(s1, idf_map, cfg.idf_min)
    log(f"{len(s1):,} S1 rows, {len(s1k):,} keys")

    # Configurations: (label, per-S1 cap, pass caps)
    configs = [
        ("cap400 newpasses",        400, dict(BASE)),
        ("cap400 no-addr-passes",   400, {k: v for k, v in BASE.items() if k not in (11, 12, 13)}),
        ("cap150 newpasses",        150, dict(BASE)),
        ("cap250 newpasses",        250, dict(BASE)),
        ("cap400 wider-addr",       400, {**BASE, 11: 4000, 12: 6000, 13: 4000}),
        ("cap600 wider-addr",       600, {**BASE, 11: 4000, 12: 6000, 13: 4000}),
        ("cap400 tighter-addr",     400, {**BASE, 11: 1000, 12: 1500, 13: 1000}),
    ]

    results = []
    print(f"\n{'config':26s} {'cands/S1':>9s} {'recall':>8s} {'full-train est':>15s} {'sec':>6s}")
    for label, cap, caps in configs:
        t0 = time.time()
        con = connect(cfg)
        con.execute("SET memory_limit='6GB'")
        con.execute("SET threads=2")
        con.register("s1k", s1k)
        _prepare_cand_keys(con, cand_sql, "s1k", caps)
        # Materialise the candidate set once, then measure from it.
        con.execute("DROP TABLE IF EXISTS cand_all")
        con.execute("CREATE TEMP TABLE cand_all AS SELECT s1_id, cand_id FROM ("
                    + " UNION ALL ".join(
                        f"SELECT s1_id, cand_id FROM ({_join_sql('s1k', cap, shard=(sh, args.shards))})"
                        for sh in range(args.shards)) + ")")
        n_c = con.execute("SELECT COUNT(*) FROM cand_all").fetchone()[0]
        per_s1 = n_c / max(len(s1), 1)
        projected = per_s1 * 2_206_821
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
        recall = found / max(total, 1)
        con.close()
        dt = time.time() - t0
        results.append({"label": label, "cap": cap, "pass_caps": caps,
                        "candidates": n_c, "candidates_per_s1": per_s1,
                        "projected_full_train_pairs": projected,
                        "found": found, "truth_pairs": total, "recall": recall,
                        "seconds": dt})
        print(f"{label:26s} {per_s1:9.1f} {recall:8.4f} {projected / 1e6:14.0f}M {dt:6.0f}",
              flush=True)
        if recall >= 0.975:
            log(f"*** 0.975 REACHED: {label} ***")

    Path(args.out).write_text(json.dumps({"n_s1": len(s1), "results": results}, indent=2),
                              encoding="utf-8")
    log(f"written {args.out}")


if __name__ == "__main__":
    main()
