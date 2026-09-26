"""Can a quality-aware per-S1 ranking recover the cap-limited recall?

The cap sweep showed recall saturating at 0.884: past cap=400 the per-S1 limit stops
binding and the loss moves to `_valid_sql`, which drops any block larger than its
per-pass cap. Widening those blocks alone *lowers* recall, because the per-S1
selection is ordered by `pass_id`, so a flood of low-value address collisions from
passes 4/5 evicts true matches that would have arrived via passes 7-10.

This compares rankings at high cap with wide blocks:
  * `pass_id`  - the shipped behaviour
  * `score`    - order by block_score DESC, then pass_id
  * `evidence` - count of distinct passes proposing the pair, then block_score

The last one is the reference repo's retrieval-agreement idea applied to the
selection step rather than the model.
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
from ber.blocking import _pass_caps, block_keys, load_token_idf
from ber.config import Config
from ber.db import connect

BASE_CAPS = {1: 5000, 3: 300, 4: 3000, 5: 1000, 6: 50, 7: 600, 8: 300, 9: 200, 10: 60}
# Modest widening only. The per-S1 row_number window has to materialise the whole join
# before the cap applies, so join size scales superlinearly with block caps: 10x widths
# exhaust 20 GiB of temp at cap=1600. 3x is enough to admit the blocks that were
# dropping truth pairs while keeping the join tractable on an 8 GB host.
WIDE = {1: 20000, 3: 1200, 4: 8000, 5: 3000, 6: 200, 7: 2500, 8: 1200,
        9: 700, 10: 250}


def log(m):
    print(f"[rank] {m}", flush=True)


def join_sql(rank_expr, cap):
    cases = " ".join(f"WHEN {int(p)} THEN {int(c)}"
                     for p, c in sorted({**BASE_CAPS, **WIDE}.items()))
    return f"""
    WITH s AS (SELECT entity_id, pass_id, key, block_score FROM s1k WHERE key <> ''),
         c AS (SELECT entity_id, pass_id, key, block_score FROM candk WHERE key <> ''),
         valid AS (
             SELECT pass_id, key FROM c GROUP BY pass_id, key
             HAVING COUNT(*) <= CASE pass_id {cases} ELSE 100000 END
         ),
         j AS (
             SELECT s.entity_id AS s1_id, c.entity_id AS cand_id,
                    s.pass_id AS pass_id, s.block_score AS block_score
             FROM s JOIN c ON s.key = c.key AND s.pass_id = c.pass_id
             JOIN valid v ON v.pass_id = s.pass_id AND v.key = c.key
         ),
         agg AS (
             SELECT s1_id, cand_id, MIN(pass_id) AS pass_id,
                    MAX(block_score) AS block_score,
                    COUNT(DISTINCT pass_id) AS n_passes
             FROM j GROUP BY s1_id, cand_id
         ),
         ranked AS (
             SELECT *, ROW_NUMBER() OVER (
                 PARTITION BY s1_id ORDER BY {rank_expr}
             ) AS rn
             FROM agg
         )
    SELECT s1_id, cand_id, pass_id, block_score, n_passes FROM ranked WHERE rn <= {int(cap)}
    """


RANKINGS = {
    "pass_id": "pass_id ASC, block_score DESC, cand_id ASC",
    "score": "block_score DESC, pass_id ASC, cand_id ASC",
    "evidence": "n_passes DESC, block_score DESC, cand_id ASC",
    "evidence_score": "n_passes DESC, block_score DESC, pass_id ASC, cand_id ASC",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--caps", default="1600,3200,6400")
    ap.add_argument("--config", default="code/business_entity_resolution/config.json")
    ap.add_argument("--out", default="DATA/reports/rank_sweep.json")
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
    s1k = block_keys(s1, idf_map, cfg.idf_min)
    log(f"{len(s1):,} S1 rows, {len(s1k):,} keys")

    con = connect(cfg)
    con.execute("PRAGMA max_temp_directory_size='20GiB'")
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=2")
    con.register("s1k", s1k)
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
    log(f"truth pairs for sample: {total:,}")

    results = []
    print(f"\n{'ranking':16s} {'cap':>6s} {'cands/S1':>9s} {'recall':>8s} {'sec':>6s}")
    for name, expr in RANKINGS.items():
        for cap in [int(c) for c in args.caps.split(",")]:
            t0 = time.time()
            con.execute("DROP TABLE IF EXISTS cand")
            con.execute(f"CREATE TEMP TABLE cand AS {join_sql(expr, cap)}")
            n = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
            found = con.execute(
                "SELECT COUNT(*) FROM truth_ids WHERE (s1_id, cand_id) IN "
                "(SELECT s1_id, cand_id FROM (SELECT DISTINCT s1_id, cand_id FROM cand))"
            ).fetchone()[0]
            recall = found / max(total, 1)
            dt = time.time() - t0
            results.append({"ranking": name, "cap": cap, "candidates": n,
                            "candidates_per_s1": n / max(len(s1), 1),
                            "found": found, "recall": recall, "seconds": dt})
            print(f"{name:16s} {cap:6d} {n / max(len(s1), 1):9.1f} {recall:8.4f} {dt:6.0f}",
                  flush=True)
            if recall >= 0.99:
                log(f"*** 0.99 recall: ranking={name} cap={cap} ***")
    con.close()
    out = Path(args.out)
    out.write_text(json.dumps({"truth_pairs": total, "results": results}, indent=2),
                   encoding="utf-8")
    log(f"written {out}")


if __name__ == "__main__":
    main()
