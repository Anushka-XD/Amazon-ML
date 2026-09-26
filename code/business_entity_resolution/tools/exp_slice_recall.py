"""Per-slice candidate recall: by country, script, and address availability.

Leave-one-country-out implied an unseen country costs 0.09-0.12 macro F0.5, but that
figure conflated a genuine generalisation gap with the candidate-recall ceiling on that
country. France is ~15% of the test split with no labels, so its recall is a
first-class measurement here rather than an extrapolation.

Usage:
    python tools/exp_slice_recall.py --split train [--cap 400]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.config import Config
from ber.db import connect


def log(m):
    print(f"[slice] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--config", default="code/business_entity_resolution/config.json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = Path(cfg.data_dir)
    processed = data / "processed"
    out_path = Path(args.out or f"DATA/reports/{args.split}_slice_recall.json")

    con = connect(cfg)
    con.execute("PRAGMA max_temp_directory_size='18GiB'")
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=3")

    cand = (data / "candidates" / f"{args.split}_candidates.parquet").as_posix()
    s1p = (processed / f"{args.split}_source1.parquet").as_posix()

    # Country lives only in the raw TSV, not in the processed parquet.
    country_files = [p for p in
                     (Path(cfg.dataset_dir) / args.split).glob(f"{args.split}_source*.tsv")]
    files_sql = "[" + ",".join(f"'{p.as_posix()}'" for p in country_files) + "]"
    con.execute(
        "CREATE TEMP TABLE country AS SELECT entity_id, lower(trim(country)) AS country "
        f"FROM read_csv({files_sql}, delim='\t', header=true, union_by_name=true)"
    )
    con.execute(
        "CREATE TEMP TABLE s1meta AS SELECT p.entity_id AS s1_id, p.name_script, "
        "p.addr_raw_missing, coalesce(c.country,'') AS country "
        f"FROM read_parquet('{s1p}') p LEFT JOIN country c ON c.entity_id = p.entity_id"
    )
    con.execute(f"CREATE TEMP TABLE found AS SELECT DISTINCT s1_id, cand_id FROM read_parquet('{cand}')")
    log(f"candidate pairs: {con.execute('SELECT COUNT(*) FROM found').fetchone()[0]:,}")

    if args.split == "train":
        con.execute(
            "CREATE TEMP TABLE truth AS SELECT source1_entity_id AS s1_id, "
            "unnest(string_split(matched_entity_ids, ',')) AS cand_id "
            f"FROM read_parquet('{(processed / 'train_ground_truth.parquet').as_posix()}') "
            "WHERE matched_entity_ids <> ''"
        )
        extra = ""
    else:
        # No labels on test: measure how many candidates each entity receives instead,
        # which is the observable proxy for whether blocking is working there.
        con.execute(
            "CREATE TEMP TABLE truth AS SELECT s1_id, NULL::VARCHAR AS cand_id FROM s1meta"
        )
        extra = ""

    if args.split == "train":
        rows = con.execute(
            """
            SELECT m.country, m.name_script, m.addr_raw_missing,
                   COUNT(*) AS truth_pairs,
                   SUM(CASE WHEN f.cand_id IS NOT NULL THEN 1 ELSE 0 END) AS found
            FROM truth t
            JOIN s1meta m ON m.s1_id = t.s1_id
            LEFT JOIN found f ON f.s1_id = t.s1_id AND f.cand_id = t.cand_id
            GROUP BY 1, 2, 3 ORDER BY truth_pairs DESC
            """
        ).fetchall()
        by_country = con.execute(
            """
            SELECT m.country, COUNT(*) AS truth_pairs,
                   SUM(CASE WHEN f.cand_id IS NOT NULL THEN 1 ELSE 0 END) AS found
            FROM truth t JOIN s1meta m ON m.s1_id = t.s1_id
            LEFT JOIN found f ON f.s1_id = t.s1_id AND f.cand_id = t.cand_id
            GROUP BY 1 ORDER BY truth_pairs DESC
            """
        ).fetchall()
        by_script = con.execute(
            """
            SELECT m.name_script, COUNT(*) AS truth_pairs,
                   SUM(CASE WHEN f.cand_id IS NOT NULL THEN 1 ELSE 0 END) AS found
            FROM truth t JOIN s1meta m ON m.s1_id = t.s1_id
            LEFT JOIN found f ON f.s1_id = t.s1_id AND f.cand_id = t.cand_id
            GROUP BY 1 ORDER BY truth_pairs DESC
            """
        ).fetchall()
        cand_per_entity = con.execute(
            """
            SELECT m.country, COUNT(*) AS entities,
                   SUM(CASE WHEN f.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS with_cands
            FROM s1meta m LEFT JOIN (SELECT DISTINCT s1_id FROM found) f
              ON f.s1_id = m.s1_id GROUP BY 1 ORDER BY entities DESC
            """
        ).fetchall()
    else:
        rows, by_country, by_script = [], [], []
        cand_per_entity = con.execute(
            """
            SELECT m.country, COUNT(*) AS entities,
                   SUM(CASE WHEN f.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS with_cands
            FROM s1meta m LEFT JOIN (SELECT DISTINCT s1_id FROM found) f
              ON f.s1_id = m.s1_id GROUP BY 1 ORDER BY entities DESC
            """
        ).fetchall()
    con.close()

    def rec(found, total):
        return (found / total) if total else None

    report = {
        "split": args.split,
        "cap": args.cap,
        "recall_by_country": {c: {"truth_pairs": int(t), "found": int(f),
                                  "recall": rec(f, t)} for c, t, f in by_country},
        "recall_by_script": {str(s): {"truth_pairs": int(t), "found": int(f),
                                      "recall": rec(f, t)} for s, t, f in by_script},
        "candidates_by_country": {c: {"entities": int(e), "with_candidates": int(w),
                                      "coverage": rec(w, e)} for c, e, w in cand_per_entity},
    }
    if rows:
        report["recall_by_country_script_addr"] = [
            {"country": c, "script": s, "addr_missing": bool(a),
             "truth_pairs": int(t), "found": int(f), "recall": rec(f, t)}
            for c, s, a, t, f in rows[:40]
        ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== recall by country ({args.split}) ===")
    for c, v in report["recall_by_country"].items():
        print(f"  {c:10s} pairs={v['truth_pairs']:>9,}  recall={v['recall']:.4f}")
    if report["recall_by_script"]:
        print("--- by script ---")
        for s, v in report["recall_by_script"].items():
            print(f"  {str(s):10s} pairs={v['truth_pairs']:>9,}  recall={v['recall']:.4f}")
    print("--- candidate coverage ---")
    for c, v in report["candidates_by_country"].items():
        print(f"  {c:10s} entities={v['entities']:>9,}  with_candidates={v['coverage']:.4f}")
    print(f"\nwritten {out_path}")


if __name__ == "__main__":
    main()
