"""The REAL key-based recall ceiling: shared tokens that are actually usable as keys.

`exp_addr_ceiling.py` reported that 91.4% of missed truth pairs share an identical
street token, implying recall 0.989. That was misleading: it counted ANY shared
street token, including ones like "rue", "main" or "street" that appear in thousands
of records. A block that large is always over the per-pass cap and is always dropped,
so those pairs remain unreachable by key-based blocking no matter how the cap is set.

This re-measures, for each missed pair, whether the two sides share a token that is
BOTH shared AND rare enough to survive the per-pass cap.

Usage:
    python tools/exp_usable_key_ceiling.py --n-s1 40000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.blocking import load_token_idf
from ber.config import Config
from ber.db import connect

BASE = {1: 5000, 3: 300, 4: 3000, 5: 1000, 6: 50, 7: 800, 8: 3000, 9: 200, 10: 60,
        11: 2000, 12: 3000, 13: 2000, 14: 500}


def log(m):
    print(f"[usable] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=40_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="code/business_entity_resolution/config.json")
    ap.add_argument("--out", default="DATA/reports/usable_key_ceiling.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = Path(cfg.data_dir)
    processed = data / "processed"
    key_dir = data / "keys"
    idf_map, _ = load_token_idf(data / "reports" / "train_token_idf.json")
    p12 = BASE[12]
    s1p = (processed / "train_source1.parquet").as_posix()
    cp = (f"['{(processed / 'train_source2.parquet').as_posix()}',"
          f"'{(processed / 'train_source3.parquet').as_posix()}']")

    # Use the candidates already built by the gate sweep's configuration, if present.
    # Otherwise build a probe set directly from the cached keys for a small sample.
    rng = np.random.default_rng(args.seed)
    gt = pd.read_parquet(processed / "train_ground_truth.parquet")
    matched = set(gt.loc[gt.matched_entity_ids != "", "source1_entity_id"])
    chosen = set(rng.choice(sorted(matched), size=min(args.n_s1, len(matched)), replace=False))

    con = connect(cfg)
    con.execute("SET memory_limit='6GB'")
    con.execute("SET threads=3")

    con.execute(
        "CREATE TEMP TABLE tmiss AS "
        "SELECT source1_entity_id AS s1_id, "
        "unnest(string_split(matched_entity_ids, ',')) AS cand_id "
        f"FROM read_parquet('{(processed / 'train_ground_truth.parquet').as_posix()}') "
        "WHERE matched_entity_ids <> ''")
    total = con.execute("SELECT COUNT(*) FROM tmiss").fetchone()[0]

    # Block sizes of street tokens on the candidate side, to decide usability.
    con.execute(
        f"CREATE TEMP TABLE cblocks AS SELECT c.stok AS tok, COUNT(*) AS n FROM "
        f"(SELECT entity_id, unnest(street_tokens) AS stok FROM read_parquet({cp})) c "
        f"WHERE c.stok <> '' GROUP BY c.stok"
    )
    log(f"truth pairs (full train): {total:,}")

    # Shared street tokens, split by whether the block is small enough to survive.
    con.execute(
        f"""
        CREATE TEMP TABLE shared AS
        SELECT DISTINCT a.s1_id, b.cand_id, a.tok
        FROM (SELECT t.s1_id AS s1_id, unnest(s.street_tokens) AS tok
              FROM tmiss t JOIN read_parquet('{s1p}') s ON s.entity_id = t.s1_id
              WHERE s.street_tokens IS NOT NULL) a
        JOIN (SELECT t.cand_id AS cand_id, unnest(c.street_tokens) AS tok
              FROM tmiss t JOIN read_parquet({cp}) c ON c.entity_id = t.cand_id
              WHERE c.street_tokens IS NOT NULL) b
          ON b.tok = a.tok
        """
    )
    shared_all = con.execute("SELECT COUNT(*) FROM (SELECT DISTINCT s1_id, cand_id FROM shared)").fetchone()[0]
    con.execute("CREATE TEMP TABLE usable AS "
                "SELECT DISTINCT s.s1_id, s.cand_id FROM shared s "
                f"JOIN cblocks b ON b.tok = s.tok WHERE b.n <= {int(p12)}")
    usable = con.execute("SELECT COUNT(*) FROM usable").fetchone()[0]

    # Name tokens, same treatment.
    con.execute(
        f"CREATE TEMP TABLE nblocks AS SELECT n.tok AS tok, COUNT(*) AS n FROM "
        f"(SELECT entity_id, unnest(name_idf_tokens) AS tok FROM read_parquet({cp})) n "
        f"WHERE n.tok <> '' GROUP BY n.tok"
    )
    con.execute(
        f"""
        CREATE TEMP TABLE nshared AS
        SELECT DISTINCT a.s1_id, b.cand_id
        FROM (SELECT t.s1_id AS s1_id, unnest(s.name_idf_tokens) AS tok
              FROM tmiss t JOIN read_parquet('{s1p}') s ON s.entity_id = t.s1_id) a
        JOIN (SELECT t.cand_id AS cand_id, unnest(c.name_idf_tokens) AS tok
              FROM tmiss t JOIN read_parquet({cp}) c ON c.entity_id = t.cand_id) b
          ON b.tok = a.tok
        """
    )
    nshared = con.execute("SELECT COUNT(*) FROM nshared").fetchone()[0]
    con.execute("CREATE TEMP TABLE nusable AS "
                "SELECT DISTINCT s.s1_id, s.cand_id FROM nshared s "
                f"JOIN nblocks b ON b.tok = s.tok WHERE b.n <= {int(BASE[3])}")
    nusable = con.execute("SELECT COUNT(*) FROM nusable").fetchone()[0]

    con.execute("CREATE TEMP TABLE either AS SELECT s1_id, cand_id FROM usable "
                "UNION SELECT s1_id, cand_id FROM nusable")
    either = con.execute("SELECT COUNT(*) FROM either").fetchone()[0]
    con.close()

    report = {
        "pass12_cap": p12,
        "pass3_cap": BASE[3],
        "truth_pairs": total,
        "share_any_street_token": shared_all,
        "share_usable_street_token": usable,
        "share_any_name_token": nshared,
        "share_usable_name_token": nusable,
        "share_either_usable": either,
        "usable_key_ceiling_recall": either / max(total, 1),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n=== usable-key ceiling (full train) ===")
    print(f"truth pairs                     : {total:,}")
    print(f"share ANY street token          : {shared_all:,}  ({shared_all / total * 100:.1f}%)")
    print(f"share USABLE street token       : {usable:,}  ({usable / total * 100:.1f}%)  <- blocks <= {p12}")
    print(f"share ANY name token            : {nshared:,}  ({nshared / total * 100:.1f}%)")
    print(f"share USABLE name token         : {nusable:,}  ({nusable / total * 100:.1f}%)  <- blocks <= {BASE[3]}")
    print(f"share EITHER usable             : {either:,}  ({either / total * 100:.1f}%)")
    print(f"\nMAX RECALL reachable by key-based blocking: "
          f"{either / max(total, 1):.4f}")
    print(f"written {args.out}")


if __name__ == "__main__":
    main()
