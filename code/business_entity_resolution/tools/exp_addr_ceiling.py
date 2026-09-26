"""What is the TRUE recall ceiling? Test whether missed pairs share an address key.

The first probe reported "recall if caps + address pass were exhaustive = 1.0000", but
that classified a pair as address-recoverable whenever the address was merely
*non-empty*. That is an upper bound on an untested assumption: a non-empty address on
both sides does not imply the two sides produce a *shared* blocking key.

This measures the real thing. For every truth pair the current candidate set misses, it
asks whether the two sides agree on any key a prospective address pass could use:

    house              equal, non-empty house numbers
    house|street[:5]   the existing pass-4 key
    street_tok         any identical street token
    street_pre5        any identical 5-char street-token prefix (pass 8)
    postal             equal, non-empty postal codes
    state              equal, non-empty state keys
    addr_tok           any identical address token at all (loosest possible)
    addr_tok_pre4      any identical 4-char address-token prefix

The share of missed pairs reachable by each key is the honest per-pass ceiling.

Usage:
    python tools/exp_addr_ceiling.py --n-s1 60000
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.blocking import _join_sql, _pass_caps, block_keys, load_token_idf
from ber.config import Config
from ber.db import connect

BASE_CAPS = {1: 5000, 3: 300, 4: 3000, 5: 1000, 6: 50, 7: 600, 8: 300, 9: 200, 10: 60}


def log(m):
    print(f"[ceiling] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=60_000)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="code/business_entity_resolution/config.json")
    ap.add_argument("--out", default="DATA/reports/addr_ceiling.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    data = Path(cfg.data_dir)
    processed = data / "processed"
    key_dir = data / "keys"
    idf_map, _ = load_token_idf(data / "reports" / "train_token_idf.json")
    cand_sql = "[" + ",".join(
        f"'{(key_dir / f'train_source{s}_keys.parquet').as_posix()}'" for s in (2, 3)) + "]"
    s1p = (processed / "train_source1.parquet").as_posix()
    cp = (f"['{(processed / 'train_source2.parquet').as_posix()}',"
          f"'{(processed / 'train_source3.parquet').as_posix()}']")

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
    log(f"{len(s1):,} S1 rows sampled")

    con = connect(cfg)
    con.execute("PRAGMA max_temp_directory_size='20GiB'")
    con.execute("SET threads=4")
    con.register("s1k", block_keys(s1, idf_map, cfg.idf_min))
    con.execute(f"CREATE TEMP VIEW candk AS SELECT * FROM read_parquet({cand_sql})")
    con.execute(f"CREATE TEMP TABLE cand AS {_join_sql('s1k', 'candk', args.cap, _pass_caps(cfg))}")
    n_cand = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
    log(f"candidates: {n_cand:,} ({n_cand / max(len(s1), 1):.1f}/S1)")

    con.execute(
        "CREATE TEMP TABLE truth_ids AS SELECT source1_entity_id AS s1_id, "
        "unnest(string_split(matched_entity_ids, ',')) AS cand_id "
        f"FROM read_parquet('{(processed / 'train_ground_truth.parquet').as_posix()}') "
        "WHERE matched_entity_ids <> ''"
    )
    con.execute(
        f"CREATE TEMP TABLE tmiss AS SELECT * FROM truth_ids "
        f"WHERE s1_id IN (SELECT entity_id FROM s1) AND (s1_id, cand_id) NOT IN "
        f"(SELECT s1_id, cand_id FROM (SELECT DISTINCT s1_id, cand_id FROM cand))"
    )
    total = con.execute(
        "SELECT COUNT(*) FROM truth_ids WHERE s1_id IN (SELECT entity_id FROM s1)"
    ).fetchone()[0]
    missed = con.execute("SELECT COUNT(*) FROM tmiss").fetchone()[0]
    recall = 1 - missed / max(total, 1)
    log(f"truth pairs {total:,}  missed {missed:,}  recall {recall:.4f}")

    # Exploded address tokens for both sides of the missed pairs only.
    con.execute(
        f"""
        CREATE TEMP TABLE miss_ids AS SELECT DISTINCT s1_id FROM tmiss;
        CREATE TEMP TABLE miss_cand AS SELECT DISTINCT cand_id FROM tmiss;
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE sa AS
        SELECT s.entity_id AS s1_id, s.house_no, s.postal, s.state_key,
               unnest(s.street_tokens) AS stok
        FROM read_parquet('{s1p}') s SEMI JOIN miss_ids m ON m.s1_id = s.entity_id
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE ca AS
        SELECT c.entity_id AS cand_id, c.house_no, c.postal, c.state_key,
               unnest(c.street_tokens) AS ctok
        FROM read_parquet({cp}) c SEMI JOIN miss_cand m ON m.cand_id = c.entity_id
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE tmiss_meta AS
        SELECT t.s1_id, t.cand_id
        FROM tmiss t
        JOIN read_parquet('{s1p}') s ON s.entity_id = t.s1_id
        JOIN read_parquet({cp}) c ON c.entity_id = t.cand_id
        """
    )
    for col, alias in (("house_no", "house"), ("postal", "postal"), ("state_key", "state")):
        con.execute(
            f"""
            CREATE TEMP TABLE hit_{alias} AS
            SELECT DISTINCT m.s1_id, m.cand_id FROM tmiss_meta m
            JOIN (SELECT DISTINCT s1_id, {col} AS v FROM sa WHERE {col} <> '') a
              ON a.s1_id = m.s1_id
            JOIN (SELECT DISTINCT cand_id, {col} AS v FROM ca WHERE {col} <> '') b
              ON b.cand_id = m.cand_id AND b.v = a.v
            """
        )
    con.execute(
        """
        CREATE TEMP TABLE hit_street_tok AS
        SELECT DISTINCT m.s1_id, m.cand_id FROM tmiss_meta m
        JOIN (SELECT DISTINCT s1_id, stok AS v FROM sa WHERE stok <> '') a ON a.s1_id = m.s1_id
        JOIN (SELECT DISTINCT cand_id, ctok AS v FROM ca WHERE ctok <> '') b
          ON b.cand_id = m.cand_id AND b.v = a.v
        """
    )
    con.execute(
        """
        CREATE TEMP TABLE hit_street_pre5 AS
        SELECT DISTINCT m.s1_id, m.cand_id FROM tmiss_meta m
        JOIN (SELECT DISTINCT s1_id, left(stok,5) AS v FROM sa WHERE length(stok) >= 5) a
          ON a.s1_id = m.s1_id
        JOIN (SELECT DISTINCT cand_id, left(ctok,5) AS v FROM ca WHERE length(ctok) >= 5) b
          ON b.cand_id = m.cand_id AND b.v = a.v
        """
    )
    # Loosest possible address evidence: any identical address token (or 4-char prefix).
    con.execute(
        f"""
        CREATE TEMP TABLE sa_all AS
        SELECT s.entity_id AS s1_id, unnest(string_split(s.addr_norm, ' ')) AS atok
        FROM read_parquet('{s1p}') s SEMI JOIN miss_ids m ON m.s1_id = s.entity_id
        WHERE s.addr_norm <> ''
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE ca_all AS
        SELECT c.entity_id AS cand_id, unnest(string_split(c.addr_norm, ' ')) AS atok
        FROM read_parquet({cp}) c SEMI JOIN miss_cand m ON m.cand_id = c.entity_id
        WHERE c.addr_norm <> ''
        """
    )
    for name, expr_a, expr_b in (
        ("addr_tok", "atok", "atok"),
        ("addr_tok_pre4", "left(atok,4)", "left(atok,4)"),
    ):
        con.execute(
            f"""
            CREATE TEMP TABLE hit_{name} AS
            SELECT DISTINCT m.s1_id, m.cand_id FROM tmiss_meta m
            JOIN (SELECT DISTINCT s1_id, {expr_a} AS v FROM sa_all
                  WHERE {expr_a} <> '') a ON a.s1_id = m.s1_id
            JOIN (SELECT DISTINCT cand_id, {expr_b} AS v FROM ca_all
                  WHERE {expr_b} <> '') b ON b.cand_id = m.cand_id AND b.v = a.v
            """
        )

    keys = ["house", "postal", "state", "street_tok", "street_pre5", "addr_tok", "addr_tok_pre4"]
    counts = {k: con.execute(f"SELECT COUNT(*) FROM hit_{k}").fetchone()[0] for k in keys}
    any_key = con.execute(
        "SELECT COUNT(*) FROM tmiss_meta m WHERE EXISTS (SELECT 1 FROM hit_house h "
        "WHERE h.s1_id = m.s1_id AND h.cand_id = m.cand_id) "
        "OR EXISTS (SELECT 1 FROM hit_street_pre5 h WHERE h.s1_id = m.s1_id AND h.cand_id = m.cand_id) "
        "OR EXISTS (SELECT 1 FROM hit_postal h WHERE h.s1_id = m.s1_id AND h.cand_id = m.cand_id) "
        "OR EXISTS (SELECT 1 FROM hit_state h WHERE h.s1_id = m.s1_id AND h.cand_id = m.cand_id) "
        "OR EXISTS (SELECT 1 FROM hit_street_tok h WHERE h.s1_id = m.s1_id AND h.cand_id = m.cand_id)"
    ).fetchone()[0]
    union_loose = con.execute(
        "SELECT COUNT(*) FROM tmiss_meta m WHERE EXISTS (SELECT 1 FROM hit_addr_tok h "
        "WHERE h.s1_id = m.s1_id AND h.cand_id = m.cand_id) "
        "OR EXISTS (SELECT 1 FROM hit_addr_tok_pre4 h WHERE h.s1_id = m.s1_id AND h.cand_id = m.cand_id)"
    ).fetchone()[0]

    con.close()
    report = {
        "cap": args.cap,
        "n_s1": len(s1),
        "candidates_per_s1": n_cand / max(len(s1), 1),
        "truth_pairs": total,
        "missed": missed,
        "recall": recall,
        "recoverable_by_key": {k: {"pairs": int(v), "share_of_missed": v / max(missed, 1),
                                    "resulting_recall": (total - missed + v) / max(total, 1)}
                               for k, v in counts.items()},
        "union_all_address_keys": {
            "pairs": int(any_key),
            "share_of_missed": any_key / max(missed, 1),
            "resulting_recall": (total - missed + any_key) / max(total, 1),
        },
        "union_loosest_addr_token": {
            "pairs": int(union_loose),
            "share_of_missed": union_loose / max(missed, 1),
            "resulting_recall": (total - missed + union_loose) / max(total, 1),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== address-key ceiling (cap={args.cap}) ===")
    print(f"current recall              : {recall:.4f}   ({missed:,} missed of {total:,})")
    print(f"\n{'key':16s} {'recovers':>10s} {'% missed':>9s} {'-> recall':>10s}")
    for k, v in counts.items():
        print(f"{k:16s} {v:10,} {v / max(missed, 1) * 100:8.1f}% "
              f"{(total - missed + v) / max(total, 1):10.4f}")
    print(f"{'UNION(all)':16s} {any_key:10,} {any_key / max(missed, 1) * 100:8.1f}% "
          f"{(total - missed + any_key) / max(total, 1):10.4f}")
    print(f"{'UNION(loose)':16s} {union_loose:10,} {union_loose / max(missed, 1) * 100:8.1f}% "
          f"{(total - missed + union_loose) / max(total, 1):10.4f}")
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
