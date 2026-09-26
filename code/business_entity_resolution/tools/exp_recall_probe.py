"""Sampled blocking-recall probe with per-cause attribution.

A full `ber.cli block --split train` at cap=400 emits ~500M candidate pairs, which is
expensive in time and disk. The Phase A decision only needs *why* truth pairs are
missed, and that is estimable from a sample of Source 1 entities: the candidate keys
for all 10.3M S2/S3 records are the only expensive artifact, and they are reusable
across probes and the eventual full block.

Usage:
    python tools/exp_recall_probe.py --n-s1 100000 [--cap 400] [--pass-caps-json ...]
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
from ber.address import house_number, street_key_tokens  # noqa: F401
from ber.blocking import _join_sql, _pass_caps, block_keys, load_token_idf
from ber.config import Config
from ber.db import connect


def log(msg):
    print(f"[probe] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-s1", type=int, default=100_000)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", default="code/business_entity_resolution/config.json")
    ap.add_argument("--out", default="DATA/reports/recall_probe.json")
    args = ap.parse_args()

    cfg = Config.load(args.config)
    cfg = type(cfg)(**{**cfg.__dict__, "cap": args.cap})
    data = Path(cfg.data_dir)
    processed = data / "processed"
    key_dir = data / "keys"
    key_dir.mkdir(parents=True, exist_ok=True)

    idf_path = data / "reports" / "train_token_idf.json"
    if idf_path.exists():
        idf_map, _ = load_token_idf(idf_path)
    else:
        from ber.blocking import compute_token_idf

        log("computing token IDF (first run only)...")
        t0 = time.time()
        idf_map = compute_token_idf(
            [processed / f"train_source{s}.parquet" for s in (1, 2, 3)],
            cfg.idf_min, idf_path, cfg=cfg,
        )
        log(f"IDF over {len(idf_map):,} tokens in {time.time() - t0:.0f}s")
    min_idf = cfg.idf_min

    # --- candidate keys (expensive, reusable) -------------------------------------
    cand_parts = []
    for source in (2, 3):
        part = key_dir / f"train_source{source}_keys.parquet"
        if not part.exists():
            t0 = time.time()
            _write_keys(processed / f"train_source{source}.parquet", part, idf_map, min_idf)
            log(f"built {part.name} in {time.time() - t0:.0f}s "
                f"({part.stat().st_size / 1e9:.2f} GB)")
        else:
            log(f"reusing {part.name}")
        cand_parts.append(part)

    # --- sample S1 ---------------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    s1_all = pq.ParquetFile(processed / "train_source1.parquet")
    n_s1_total = s1_all.metadata.num_rows
    log(f"train S1 rows: {n_s1_total:,}")

    gt = pd.read_parquet(processed / "train_ground_truth.parquet")
    keep_ids = set(gt.loc[gt.matched_entity_ids != "", "source1_entity_id"])
    log(f"S1 entities with >=1 true match: {len(keep_ids):,}")

    # Read only the S1 rows we need, plus a random slice of singletons for the
    # false-merge side of the ledger.
    chosen = set(rng.choice(sorted(keep_ids), size=min(args.n_s1, len(keep_ids)), replace=False))
    singles = gt.loc[gt.matched_entity_ids == "", "source1_entity_id"].to_numpy()
    n_single = min(args.n_s1 // 5, len(singles))
    chosen |= set(rng.choice(singles, size=n_single, replace=False).tolist())
    log(f"sampled S1 entities: {len(chosen):,} ({n_single:,} singletons)")

    s1_frames = []
    for batch in s1_all.iter_batches(batch_size=250_000):
        df = batch.to_pandas()
        hit = df[df.entity_id.isin(chosen)]
        if len(hit):
            s1_frames.append(hit)
        if sum(len(f) for f in s1_frames) >= len(chosen):
            break
    s1 = pd.concat(s1_frames, ignore_index=True)
    log(f"loaded {len(s1):,} sampled S1 rows")

    # --- block -------------------------------------------------------------------
    con = connect(cfg)
    t0 = time.time()
    con.register("s1k", block_keys(s1, idf_map, min_idf))
    cand_sql = "[" + ",".join(f"'{p.as_posix()}'" for p in cand_parts) + "]"
    con.execute(f"CREATE TEMP VIEW candk AS SELECT * FROM read_parquet({cand_sql})")
    sql = _join_sql("s1k", "candk", cfg.cap, _pass_caps(cfg))
    con.execute(f"CREATE TEMP TABLE cand AS {sql}")
    n_cand = con.execute("SELECT COUNT(*) FROM cand").fetchone()[0]
    log(f"candidates: {n_cand:,} ({n_cand / max(len(s1), 1):.1f}/S1) in {time.time() - t0:.0f}s")

    # --- recall + attribution ----------------------------------------------------
    con.execute("CREATE TEMP TABLE truth AS SELECT source1_entity_id AS s1_id, "
                "unnest(string_split(matched_entity_ids, ',')) AS cand_id "
                "FROM read_parquet("
                f"'{_sqlp(processed / 'train_ground_truth.parquet')}') "
                "WHERE matched_entity_ids <> ''")
    con.execute("CREATE TEMP TABLE t AS SELECT * FROM truth "
                "WHERE s1_id IN (SELECT entity_id FROM s1)")
    total = con.execute("SELECT COUNT(*) FROM t").fetchone()[0]
    con.execute("CREATE TEMP TABLE found_pairs AS SELECT DISTINCT s1_id, cand_id FROM cand")
    found = con.execute("SELECT COUNT(*) FROM t WHERE (s1_id, cand_id) IN "
                        "(SELECT s1_id, cand_id FROM found_pairs)").fetchone()[0]
    log(f"truth pairs in sample: {total:,}  found: {found:,}  recall={found / max(total, 1):.4f}")

    # Explode both sides' tokens so shared-token / shared-prefix tests are plain
    # equi-joins. Lambdas over list columns are not bindable in this DuckDB build.
    s1p = _sqlp(processed / "train_source1.parquet")
    cp = (f"['{(processed / 'train_source2.parquet').as_posix()}',"
          f"'{(processed / 'train_source3.parquet').as_posix()}']")
    con.execute(f"CREATE TEMP TABLE tmiss AS SELECT * FROM t WHERE NOT EXISTS "
                f"(SELECT 1 FROM found_pairs f WHERE f.s1_id = t.s1_id "
                f"AND f.cand_id = t.cand_id)")
    con.execute(
        f"""
        CREATE TEMP TABLE stok AS
        SELECT s1_id, unnest(name_idf_tokens) AS tok FROM (
            SELECT s.entity_id AS s1_id, s.name_idf_tokens
            FROM read_parquet('{s1p}') s SEMI JOIN (SELECT DISTINCT s1_id FROM tmiss) x
              ON x.s1_id = s.entity_id
        )
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE ctok AS
        SELECT cand_id, unnest(name_idf_tokens) AS tok FROM (
            SELECT c.entity_id AS cand_id, c.name_idf_tokens
            FROM read_parquet({cp}) c SEMI JOIN (SELECT DISTINCT cand_id FROM tmiss) x
              ON x.cand_id = c.entity_id
        )
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE shared_tok AS
        SELECT DISTINCT a.s1_id, b.cand_id
        FROM stok a JOIN ctok b ON a.tok = b.tok
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE shared_pre AS
        SELECT DISTINCT a.s1_id, b.cand_id
        FROM (SELECT s1_id, left(tok, 5) AS p FROM stok) a
        JOIN (SELECT cand_id, left(tok, 5) AS p FROM ctok) b ON a.p = b.p
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE miss AS
        SELECT
          CASE WHEN st.s1_id IS NOT NULL THEN 1 ELSE 0 END AS has_shared_tok,
          CASE WHEN sp.s1_id IS NOT NULL THEN 1 ELSE 0 END AS has_shared_pre,
          c.house_no, c.postal, length(coalesce(c.street_tokens, [])) AS n_c_street,
          c.addr_norm, s.addr_raw_missing AS s1_noaddr, c.addr_raw_missing AS c_noaddr,
          s.name_script AS s1_script, c.name_script AS c_script
        FROM tmiss t
        JOIN read_parquet('{s1p}') s ON s.entity_id = t.s1_id
        JOIN read_parquet({cp}) c ON c.entity_id = t.cand_id
        LEFT JOIN shared_tok st ON st.s1_id = t.s1_id AND st.cand_id = t.cand_id
        LEFT JOIN shared_pre sp ON sp.s1_id = t.s1_id AND sp.cand_id = t.cand_id
        """
    )
    causes = con.execute(
        """
        SELECT
          CASE
            WHEN has_shared_tok = 1 THEN '1_blocked_but_dropped_by_cap'
            WHEN has_shared_pre = 1 THEN '2_prefix_only_cap'
            WHEN (house_no <> '' OR n_c_street > 0 OR postal <> '')
                 THEN '3_addr_exists_but_no_name_overlap'
            WHEN s1_noaddr AND c_noaddr THEN '4_both_addr_missing'
            WHEN s1_script <> c_script THEN '5_script_mismatch'
            ELSE '6_other'
          END AS cause, COUNT(*) AS n
        FROM miss GROUP BY cause ORDER BY cause
        """
    ).fetchall()
    con.close()

    missing = total - found
    by_cause = {c: int(n) for c, n in causes}
    report = {
        "n_s1_sampled": len(s1),
        "cap": args.cap,
        "candidates": int(n_cand),
        "candidates_per_s1": n_cand / max(len(s1), 1),
        "truth_pairs": int(total),
        "found_pairs": int(found),
        "recall": found / max(total, 1),
        "missing_pairs": int(missing),
        "missing_by_cause": by_cause,
        "missing_share_by_cause": {k: v / max(missing, 1) for k, v in by_cause.items()},
        "recall_if_caps_and_addr_pass_fixed":
            (found + by_cause.get("1_blocked_but_dropped_by_cap", 0)
             + by_cause.get("2_prefix_only_cap", 0)
             + by_cause.get("3_addr_exists_but_no_name_overlap", 0)) / max(total, 1),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n=== recall probe ===")
    print(f"recall                 : {report['recall']:.4f}")
    print(f"candidates / S1        : {report['candidates_per_s1']:.1f}")
    print(f"missing pairs          : {missing:,}")
    for c, n in by_cause.items():
        print(f"  {c:32s} {n:>10,}  {n / max(missing, 1) * 100:5.1f}%")
    print(f"\nrecall if caps + address passes were exhaustive: "
          f"{report['recall_if_caps_and_addr_pass_fixed']:.4f}")
    print(f"written: {out}")


def _sqlp(p):
    return Path(p).as_posix()


def _write_keys(parquet_path, key_path, idf_map, min_idf):
    import pyarrow as pa

    writer = None
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=1_000_000):
        keys = block_keys(batch.to_pandas(), idf_map, min_idf)
        table = pa.Table.from_pandas(keys, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(key_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
