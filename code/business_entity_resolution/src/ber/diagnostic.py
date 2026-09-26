"""Classify WHY truth pairs are missing from the candidate set.

Recall-by-pass alone does not say what to fix: a pass that "recalls 0.30" may be
limited by caps, by script mismatch, or by genuinely absent signal. This module
labels every missed truth pair with a cause so the next blocking pass is chosen
from measurement rather than intuition.
"""

import json
import time
from pathlib import Path

from ber.db import connect

# Ordered most-specific first; the first matching label wins.
CAUSES = [
    "roman_name_bridge_would_fire",
    "no_shared_name_token",
    "no_shared_name_prefix5",
    "addr_signal_present_but_capped",
    "both_addr_missing",
    "name_script_mismatch",
]


def _progress(split):
    start = last = time.perf_counter()

    def log(message):
        nonlocal last
        now = time.perf_counter()
        print(f"[diag:{split}] {message} (+{now - last:.1f}s, total {now - start:.1f}s)", flush=True)
        last = now

    return log


def _sql(path) -> str:
    return Path(path).as_posix()


def classify_missing(cfg, split="train") -> dict:
    """Label each truth pair as found or missing-with-cause.

    Works off the processed metadata rather than the candidate parquet, so it can
    run before or without a full block/audit cycle.
    """
    log = _progress(split)
    data = Path(cfg.data_dir)
    processed = data / "processed"
    gt_path = processed / "train_ground_truth.parquet"
    cand_path = data / "candidates" / f"{split}_candidates.parquet"

    con = connect(cfg)
    log("loading candidate pair keys")
    con.execute(
        "CREATE TEMP TABLE found AS SELECT DISTINCT s1_id, cand_id "
        f"FROM read_parquet('{_sql(cand_path)}')"
    )
    n_found = con.execute("SELECT COUNT(*) FROM found").fetchone()[0]
    log(f"candidate pairs: {n_found:,}")

    cols = (
        "entity_id, name_norm, name_roman, name_idf_tokens, name_script, "
        "addr_norm, addr_raw_missing, house_no, street_tokens, postal, state_key"
    )
    con.execute(
        "CREATE TEMP TABLE meta AS SELECT entity_id, name_norm, name_roman, name_idf_tokens, "
        "name_script, addr_norm, addr_raw_missing, house_no, street_tokens, postal, state_key, "
        f"'s1' AS side FROM read_parquet('{_sql(processed / f'{split}_source1.parquet')}')"
        f" WHERE entity_id IN (SELECT s1_id FROM read_parquet('{_sql(gt_path)}'))"
    )
    for source in (2, 3):
        con.execute(
            "INSERT INTO meta SELECT entity_id, name_norm, name_roman, name_idf_tokens, "
            "name_script, addr_norm, addr_raw_missing, house_no, street_tokens, postal, state_key, "
            f"'s{source}' AS side FROM read_parquet('{_sql(processed / f'{split}_source{source}.parquet')}') "
            f"WHERE entity_id IN (SELECT DISTINCT unnest(string_split(matched_entity_ids, ',')) "
            f"FROM read_parquet('{_sql(gt_path)}') WHERE matched_entity_ids <> '')"
        )
    n_meta = con.execute("SELECT COUNT(*) FROM meta").fetchone()[0]
    log(f"metadata rows for S1 + truth candidates: {n_meta:,}")

    con.execute(
        "CREATE TEMP TABLE truth AS SELECT t.s1_id, t.cand_id, "
        "  a.name_norm AS s1_name, a.name_roman AS s1_roman, a.name_idf_tokens AS s1_tokens, "
        "  a.name_script AS s1_script, a.addr_raw_missing AS s1_noaddr, "
        "  a.addr_norm AS s1_addr, a.house_no AS s1_house, a.street_tokens AS s1_street, "
        "  a.postal AS s1_postal, a.state_key AS s1_state, "
        "  b.name_norm AS c_name, b.name_roman AS c_roman, b.name_idf_tokens AS c_tokens, "
        "  b.name_script AS c_script, b.addr_raw_missing AS c_noaddr, "
        "  b.addr_norm AS c_addr, b.house_no AS c_house, b.street_tokens AS c_street, "
        "  b.postal AS c_postal, b.state_key AS c_state, "
        "  CASE WHEN f.s1_id IS NULL THEN 0 ELSE 1 END AS is_found "
        "FROM (SELECT source1_entity_id AS s1_id, "
        "        unnest(string_split(matched_entity_ids, ',')) AS cand_id "
        f"      FROM read_parquet('{_sql(gt_path)}') WHERE matched_entity_ids <> '' "
        "      ) t "
        "JOIN meta a ON a.entity_id = t.s1_id "
        "JOIN meta b ON b.entity_id = t.cand_id "
        "LEFT JOIN found f ON f.s1_id = t.s1_id AND f.cand_id = t.cand_id"
    )
    total = con.execute("SELECT COUNT(*) FROM truth").fetchone()[0]
    log(f"truth pairs joined to metadata: {total:,}")

    # Shared-token helper: a 5-char name-token prefix shared by both sides.
    con.execute(
        "CREATE TEMP TABLE truth2 AS SELECT *, "
        "  len(list_intersect(s1_tokens, c_tokens)) AS n_shared_tok, "
        "  len(list_intersect(list_transform(s1_tokens, x -> left(x,5)), "
        "                       list_transform(c_tokens, x -> left(x,5)))) AS n_shared_pre, "
        "  len(list_intersect(list_transform(s1_roman, x -> left(x,5)), "
        "                       list_transform(c_roman, x -> left(x,5)))) AS n_roman_pre, "
        "  list_has_any(list_transform(s1_roman, x -> left(x,5)), "
        "               list_transform(c_roman, x -> left(x,5))) AS roman_pre_hit "
        "FROM truth"
    )

    rows = con.execute(
        f"""
        SELECT
          CASE
            WHEN n_shared_tok > 0 THEN 'no_shared_name_token'
            WHEN n_shared_pre > 0 THEN 'no_shared_name_prefix5'
            WHEN roman_pre_hit AND (c_addr <> '' OR c_street <> [] OR c_postal <> '') THEN
                 'addr_signal_present_but_capped'
            WHEN s1_noaddr AND c_noaddr THEN 'both_addr_missing'
            WHEN s1_script <> c_script THEN 'name_script_mismatch'
            ELSE 'other'
          END AS cause,
          is_found, s1_script, c_script
        FROM truth2
        """
    ).fetchall()
    log(f"classified {len(rows):,} rows")

    report = _summarize(rows, con, truth2)
    reports = data / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / f"{split}_blocking_diagnostic.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    con.close()
    return report


def _summarize(rows, con, truth2) -> dict:
    by_cause = {}
    by_cause_missing = {}
    script_pairs_missing = {}
    total = len(rows)
    for cause, is_found, s1_script, c_script in rows:
        by_cause[cause] = by_cause.get(cause, 0) + 1
        if not is_found:
            by_cause_missing[cause] = by_cause_missing.get(cause, 0) + 1
            key = f"{s1_script}->{c_script}"
            script_pairs_missing[key] = script_pairs_missing.get(key, 0) + 1
    found = sum(v for k, v in by_cause.items() if k) and total - sum(by_cause_missing.values())
    return {
        "truth_pairs": total,
        "found_pairs": found,
        "recall": (found / total) if total else 0.0,
        "missing_pairs": sum(by_cause_missing.values()),
        "by_cause_total": by_cause,
        "by_cause_missing": by_cause_missing,
        "missing_share_by_cause": {
            k: (v / max(1, sum(by_cause_missing.values()))) for k, v in by_cause_missing.items()
        },
        "script_pairs_missing": dict(
            sorted(script_pairs_missing.items(), key=lambda kv: -kv[1])[:15]
        ),
        "recoverable_if_roman_bridge": (
            _recoverable(con, truth2) if truth2 is not None else None
        ),
    }


def _recoverable(con, truth2) -> dict:
    """Upper bound on what a romanized-name blocking pass could add."""
    total = con.execute("SELECT COUNT(*) FROM truth2").fetchone()[0]
    missing = con.execute("SELECT COUNT(*) FROM truth2 WHERE is_found = 0").fetchone()[0]
    roman = con.execute(
        "SELECT COUNT(*) FROM truth2 WHERE is_found = 0 AND roman_pre_hit = true"
    ).fetchone()[0]
    return {
        "missing": missing,
        "roman_prefix5_shared": roman,
        "share_of_missing": (roman / missing) if missing else 0.0,
        "recall_if_all_recovered": ((total - missing + roman) / total) if total else 0.0,
    }


def oracle_macro(cfg, split="train") -> float:
    """Macro F0.5 of a perfect matcher restricted to the current candidate set.

    This is the hard ceiling: no classifier can beat it, because a truth pair that
    blocking never proposed can never be predicted.
    """
    data = Path(cfg.data_dir)
    con = connect(cfg)
    gt_path = data / "processed" / "train_ground_truth.parquet"
    cand_path = data / "candidates" / f"{split}_candidates.parquet"
    con.execute(
        "CREATE TEMP TABLE truth AS SELECT source1_entity_id AS s1_id, "
        "  unnest(string_split(matched_entity_ids, ',')) AS cand_id "
        f"FROM read_parquet('{_sql(gt_path)}') WHERE matched_entity_ids <> ''"
    )
    con.execute(
        "CREATE TEMP TABLE found AS SELECT DISTINCT s1_id, cand_id "
        f"FROM read_parquet('{_sql(cand_path)}')"
    )
    con.execute(
        "CREATE TEMP TABLE per AS SELECT t.s1_id, COUNT(*) AS n_truth, "
        "  SUM(CASE WHEN f.s1_id IS NOT NULL THEN 1 ELSE 0 END) AS n_found "
        "FROM truth t LEFT JOIN found f ON f.s1_id = t.s1_id AND f.cand_id = t.cand_id "
        "GROUP BY t.s1_id"
    )
    # Singletons score 1.0 when predicted empty; a perfect matcher predicts empty.
    singles = con.execute(
        "SELECT COUNT(*) FROM read_parquet("
        f"'{_sql(gt_path)}') WHERE matched_entity_ids = ''"
    ).fetchone()[0]
    # Per-entity F0.5 with perfect precision reduces to 1.25R/(0.25+R).
    macro = con.execute(
        "SELECT AVG(1.25 * (n_found::DOUBLE / n_truth) / (0.25 + n_found::DOUBLE / n_truth)) "
        "FROM per"
    ).fetchone()[0]
    n_entities = con.execute("SELECT COUNT(*) FROM per").fetchone()[0]
    con.close()
    overall = (macro * n_entities + singles) / (n_entities + singles)
    return {"oracle_macro_matched": macro, "oracle_macro_with_singletons": overall,
            "matched_entities": n_entities, "singletons": singles}
