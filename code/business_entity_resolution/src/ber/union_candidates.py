"""Union the key-based and dense-retrieval candidate sets.

The two sources are complementary and their provenance is preserved rather than
collapsed, because agreement between them is itself a feature: a pair proposed by an
exact-name key *and* by embedding retrieval is far more trustworthy than one proposed
by a single fuzzy pass, and macro F_0.5 weights precision twice as heavily as recall.

  * key-only       -> pass_id > 0, emb_cos = -1
  * retrieval-only -> pass_id = 0,  emb_cos >= 0
  * both           -> pass_id > 0, emb_cos >= 0   (strongest evidence class)
"""
from pathlib import Path

from ber.db import connect

SCHEMA = (
    "s1_id VARCHAR, cand_id VARCHAR, pass_id TINYINT, block_score FLOAT, "
    "n_passes SMALLINT, emb_cos FLOAT, is_s2 BOOLEAN"
)


def _key_sql(path: Path) -> str:
    cols = _columns(path)
    n_passes = "n_passes" if "n_passes" in cols else "CAST(NULL AS SMALLINT)"
    is_s2 = "is_s2" if "is_s2" in cols else "(cand_id LIKE 'S2-%')"
    return (f"SELECT s1_id, cand_id, pass_id, block_score, "
            f"{n_passes} AS n_passes, CAST(-1.0 AS FLOAT) AS emb_cos, "
            f"{is_s2} AS is_s2 FROM read_parquet('{path.as_posix()}')")


def _columns(path: Path) -> set:
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema_arrow.names)


def union_candidates(cfg, key_path, ann_path, out_path):
    key_path, ann_path, out_path = Path(key_path), Path(ann_path), Path(out_path)
    if not ann_path.exists():
        raise FileNotFoundError(f"missing retrieval candidates: {ann_path}")
    if not key_path.exists():
        raise FileNotFoundError(f"missing key candidates: {key_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    k, a = _key_sql(key_path), _ann_sql(ann_path)
    sql = f"""
    WITH k AS ({k}), a AS ({a}),
    j AS (
        SELECT COALESCE(k.s1_id, a.s1_id) AS s1_id,
               COALESCE(k.cand_id, a.cand_id) AS cand_id,
               COALESCE(k.pass_id, 0) AS pass_id,
               COALESCE(k.block_score, 0.0) AS block_score,
               COALESCE(k.n_passes, 0) AS n_passes,
               COALESCE(a.emb_cos, -1.0) AS emb_cos,
               COALESCE(k.is_s2, a.is_s2) AS is_s2
        FROM k FULL OUTER JOIN a
          ON k.s1_id = a.s1_id AND k.cand_id = a.cand_id
    )
    SELECT s1_id, cand_id, pass_id, block_score, n_passes, emb_cos, is_s2 FROM j
    """
    con = connect(cfg)
    con.execute(f"COPY ({sql}) TO '{out_path.as_posix()}' (FORMAT PARQUET)")
    counts = con.execute(
        f"SELECT COUNT(*), "
        f"SUM(CASE WHEN pass_id > 0 AND emb_cos >= 0 THEN 1 ELSE 0 END), "
        f"SUM(CASE WHEN pass_id > 0 AND emb_cos < 0 THEN 1 ELSE 0 END), "
        f"SUM(CASE WHEN pass_id = 0 THEN 1 ELSE 0 END) "
        f"FROM read_parquet('{out_path.as_posix()}')"
    ).fetchone()
    con.close()
    total, both, key_only, ann_only = (int(c or 0) for c in counts)
    return {
        "total_pairs": total,
        "both_sources": both,
        "key_only": key_only,
        "retrieval_only": ann_only,
        "out": str(out_path),
    }


def _ann_sql(path: Path) -> str:
    return (f"SELECT s1_id, cand_id, emb_cos, cand_id LIKE 'S2-%' AS is_s2 "
            f"FROM read_parquet('{path.as_posix()}')")
