import json
import shutil
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ber.address import street_key_component
from ber.db import connect

DEFAULT_PASS_CAPS = {1: 5000, 3: 200, 4: 2000, 5: 1000, 6: 50, 7: 500, 8: 300, 9: 100, 10: 30,
                     11: 500, 12: 500, 13: 500, 14: 600}


def compute_token_idf(files, min_idf, out_path, cfg=None, min_df=50):
    """Document frequency over name AND street tokens.

    Street tokens are included because the address passes now gate on rarity: only
    street tokens above `min_idf` are emitted, since a token shared by a large share
    of records produces a block that is always over the per-pass cap and always
    dropped. Without street tokens in this table every street token scores 0 and the
    address passes emit nothing.
    """
    from ber.db import connect

    con = connect(cfg)
    con.execute("SET enable_progress_bar=false")
    files_sql = "[" + ",".join("'" + str(f).replace("\\", "/") + "'" for f in files) + "]"
    total = con.execute(f"SELECT COUNT(*) FROM read_parquet({files_sql})").fetchone()[0]
    # min_df prunes the long tail of hapax street tokens before the Python dict is
    # built; they can never clear min_idf and only bloat the artefact.
    counts = con.execute(
        "SELECT token, COUNT(DISTINCT entity_id) AS df FROM ("
        f"SELECT entity_id, UNNEST(name_idf_tokens) AS token FROM read_parquet({files_sql})"
        " UNION ALL "
        f"SELECT entity_id, UNNEST(street_tokens) AS token FROM read_parquet({files_sql})"
        ") GROUP BY token HAVING df >= " + str(int(min_df))
    ).fetchall()
    idf = {token: float(np.log(total / df)) for token, df in counts if df > 0}
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"n": total, "min_idf": min_idf, "min_df": min_df, "idf": idf},
                   ensure_ascii=False),
        encoding="utf-8",
    )
    con.close()
    return idf


def load_token_idf(path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload["idf"], float(payload["min_idf"])


def normalize_keys(values):
    return pd.Series(values, dtype="string").fillna("").astype(str).tolist()


def block_keys(df, idf_map=None, min_idf=4.0, street_min_idf=None):
    idf_map = idf_map or {}
    # Street tokens gate on their own, lower threshold by default. Measured coverage on
    # missed truth pairs is highest for *common* street tokens, but those produce
    # blocks over the per-pass cap, so the gate trades coverage against block size.
    # `street_min_idf=0` disables it and lets the per-pass cap do the filtering.
    s_min_idf = min_idf if street_min_idf is None else float(street_min_idf)
    eids = df["entity_id"].tolist()
    norm_col = df["name_norm"].tolist()
    fold_col = df["name_fold"].tolist() if "name_fold" in df.columns else norm_col
    idf_col = df["name_idf_tokens"] if "name_idf_tokens" in df.columns else df["name_tokens"]
    idf_col = [list(x) if x is not None else [] for x in idf_col.tolist()]
    house_col = df["house_no"].tolist()
    street_col = [list(x) if x is not None else [] for x in df["street_tokens"].tolist()]
    postal_col = df["postal"].tolist()
    state_col = df["state_key"].tolist()

    frames = [
        pd.DataFrame(
            {
                "entity_id": eids,
                "pass_id": 1,
                "key": normalize_keys(norm_col),
                "block_score": 1.0,
            }
        )
    ]
    pass3 = []
    pass4 = []
    pass5 = []
    pass6 = []
    pass7 = []
    pass8 = []
    pass9 = []
    pass10 = []
    pass11 = []
    pass12 = []
    pass13 = []
    pass14 = []
    for eid, tokens, norm, fold, house, street, postal, state in zip(
        eids, idf_col, norm_col, fold_col, house_col, street_col, postal_col, state_col
    ):
        unique = set(tokens)
        has_rare = False
        for token in unique:
            score = idf_map.get(token, 0.0)
            if score >= min_idf:
                has_rare = True
                pass3.append((eid, 3, token, float(score)))
            if len(token) >= 5:
                pass7.append((eid, 7, token[:5], 0.75))
        rarest = sorted(((idf_map.get(t, 0.0), t) for t in unique), reverse=True)
        top = [t for _, t in rarest[:4]]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                a, b = sorted((top[i], top[j]))
                pass9.append((eid, 9, f"{a}|{b}", 0.65))
        if len(top) >= 3:
            a, b, c = sorted(top[:3])
            pass10.append((eid, 10, f"{a}|{b}|{c}", 0.6))
        # Rarest street token, not the first one: `street[0]` degenerates to particles
        # like "de"/"du"/"des" in French addresses and to similar fillers in Indian
        # ones, which makes the key collide across a huge share of records and then get
        # dropped by the per-pass block cap.
        scomp = street_key_component(street, idf_map, min_idf)

        if house and scomp:
            pass4.append((eid, 4, f"{house}|{scomp}", 0.8))
        for token in set(street):
            if len(token) >= 5 and idf_map.get(token, 0.0) >= s_min_idf:
                pass8.append((eid, 8, token[:5], 0.7))
        first_token = norm.split()[0] if norm.split() else ""
        if postal and first_token:
            pass5.append((eid, 5, f"{postal}|{first_token}", 0.7))
        if not has_rare and len(norm.split()) <= 2 and norm:
            pass6.append((eid, 6, f"{state}|{norm[:3]}", 0.5))

        # --- address passes -----------------------------------------------------
        # Measured on real missed truth pairs: an identical street token recovers
        # 91.4% of them and a shared house number 38.5%, so these carry most of the
        # remaining recall. The existing passes emitted only a street prefix and only
        # when a house number was present, which discarded most of that signal.
        if house:
            pass11.append((eid, 11, f"h{house}", 0.85))
        for token in set(street):
            if len(token) >= 4 and idf_map.get(token, 0.0) >= s_min_idf:
                pass12.append((eid, 12, f"s{token}", 0.78))
        if postal and scomp:
            pass13.append((eid, 13, f"p{postal}|{scomp}", 0.76))

        # --- accent-folded name passes -----------------------------------------
        # `name_norm` is NFKC + casefold and so KEEPS accents, while train Source 1
        # names are 100% ASCII. `Cafe` and `Café` therefore produce different blocking
        # keys, which silently costs recall on the accented slice (France). Folding
        # NFKD-strips diacritics and bridges that.
        if fold and fold != norm:
            for token in set(fold.split()):
                if len(token) >= 5:
                    pass14.append((eid, 14, token[:5], 0.72))

    for rows, pid in (
        (pass3, 3), (pass4, 4), (pass5, 5), (pass6, 6), (pass7, 7), (pass8, 8),
        (pass9, 9), (pass10, 10), (pass11, 11), (pass12, 12), (pass13, 13), (pass14, 14),
    ):
        frames.append(pd.DataFrame(rows, columns=["entity_id", "pass_id", "key", "block_score"]))
    out = pd.concat([f for f in frames if len(f)], ignore_index=True)
    out = out[out["key"].astype(str).str.len() > 0].copy()
    out["key"] = out["key"].astype(str)
    out["block_score"] = out["block_score"].astype("float32")
    out["pass_id"] = out["pass_id"].astype("int8")
    return out


def _pass_caps(cfg):
    caps = dict(DEFAULT_PASS_CAPS)
    for key, value in (getattr(cfg, "pass_caps", None) or {}).items():
        caps[int(key)] = int(value)
    return caps


def _valid_sql(s1_src, cand_src, caps):
    cases = " ".join(f"WHEN {int(p)} THEN {int(c)}" for p, c in sorted(caps.items()))
    return (
        "SELECT pass_id, key FROM {src} GROUP BY pass_id, key "
        "HAVING COUNT(*) <= CASE pass_id {cases} ELSE {fallback} END"
    ).format(src=cand_src, cases=cases, fallback=int(max(caps.values())))


def _prepare_cand_keys(con, cand_sql, s1k_sql, caps):
    """Pre-filter candidate keys to probed blocks, and cache the valid block set.

    Doing this once, globally, is what makes sharding viable. Measured on train: one
    shard of 16 produced a 121M-row join, because the per-S1 `ROW_NUMBER` forces the
    whole join resident and DuckDB materialises it before applying the block cap.
    Restricting the candidate table to keys some Source 1 entity actually probes, and
    caching the surviving blocks, shrinks every subsequent shard to a fraction of the
    original.
    """
    cases = " ".join(f"WHEN {int(p)} THEN {int(c)}" for p, c in sorted(caps.items()))
    fallback = int(max(caps.values()))
    con.execute("DROP TABLE IF EXISTS candk_f")
    con.execute("DROP TABLE IF EXISTS validb")
    con.execute("CREATE TEMP TABLE candk_f AS "
                f"SELECT c.entity_id, c.pass_id, c.key FROM {cand_sql} c "
                "SEMI JOIN (SELECT DISTINCT pass_id, key FROM "
                f"{s1k_sql} WHERE key <> '') sk "
                "ON sk.pass_id = c.pass_id AND sk.key = c.key")
    n_cand = con.execute("SELECT COUNT(*) FROM candk_f").fetchone()[0]
    con.execute("CREATE TEMP TABLE validb AS "
                "SELECT pass_id, key FROM candk_f GROUP BY pass_id, key "
                f"HAVING COUNT(*) <= CASE pass_id {cases} ELSE {fallback} END")
    n_blocks = con.execute("SELECT COUNT(*) FROM validb").fetchone()[0]
    return n_cand, n_blocks


def _join_sql(s1_src, cap, shard=None):
    """Join blocking keys into candidate pairs, then keep the best `cap` per S1.

    Selection is ordered by `block_score` first, then by how many independent passes
    proposed the pair (`n_passes`). Ordering by pass id made recall *fall* when the
    address passes were widened (measured 0.8714 -> 0.7563), because a flood of
    low-value address collisions evicted true matches arriving via later passes.
    `n_passes` is a strong signal but must NOT lead: doing so let a fuzzy multi-pass
    pair outrank an exact-name match and push it out of the per-S1 cap entirely.

    Reads the pre-filtered `candk_f` / `validb` tables from _prepare_cand_keys.
    `shard` restricts to one `hash(entity_id) % n` bucket; the predicate is on the
    Source 1 side, before the join, or the shard accomplishes nothing.
    """
    s_where = "key <> ''" if shard is None else (
        f"key <> '' AND hash(entity_id) % {int(shard[1])} = {int(shard[0])}")
    return f"""
WITH s AS (SELECT entity_id, pass_id, key, block_score FROM {s1_src} WHERE {s_where}),
     j AS (
         SELECT s.entity_id AS s1_id, c.entity_id AS cand_id,
                s.pass_id AS pass_id, s.block_score AS block_score
         FROM s
         JOIN candk_f c ON s.key = c.key AND s.pass_id = c.pass_id
         SEMI JOIN validb v ON v.pass_id = s.pass_id AND v.key = s.key
     ),
     dedup AS (
         SELECT s1_id, cand_id, MIN(pass_id) AS pass_id, MAX(block_score) AS block_score,
                COUNT(DISTINCT pass_id) AS n_passes
         FROM j GROUP BY s1_id, cand_id
     ),
     ranked AS (
         -- block_score leads, so an EXACT name match (score 1.0) is never crowded out
         -- of the per-S1 cap by a fuzzy pair that several passes happened to propose.
         -- Ordering by n_passes first caused exactly that: measured missed pairs
         -- included identical names scoring cosine 1.000, dropped purely because a
         -- 5-pass fuzzy rival outranked them.
         SELECT *, ROW_NUMBER() OVER (
             PARTITION BY s1_id ORDER BY block_score DESC, n_passes DESC, cand_id
         ) AS rn
         FROM dedup
     )
SELECT s1_id, cand_id, pass_id, block_score, CAST(n_passes AS SMALLINT) AS n_passes,
       CASE WHEN cand_id LIKE 'S2-%' THEN TRUE ELSE FALSE END AS is_s2
FROM ranked WHERE rn <= {int(cap)}
"""


def generate_candidates(s1, cands, cfg):
    from ber.db import connect

    con = connect(cfg)
    idf_map = getattr(cfg, "idf_map", None)
    min_idf = getattr(cfg, "idf_min", 4.0)
    con.register("s1k", block_keys(s1, idf_map, min_idf))
    con.register("candk", block_keys(cands, idf_map, min_idf))
    _prepare_cand_keys(con, "candk", "s1k", _pass_caps(cfg))
    sql = _join_sql("s1k", cfg.cap)
    table = con.execute(sql)
    table = table.to_arrow_table() if hasattr(table, "to_arrow_table") else table.fetch_arrow_table()
    out = table.to_pandas()
    con.close()
    return out


def _write_keys_chunked(parquet_path, key_path, idf_map, min_idf):
    Path(key_path).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=1_000_000):
        df = batch.to_pandas()
        keys = block_keys(df, idf_map, min_idf)
        table = __import__("pyarrow").Table.from_pandas(keys, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(key_path, table.schema)
        writer.write_table(table)
    if writer is not None:
        writer.close()


def run_block(cfg, split):
    processed = Path(cfg.data_dir) / "processed"
    reports = Path(cfg.data_dir) / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    idf_path = reports / f"{split}_token_idf.json"
    if idf_path.exists():
        idf_map, min_idf = load_token_idf(idf_path)
        min_idf = cfg.idf_min
    else:
        idf_map = compute_token_idf(
            [
                processed / f"{split}_source1.parquet",
                processed / f"{split}_source2.parquet",
                processed / f"{split}_source3.parquet",
            ],
            cfg.idf_min,
            idf_path,
            cfg=cfg,
        )
        min_idf = cfg.idf_min
    key_dir = Path(cfg.data_dir) / "keys"
    key_dir.mkdir(parents=True, exist_ok=True)
    s1_keys = key_dir / f"{split}_s1_keys.parquet"
    if not s1_keys.exists():
        _write_keys_chunked(processed / f"{split}_source1.parquet", s1_keys, idf_map, min_idf)
    cand_parts = []
    for source in (2, 3):
        part = key_dir / f"{split}_source{source}_keys.parquet"
        if not part.exists():
            _write_keys_chunked(processed / f"{split}_source{source}.parquet", part, idf_map, min_idf)
        cand_parts.append(part)

    n_shards = int((getattr(cfg, "db", None) or {}).get("block_shards", 1))
    caps = _pass_caps(cfg)
    out = Path(cfg.data_dir) / "candidates" / f"{split}_candidates.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    shard_dir = out.parent / f"{split}_cand_shards"
    if shard_dir.exists():
        shutil.rmtree(shard_dir)
    shard_dir.mkdir(parents=True, exist_ok=True)

    s1_sql = f"read_parquet('{s1_keys.as_posix()}')"
    cand_sql = ("read_parquet("
                f"['{cand_parts[0].as_posix()}','{cand_parts[1].as_posix()}'])")
    total = 0
    con = connect(cfg)
    t0 = time.perf_counter()
    n_cand, n_blocks = _prepare_cand_keys(con, cand_sql, s1_sql, caps)
    print(f"[block] candidate keys on probed blocks: {n_cand:,}; "
          f"valid blocks: {n_blocks:,} ({time.perf_counter() - t0:.0f}s)", flush=True)
    for shard in range(n_shards):
        part = shard_dir / f"shard_{shard:03d}.parquet"
        sql = _join_sql(s1_sql, cfg.cap,
                        shard=(shard, n_shards) if n_shards > 1 else None)
        t1 = time.perf_counter()
        con.execute(f"COPY ({sql}) TO '{part.as_posix()}' (FORMAT PARQUET)")
        n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{part.as_posix()}')").fetchone()[0]
        total += n
        print(f"[block] {split} shard {shard + 1}/{n_shards}: {n:,} pairs "
              f"({time.perf_counter() - t1:.0f}s)", flush=True)
    con.close()

    # Reassemble in shard order; the ranking is per-S1 and each S1 lives in exactly
    # one shard, so concatenation preserves it.
    writer = None
    for part in sorted(shard_dir.glob("shard_*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(batch_size=1_000_000):
            table = pa.Table.from_batches([batch])
            if writer is None:
                writer = pq.ParquetWriter(out, table.schema)
            writer.write_table(table)
    if writer is not None:
        writer.close()
    shutil.rmtree(shard_dir, ignore_errors=True)
    return {"split": split, "candidates": int(total), "tokens": len(idf_map),
            "shards": n_shards, "pass_caps": caps}
