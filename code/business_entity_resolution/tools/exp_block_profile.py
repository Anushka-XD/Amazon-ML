"""Where does the sharded block join actually consume memory?

Repeated OOM at 5.5 GB with the shard filter in place, so measure each stage of the
join separately on one shard: candidate keys, S1 keys, probed blocks, raw join rows.
"""
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, "code/business_entity_resolution/src")
from ber.config import Config
from ber.db import connect

cfg = Config.load("code/business_entity_resolution/config.json")
key_dir = Path(cfg.data_dir) / "keys"
s1k = (key_dir / "train_s1_keys.parquet").as_posix()
cand = (f"['{(key_dir / 'train_source2_keys.parquet').as_posix()}',"
        f"'{(key_dir / 'train_source3_keys.parquet').as_posix()}']")
N_SHARDS, SHARD = 16, 0
caps = {int(k): int(v) for k, v in cfg.pass_caps.items()}
cases = " ".join(f"WHEN {p} THEN {v}" for p, v in sorted(caps.items()))
fallback = max(caps.values())

con = connect(cfg)
con.execute("SET memory_limit='6GB'")
con.execute("SET threads=3")


def step(label, sql):
    t0 = time.time()
    con.execute(sql)
    n = con.execute("SELECT COUNT(*) FROM probe").fetchone()[0]
    print(f"{label:34s} rows={n:>14,}  {time.time() - t0:6.1f}s", flush=True)
    return n


con.execute(f"CREATE TEMP TABLE probe AS SELECT entity_id, pass_id, key FROM read_parquet('{s1k}') "
            f"WHERE key <> '' AND hash(entity_id) % {N_SHARDS} = {SHARD} LIMIT 5")
print("shard S1 key rows:",
      con.execute("SELECT COUNT(*) FROM probe").fetchone()[0])
con.execute("DROP TABLE probe")

con.execute("CREATE TEMP TABLE s AS SELECT entity_id, pass_id, key, block_score "
            f"FROM read_parquet('{s1k}') WHERE key <> '' AND hash(entity_id) % {N_SHARDS} = {SHARD}")
print(f"{'S1 keys in shard':34s} rows={con.execute('SELECT COUNT(*) FROM s').fetchone()[0]:>14,}")
con.execute("CREATE TEMP TABLE skey AS SELECT DISTINCT pass_id, key FROM s")
print(f"{'distinct S1 (pass,key)':34s} rows={con.execute('SELECT COUNT(*) FROM skey').fetchone()[0]:>14,}")

con.execute("CREATE TEMP TABLE probe AS SELECT * FROM s")
print(f"{'baseline':34s} rows={con.execute('SELECT COUNT(*) FROM probe').fetchone()[0]:>14,}")

# 1) candidate rows holding a probed key
con.execute("DROP TABLE probe")
con.execute("CREATE TEMP TABLE probe AS "
            "SELECT c.entity_id, c.pass_id, c.key FROM "
            f"(SELECT entity_id, pass_id, key FROM read_parquet({cand}) WHERE key <> '') c "
            "SEMI JOIN skey ON skey.pass_id = c.pass_id AND skey.key = c.key")
step("cand rows on probed keys", "SELECT 1")

# 2) valid blocks after the cap
con.execute("DROP TABLE probe")
con.execute(f"CREATE TEMP TABLE probe AS "
            "SELECT c.pass_id, c.key FROM "
            f"(SELECT entity_id, pass_id, key FROM read_parquet({cand}) WHERE key <> '') c "
            "SEMI JOIN skey ON skey.pass_id = c.pass_id AND skey.key = c.key "
            f"GROUP BY c.pass_id, c.key HAVING COUNT(*) <= CASE c.pass_id {cases} ELSE {fallback} END")
step("valid blocks after cap", "SELECT 1")

# 3) the raw join
con.execute("DROP TABLE probe")
con.execute("CREATE TEMP TABLE probe AS "
            "SELECT s.entity_id AS s1_id, c.entity_id AS cand_id, s.pass_id, s.block_score "
            "FROM s JOIN (SELECT entity_id, pass_id, key FROM "
            f"read_parquet({cand}) WHERE key <> '') c "
            "ON s.key = c.key AND s.pass_id = c.pass_id")
n = step("raw join (no cap filter)", "SELECT 1")
print(f"   -> raw join is {n / max(con.execute('SELECT COUNT(*) FROM s').fetchone()[0], 1):.0f} rows per S1 key")
con.close()
