import duckdb

DEFAULTS = {
    "memory_limit": "5GB",
    "threads": 4,
    "max_temp": "12GiB",
}


def connect(cfg=None, memory_limit=None, threads=None, max_temp=None):
    """Single DuckDB connection factory so limits come from config, not per-module constants.

    The pipeline was written for a 23 GB / 50-60 GiB-temp host. Callers that pass explicit
    values keep their old behaviour; otherwise values come from cfg.db, then from DEFAULTS.
    """
    db = (getattr(cfg, "db", None) or {}) if cfg is not None else {}
    memory_limit = memory_limit or db.get("memory_limit", DEFAULTS["memory_limit"])
    threads = int(threads or db.get("threads", DEFAULTS["threads"]))
    max_temp = max_temp or db.get("max_temp", DEFAULTS["max_temp"])
    data_dir = getattr(cfg, "data_dir", None) or "DATA"
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    tmp = f"{data_dir}/tmp"
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute(f"PRAGMA max_temp_directory_size='{max_temp}'")
    return con
