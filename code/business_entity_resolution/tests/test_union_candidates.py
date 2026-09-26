import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ber.config import Config
from ber.union_candidates import union_candidates


def _cfg(tmp_path):
    return Config.load("code/business_entity_resolution/config.json")._replace(
        data_dir=tmp_path) if hasattr(Config, "_replace") else type("C", (), {
            "data_dir": tmp_path, "db": {"memory_limit": "1GB", "threads": 1,
                                          "max_temp": "2GiB"},
        })()


def _write(path, rows, schema):
    pq.write_table(pa.table(rows, schema=schema), path)


def test_union_preserves_provenance_of_both_sources(tmp_path):
    key = tmp_path / "key.parquet"
    ann = tmp_path / "ann.parquet"
    _write(key,
           {"s1_id": ["S1-1", "S1-1", "S1-2"],
            "cand_id": ["S2-1", "S2-2", "S2-3"],
            "pass_id": np.array([1, 8, 11], dtype="int8"),
            "block_score": np.array([1.0, 0.7, 0.85], dtype="float32"),
            "n_passes": np.array([1, 2, 1], dtype="int16"),
            "is_s2": [True, True, False]},
           pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()),
                      ("pass_id", pa.int8()), ("block_score", pa.float32()),
                      ("n_passes", pa.int16()), ("is_s2", pa.bool_())]))
    # S2-1 is proposed by BOTH sources; S2-9 only by retrieval.
    _write(ann,
           {"s1_id": ["S1-1", "S1-3"], "cand_id": ["S2-1", "S2-9"],
            "emb_cos": np.array([0.97, 0.91], dtype="float32")},
           pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()),
                      ("emb_cos", pa.float32())]))

    out = tmp_path / "union.parquet"
    stats = union_candidates(_cfg(tmp_path), key, ann, out)

    df = pq.read_table(out).to_pandas().set_index(["s1_id", "cand_id"])
    assert stats["total_pairs"] == 4
    assert stats["both_sources"] == 1
    assert stats["key_only"] == 2
    assert stats["retrieval_only"] == 1

    # Agreed pair keeps both kinds of evidence.
    agreed = df.loc[("S1-1", "S2-1")]
    assert agreed["pass_id"] == 1
    assert agreed["emb_cos"] == pytest.approx(0.97)
    # Key-only keeps its pass but has no retrieval score; -1 marks "not retrieved".
    key_only = df.loc[("S1-1", "S2-2")]
    assert key_only["pass_id"] == 8
    assert key_only["emb_cos"] == pytest.approx(-1.0)
    # Retrieval-only has no key provenance.
    ann_only = df.loc[("S1-3", "S2-9")]
    assert ann_only["pass_id"] == 0
    assert ann_only["emb_cos"] == pytest.approx(0.91)


def test_union_never_drops_a_candidate(tmp_path):
    key = tmp_path / "key.parquet"
    ann = tmp_path / "ann.parquet"
    _write(key,
           {"s1_id": [f"S1-{i}" for i in range(50)],
            "cand_id": [f"S2-{i}" for i in range(50)],
            "pass_id": np.ones(50, dtype="int8"),
            "block_score": np.ones(50, dtype="float32"),
            "n_passes": np.ones(50, dtype="int16"),
            "is_s2": [True] * 50},
           pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()),
                      ("pass_id", pa.int8()), ("block_score", pa.float32()),
                      ("n_passes", pa.int16()), ("is_s2", pa.bool_())]))
    _write(ann,
           {"s1_id": [f"S1-{i}" for i in range(25, 75)],
            "cand_id": [f"S2-{i}" for i in range(25, 75)],
            "emb_cos": np.full(50, 0.9, dtype="float32")},
           pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()),
                      ("emb_cos", pa.float32())]))
    out = tmp_path / "union.parquet"
    stats = union_candidates(_cfg(tmp_path), key, ann, out)
    # 50 key + 50 ann - 25 overlapping = 75
    assert stats["total_pairs"] == 75
