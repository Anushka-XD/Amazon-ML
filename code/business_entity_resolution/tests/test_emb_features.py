import pandas as pd
import pytest

from ber.features import _feature_block

BASE = {
    "name_fold": ["a"], "name_fold_2": ["a"],
    "name_norm": ["a"], "name_norm_2": ["a"],
    "name_roman": ["a"], "name_roman_2": ["a"],
    "addr_norm": ["x"], "addr_norm_2": ["x"],
    "name_tokens": [["a"]], "name_tokens_2": [["a"]],
    "name_script": ["latin"], "name_script_2": ["latin"],
    "street_tokens": [["x"]], "street_tokens_2": [["x"]],
    "house_no": ["1"], "house_no_2": ["1"],
    "postal": ["12345"], "postal_2": ["12345"],
    "state_key": ["s"], "state_key_2": ["s"],
    "landmark_flag": [False], "addr_raw_missing": [False],
    "country": ["us"], "country_2": ["us"],
    "is_s2": [True], "pass_id": [1], "block_score": [1.0],
    "s1_degree": [1.0], "cand_degree": [1.0],
}


def test_key_only_pair_gets_not_retrieved_sentinel():
    out = _feature_block(pd.DataFrame(BASE))
    assert out["emb_cos"].iloc[0] == -1.0
    assert out["emb_retrieved"].iloc[0] == 0.0
    assert out["emb_agrees_with_block"].iloc[0] == 0.0


def test_agreement_requires_both_sources():
    out = _feature_block(pd.DataFrame({**BASE, "emb_cos": [0.97]}))
    assert out["emb_cos"].iloc[0] == pytest.approx(0.97)
    assert out["emb_retrieved"].iloc[0] == 1.0
    assert out["emb_agrees_with_block"].iloc[0] == 1.0


def test_retrieval_only_pair_is_not_an_agreement():
    out = _feature_block(pd.DataFrame({**BASE, "pass_id": [0], "emb_cos": [0.95]}))
    assert out["emb_retrieved"].iloc[0] == 1.0
    assert out["emb_agrees_with_block"].iloc[0] == 0.0


def test_missing_emb_column_is_treated_as_not_retrieved():
    frame = pd.DataFrame(BASE).drop(columns=["pass_id"])
    frame["pass_id"] = [1]
    out = _feature_block(pd.DataFrame(BASE))
    assert out["emb_cos"].iloc[0] == -1.0


def test_emb_cos_columns_are_declared_in_feature_order():
    from ber.features import feature_columns

    cols = feature_columns()
    for name in ("emb_cos", "emb_retrieved", "emb_agrees_with_block"):
        assert name in cols
