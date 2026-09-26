def test_oracle_entity_f05_is_precision_one():
    # With a perfect matcher, per-entity F0.5 collapses to 1.25R/(0.25+R).
    # R=1 -> 1.0 ; R=0.5 -> 1.25*.5/0.75 = 0.8333
    def f(r):
        return 1.25 * r / (0.25 + r)
    assert round(f(1.0), 6) == 1.0
    assert round(f(0.5), 4) == 0.8333
    assert f(0.0) == 0.0


def test_recall_monotonic_in_found_ratio():
    vals = [1.25 * (r / 4) / (0.25 + r / 4) for r in range(5)]
    assert vals == sorted(vals)


def test_summarize_counts_missing_by_cause():
    from ber.diagnostic import _summarize

    rows = [
        ("no_shared_name_token", 1, "latin", "latin"),
        ("no_shared_name_token", 0, "latin", "indic"),
        ("both_addr_missing", 0, "latin", "latin"),
        ("other", 0, "latin", "latin"),
    ]

    class FakeCon:
        def execute(self, sql):
            return [(100, 0.0, {})]

    report = _summarize(rows, FakeCon(), None)
    assert report["truth_pairs"] == 4
    assert report["missing_pairs"] == 3
    assert report["by_cause_missing"]["no_shared_name_token"] == 1
    assert report["by_cause_missing"]["both_addr_missing"] == 1
    assert report["script_pairs_missing"]["latin->indic"] == 1
    assert report["missing_share_by_cause"]["no_shared_name_token"] == 1 / 3
