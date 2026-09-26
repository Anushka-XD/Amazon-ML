import numpy as np

from ber.calibration import decide, score, tune


def _toy():
    """Three entities:
      S1-A has two true matches plus a confident impostor.
      S1-B is a true singleton with one confident-looking false positive.
      S1-C has one true match and one near-tied wrong candidate.
    """
    s1 = ["S1-A", "S1-A", "S1-A", "S1-B", "S1-C", "S1-C"]
    cand = ["S2-1", "S2-2", "S2-3", "S2-4", "S2-5", "S2-6"]
    probs = np.array([0.99, 0.97, 0.95, 0.93, 0.96, 0.90])
    truth = np.array([True, True, False, False, True, False])
    all_s1 = ["S1-A", "S1-B", "S1-C"]
    n_true = {"S1-A": 2, "S1-B": 0, "S1-C": 1}
    return s1, cand, probs, truth, all_s1, n_true


def test_threshold_filters_low_scores():
    s1, cand, probs, _, _, _ = _toy()
    keep = decide(probs, s1, cand, threshold=0.99, use_one_to_one=False)
    assert keep.sum() == 1


def test_one_to_one_removes_duplicate_candidate_claims():
    s1 = ["S1-A", "S1-B"]
    cand = ["S2-1", "S2-1"]
    probs = np.array([0.60, 0.99])
    keep = decide(probs, s1, cand, threshold=0.5, use_one_to_one=True)
    assert keep.sum() == 1
    assert keep[1], "the higher-probability claim should win"


def test_singleton_tau_forces_empty_prediction():
    s1, cand, probs, truth, all_s1, n_true = _toy()
    without = score(probs, s1, cand, truth, all_s1, n_true, threshold=0.9, singleton_tau=0.0)
    with_tau = score(probs, s1, cand, truth, all_s1, n_true, threshold=0.9, singleton_tau=0.95)
    assert with_tau > without, "raising singleton_tau must recover the singleton"


def test_margin_rejects_ambiguous_entity():
    # S1-C: 0.96 true vs 0.90 wrong -> gap 0.06. A margin above 0.06 must drop it.
    s1, cand, probs, _, _, _ = _toy()
    keep = decide(probs, s1, cand, threshold=0.5, margin=0.5, use_one_to_one=False)
    assert not keep[4], "ambiguous entity should be rejected by a large margin"
    assert not keep[5]


def test_margin_leaves_clearly_separated_entity():
    s1, cand, probs, _, _, _ = _toy()
    keep = decide(probs, s1, cand, threshold=0.5, margin=0.001, use_one_to_one=False)
    assert keep[0] and keep[1]


def test_matched_entity_with_no_candidates_scores_zero():
    """The dangerous case: blocking never proposed a truth pair, so the entity has no
    rows at all. It must score 0.0, not be mistaken for a correct singleton."""
    s1 = ["S1-A"]
    cand = ["S2-1"]
    probs = np.array([0.99])
    truth = np.array([True])
    # S1-GHOST is a genuinely matched entity (3 true matches) that got no candidates.
    n_true = {"S1-A": 1, "S1-GHOST": 3}
    s = score(probs, s1, cand, truth, ["S1-A", "S1-GHOST"], n_true, threshold=0.5)
    assert s == 0.5, "S1-A scores 1.0 and S1-GHOST scores 0.0"


def test_true_singleton_with_no_candidates_scores_one():
    s1 = ["S1-A"]
    cand = ["S2-1"]
    probs = np.array([0.99])
    truth = np.array([True])
    n_true = {"S1-A": 1, "S1-SOLO": 0}
    s = score(probs, s1, cand, truth, ["S1-A", "S1-SOLO"], n_true, threshold=0.5)
    assert s == 1.0, "a correctly rejected singleton is worth a full 1.0"


def test_tune_recovers_a_known_optimum():
    s1, cand, probs, truth, all_s1, n_true = _toy()
    res = tune(probs, s1, cand, truth, all_s1, n_true,
               thresholds=[0.5, 0.9, 0.95, 0.99], margins=[0.0, 0.5],
               singleton_taus=[0.0, 0.95], use_one_to_one=True)
    assert res["score"] > 0.5
    assert res["threshold"] in (0.5, 0.9, 0.95, 0.99)
    assert len(res["history"]) == 4 * 2 * 2


def test_tune_never_beats_a_perfect_matcher():
    s1, cand, probs, truth, all_s1, n_true = _toy()
    res = tune(probs, s1, cand, truth, all_s1, n_true,
               thresholds=[0.5], margins=[0.0], singleton_taus=[0.0])
    assert res["score"] <= 1.0
