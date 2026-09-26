import numpy as np
import pytest

from ber.ann_retrieve import _oracle_bound


def test_oracle_bound_matches_the_metric_formula():
    # At perfect recall and precision a matcher scores 1.0.
    assert _oracle_bound(1.0) == pytest.approx(1.0)
    # At zero recall nothing can be predicted, so the ceiling is 0.
    assert _oracle_bound(0.0) == pytest.approx(0.0)
    # f(R) = 1.25R/(0.25+R): R=0.5 -> 1.25*0.5/0.75 = 0.8333
    assert _oracle_bound(0.5) == pytest.approx(0.8333333, abs=1e-6)


def test_oracle_bound_is_monotonic_in_recall():
    vals = [_oracle_bound(r) for r in np.linspace(0.1, 1.0, 20)]
    assert vals == sorted(vals)


def test_oracle_bound_clamps_out_of_range_recall():
    assert _oracle_bound(1.5) == pytest.approx(1.0)
    assert _oracle_bound(-0.2) == pytest.approx(0.0)


def test_recall_needed_for_a_target_score():
    """0.988 macro F0.5 at perfect precision implies pair recall ~0.941."""
    r = 0.941
    assert _oracle_bound(r) == pytest.approx(0.988, abs=0.001)
    # And the 0.99 target needs ~0.952.
    assert _oracle_bound(0.952) == pytest.approx(0.99, abs=0.001)
    # A 0.99 target is unreachable below this recall even with a flawless matcher.
    assert _oracle_bound(0.90) < 0.99
    assert _oracle_bound(0.95) < 0.99
