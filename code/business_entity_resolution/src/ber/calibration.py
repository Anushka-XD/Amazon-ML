"""Decision-rule calibration for macro F_0.5.

Three knobs, tuned jointly on the held-out full-candidate set:

1. `threshold`   - minimum matcher probability to accept a pair.
2. `margin`      - reject an S1 entity when its top-1 and top-2 candidate scores are
                   closer than this. Macro F_0.5 weights precision twice as heavily as
                   recall, so "the model cannot tell these apart" must be expressible
                   as a decision, not just as a low score.
3. `singleton_tau` - below this, an entity is predicted empty even if a pair clears
                   the threshold. 5.59% of S1 entities are true singletons and each is
                   worth 1.0 for predicting empty versus 0.0 for a false merge, so
                   this single scalar is worth up to ~0.056 of the metric.

The injective one-to-one post-process is applied last and is already provably optimal
for this constraint set (see docs/FAILURES_AND_FIXES.md F14).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ber.postprocess import one_to_one
from ber.threshold import _entity_f05_from_codes


def _codes(values):
    codes, uniques = pd.factorize(np.asarray(values, dtype=object))
    return np.asarray(codes), uniques


def decide(probs, s1_ids, cand_ids, threshold, margin=0.0, singleton_tau=0.0,
           use_one_to_one=True, all_s1_ids=None):
    """Apply the decision rule and return a boolean keep-mask over the input pairs.

    `all_s1_ids` must include entities that end up with no kept pair (notably true
    singletons); they still contribute to the macro average.
    """
    probs = np.asarray(probs, dtype=np.float64)
    s1_codes, s1_uniques = _codes(s1_ids)
    n_s1 = len(s1_uniques)
    frame = pd.DataFrame({"s1": s1_codes, "p": probs})

    keep = probs >= threshold

    if margin > 0:
        # An entity is ambiguous when its best candidate barely beats its runner-up.
        # Entities with only one candidate above threshold are never ambiguous.
        sub = frame[keep]
        if len(sub):
            drop_entities = set()
            for s1, grp in sub.groupby("s1")["p"]:
                if len(grp) < 2:
                    continue
                top2 = grp.nlargest(2).to_numpy()
                if top2[0] - top2[1] < margin:
                    drop_entities.add(s1)
            if drop_entities:
                keep = keep & ~frame["s1"].isin(drop_entities).to_numpy()

    if singleton_tau > 0:
        best_per_s1 = pd.Series(probs).groupby(s1_codes).max()
        weak = set(best_per_s1[best_per_s1 < singleton_tau].index)
        if weak:
            keep = keep & ~frame["s1"].isin(weak).to_numpy()

    if use_one_to_one and keep.any():
        pairs = pd.DataFrame({"s1_id": pd.Series(s1_ids).to_numpy(),
                              "cand_id": pd.Series(cand_ids).to_numpy()})
        keep = keep & one_to_one(pd.Series(probs), pairs).to_numpy()

    return keep


def score(probs, s1_ids, cand_ids, truth_mask, all_s1_ids, n_true_by_s1=None, **kw):
    """Macro F_0.5 of a decision rule against the official entity-level definition.

    `all_s1_ids` is every evaluated Source 1 entity and `n_true_by_s1` maps each to
    its number of true matches. Both are required for a faithful score, because the
    metric distinguishes two kinds of entity that receive no candidate rows at all:

      * a true singleton (0 true matches, 0 predicted) scores **1.0** - correct rejection;
      * a matched entity whose truth pairs blocking never proposed scores **0.0**.

    Without `n_true_by_s1` the second case is indistinguishable from the first and
    would be scored as a free 1.0, inflating the result.
    """
    keep = decide(probs, s1_ids, cand_ids, **kw)
    probs = np.asarray(probs, dtype=np.float64)
    y = np.asarray(truth_mask, dtype=bool)

    s1_codes, s1_uniques = _codes(s1_ids)
    n_pred = np.bincount(s1_codes, weights=keep.astype(np.float64), minlength=len(s1_uniques))
    n_hit = np.bincount(s1_codes, weights=(y & keep).astype(np.float64), minlength=len(s1_uniques))

    index = {u: i for i, u in enumerate(s1_uniques.tolist())}
    scores = []
    for entity in np.asarray(all_s1_ids, dtype=object):
        i = index.get(entity)
        n_true = 0 if n_true_by_s1 is None else int(n_true_by_s1.get(entity, 0))
        pred = int(n_pred[i]) if i is not None else 0
        hit = int(n_hit[i]) if i is not None else 0
        if n_true == 0 and pred == 0:
            scores.append(1.0)
        elif n_true == 0 or pred == 0:
            scores.append(0.0)
        else:
            precision = hit / pred
            recall = hit / n_true
            denom = 0.25 * precision + recall
            scores.append((1.25 * precision * recall / denom) if denom > 0 else 0.0)
    return float(np.mean(scores)) if scores else 0.0


def tune(probs, s1_ids, cand_ids, truth_mask, all_s1_ids, n_true_by_s1,
         thresholds=None, margins=(0.0, 0.02, 0.05, 0.10, 0.20), singleton_taus=(0.0,),
         use_one_to_one=True, verbose=False):
    """Grid-search (threshold, margin, singleton_tau) to maximise macro F_0.5."""
    if thresholds is None:
        thresholds = np.round(np.arange(0.50, 0.981, 0.01), 3)
    best = {"score": -1.0, "threshold": 0.5, "margin": 0.0, "singleton_tau": 0.0}
    history = []
    for tau in singleton_taus:
        for margin in margins:
            for t in thresholds:
                s = score(probs, s1_ids, cand_ids, truth_mask, all_s1_ids, n_true_by_s1,
                          threshold=float(t), margin=float(margin), singleton_tau=float(tau),
                          use_one_to_one=use_one_to_one)
                history.append({"threshold": float(t), "margin": float(margin),
                                "singleton_tau": float(tau), "macro_f05": s})
                if s > best["score"]:
                    best = {"score": s, "threshold": float(t), "margin": float(margin),
                            "singleton_tau": float(tau)}
            if verbose:
                print(f"  margin={margin} tau={tau} best so far {best['score']:.4f} "
                      f"@ t={best['threshold']}", flush=True)
    best["use_one_to_one"] = use_one_to_one
    best["history"] = history
    return best


def save(cfg, params, path=None):
    path = Path(path or (Path(cfg.models_dir) / "threshold.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params, indent=2), encoding="utf-8")
    return str(path)


def load(cfg, path=None):
    path = Path(path or (Path(cfg.models_dir) / "threshold.json"))
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
