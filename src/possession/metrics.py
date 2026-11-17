"""Thin wrapper around the OFFICIAL competition scorer (possession_metrics.py).

Nothing here re-implements metric logic: the official module is imported as-is and only its own
functions are composed. Label codes are the official ones: 0=H, 1=A, 2=D, 3=X.

The official file is located in ``<repo>/../more-than-a-hack-2027/evaluation`` by default; set the
environment variable ``POSSESSION_EVAL_DIR`` to point somewhere else.
"""

import os
import sys
from pathlib import Path

import numpy as np

_DEFAULT_EVAL_DIR = (Path(__file__).resolve().parents[3]
                     / "more-than-a-hack-2027" / "evaluation")
_EVAL_DIR = Path(os.environ.get("POSSESSION_EVAL_DIR", str(_DEFAULT_EVAL_DIR)))
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

import possession_metrics as pm  # noqa: E402  (official scorer, imported unmodified)

H, A, D, X = pm.H, pm.A, pm.D, pm.X
LABELS = pm.LABELS

# Full official report, for when every column is wanted (slower: transitions, stats, ...).
evaluate = pm.evaluate
valid_runs = pm.valid_runs


def score(gt_codes, pred_codes):
    """Headline numbers of the official scorer for one sequence.

    gt_codes, pred_codes: integer arrays of equal length with codes 0..3. The graded spans are the
    non-X runs of gt_codes (official ``valid_runs``), so the arrays can be any length -- a full
    match with its halftime hole, a single half, or a synthetic sequence.

    Returns a dict with spell_f1_mean (the ranking metric), macro_f1 (first tie-break),
    n_gt_spells and n_pred_spells. Values are identical to the same keys of ``pm.evaluate``; this
    just skips the transition / bootstrap work so it is cheap enough for grid searches.
    """
    gt = np.asarray(gt_codes).astype(np.int8)
    pred = np.asarray(pred_codes).astype(np.int8)
    if len(gt) != len(pred):
        raise ValueError("length mismatch: gt {} vs pred {}".format(len(gt), len(pred)))

    cm3 = pm.scored_confusion(pm.confusion(gt, pred))
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)  # exactly as pm.evaluate does for spell scores
    spells = pm.spell_scores(gt, filled, runs)
    return {
        "spell_f1_mean": spells["mean_f1"],
        "macro_f1": pm.macro_f1(cm3),
        "n_gt_spells": spells["n_gt"],
        "n_pred_spells": spells["n_pred"],
    }
