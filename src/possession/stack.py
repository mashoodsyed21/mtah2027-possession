"""Second-stage ("stacked") refinement of the frame probabilities.

Stage 1 (scripts/train_eval.py) predicts, per frame, P(H), P(A), P(D) and P(home restarts) from
tracking features. Its mistakes are temporally structured: short flickers inside a possession,
edges that lag or lead the true change, dead-ball stretches that fade in and out. Stage 2 sees the
stage-1 probabilities *in temporal context* (rolling statistics, past-vs-future contrasts, distance
to the nearest flip) plus a handful of raw ball / touch features and re-predicts the same two
quantities as stage 1:

    dead   P(ball dead)            on all graded frames
    team   P(home | ball alive)    on live frames, trained on both team orientations

Team symmetry is handled the same way as in stage 1: every frame is also seen with the teams
swapped (H <-> A probabilities, P(home restarts) -> 1 - P, h_* <-> a_* raw features) and the team
target flipped. At prediction time both orientations are averaged, so the output is exactly
team-symmetric. The restarter column is passed through from stage 1 unchanged.

All temporal operations run inside each graded run (normally the two halves) and never across one.
Everything is deterministic (LightGBM with deterministic=True, a fixed seed and LGB_THREADS threads).
"""

import lightgbm as lgb
import numpy as np
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from . import LGB_THREADS
from .features import _segments, roll_mean, side_mean, swap_teams

H, A, D, X = 0, 1, 2, 3
EPS = 1e-6

PROB_WINDOWS = (5, 25, 75, 250)   # frames, centred rolling statistics of P(D) and P(H | alive)
CONTRAST_K = (13, 25, 75)         # frames, mean of the next k minus mean of the previous k
CHANGE_CAP = 750                  # frames (30 s), cap for the distance-to-flip features
CHANGE_SMOOTH = 25                # frames, the flip features are also computed on a 1 s average

# Raw tracking features appended to the stage-2 input (whichever exist in the feature file).
RAW_FEATURES = ("ball_vis", "t_since_ball", "t_until_ball", "all_speed_m25")
RAW_PREFIXES = ("h_ball_tk_", "a_ball_tk_")   # soft touch-kernel shares, per team

# Modest model: the stage-2 input is low-dimensional and highly redundant in time.
LGB_STAGE2 = dict(objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=200,
                  feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=200, deterministic=True, force_col_wise=True, seed=0,
                  num_threads=LGB_THREADS, verbose=-1)


def raw_feature_names(available):
    """The raw features stage 2 uses, restricted to those present in ``available``."""
    avail = set(available)
    names = [n for n in RAW_FEATURES if n in avail]
    names += sorted(n for n in avail if n.startswith(RAW_PREFIXES))
    return names


def swap_probs(p):
    """Stage-1 probabilities [T, 4] with the teams exchanged: (A, H, D, 1 - P(home restarts))."""
    q = np.asarray(p, dtype=np.float32).copy()
    q[:, 0], q[:, 1] = p[:, 1], p[:, 0]
    q[:, 3] = 1.0 - p[:, 3]
    return q


def _logit(x):
    x = np.clip(np.asarray(x, dtype=np.float64), EPS, 1.0 - EPS)
    return np.log(x / (1.0 - x))


def _roll_extreme(x, w, segs, fn):
    """Centred rolling min / max over 2 * (w // 2) + 1 frames, inside each segment.

    mode='nearest' repeats the edge value, which equals shrinking the window at a segment edge.
    """
    out = np.full(len(x), np.nan, np.float32)
    size = 2 * (int(w) // 2) + 1
    for a, b in segs:
        out[a:b] = fn(x[a:b].astype(np.float32), size=size, mode="nearest")
    return out


def _roll_std(x, w, segs):
    """Centred rolling standard deviation, same window convention as roll_mean."""
    m1 = roll_mean(x, w, segs).astype(np.float64)
    m2 = roll_mean(np.asarray(x, dtype=np.float64) ** 2, w, segs).astype(np.float64)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0)).astype(np.float32)


def since_until_flip(flag, segs, cap=CHANGE_CAP):
    """Frames since the previous and until the next value change of a boolean series.

    A change "happens" at the first frame of the new value, so that frame has since = until = 0.
    Distances are capped at ``cap`` (also used when there is no change in that direction).
    """
    n = len(flag)
    since = np.full(n, cap, np.float32)
    until = np.full(n, cap, np.float32)
    for a, b in segs:
        f = np.asarray(flag[a:b], dtype=bool)
        m = b - a
        if m == 0:
            continue
        idx = np.arange(m)
        change = np.zeros(m, dtype=bool)
        change[1:] = f[1:] != f[:-1]
        last = np.maximum.accumulate(np.where(change, idx, -1))
        nxt = np.minimum.accumulate(np.where(change, idx, m)[::-1])[::-1]
        since[a:b] = np.where(last >= 0, np.minimum(idx - last, cap), cap)
        until[a:b] = np.where(nxt < m, np.minimum(nxt - idx, cap), cap)
    return since, until


def stage2_features(p, raw, period):
    """Build the stage-2 design matrix for one match.

    p: stage-1 probabilities [T, 4] (H, A, D, home restarts). raw: dict name -> float array [T]
    (see raw_feature_names). period: int array [T], 0 outside the graded halves.
    Returns (names, X) with X float32 [T, F]. Frames outside the halves get NaN context features.
    """
    p = np.asarray(p, dtype=np.float32)
    segs = _segments(period)
    p_dead = p[:, D]
    p_home = (p[:, H] / np.maximum(p[:, H] + p[:, A], EPS)).astype(np.float32)

    cols = {}
    for j, tag in enumerate(("h", "a", "d", "r")):
        cols["p_" + tag] = p[:, j]
        cols["lg_" + tag] = _logit(p[:, j])
    cols["p_home_alive"] = p_home
    cols["lg_home_alive"] = _logit(p_home)

    for tag, x in (("pd", p_dead), ("ph", p_home)):
        for w in PROB_WINDOWS:
            cols["%s_mean%d" % (tag, w)] = roll_mean(x, w, segs)
            cols["%s_min%d" % (tag, w)] = _roll_extreme(x, w, segs, minimum_filter1d)
            cols["%s_max%d" % (tag, w)] = _roll_extreme(x, w, segs, maximum_filter1d)
            cols["%s_std%d" % (tag, w)] = _roll_std(x, w, segs)
        for k in CONTRAST_K:
            cols["%s_ctr%d" % (tag, k)] = (side_mean(x, k, segs, future=True)
                                           - side_mean(x, k, segs, future=False))
        # Distance to the nearest flip of the frame-wise argmax, raw and on a 1 s average.
        for suffix, series in (("", x), ("_s%d" % CHANGE_SMOOTH, roll_mean(x, CHANGE_SMOOTH, segs))):
            since, until = since_until_flip(np.nan_to_num(series, nan=0.5) > 0.5, segs)
            cols["%s_since_flip%s" % (tag, suffix)] = since
            cols["%s_until_flip%s" % (tag, suffix)] = until
            cols["%s_near_flip%s" % (tag, suffix)] = np.minimum(since, until)

    for name in sorted(raw):
        cols["raw_" + name] = raw[name]

    names = list(cols)
    X_ = np.empty((len(p), len(names)), dtype=np.float32)
    for i, name in enumerate(names):
        X_[:, i] = cols[name]
    return names, X_


def match_matrices(p, raw, period):
    """Stage-2 matrices for one match in both team orientations: (names, X, X_swapped)."""
    names, X_ = stage2_features(p, raw, period)
    names_s, Xs = stage2_features(swap_probs(p), swap_teams(raw), period)
    if names_s != names:   # swap_teams must map the raw feature set onto itself
        raise ValueError("raw features are not closed under the team swap")
    return names, X_, Xs


def training_rows(matches, step=4, team_target="raw"):
    """Stack the training rows of several matches.

    matches: iterable of dicts with keys X, Xs (from match_matrices), g (graded codes, relabelled,
    3 = ungraded) and label (raw frame labels 0/1/2). Every ``step``-th graded frame is used.
    The dead target is g == D. The team target uses alive frames of the raw label by default
    (like stage 1, which trained its team model with team_relabel 0); team_target="relabelled"
    uses g instead. Both orientations are included; in the swapped one the team target flips.
    """
    dx, dy, tx, ty = [], [], [], []
    for m in matches:
        g = m["g"]
        lab_t = m["label"] if team_target == "raw" else g
        idx = np.flatnonzero(g < X)[::step]
        y_dead = (g[idx] == D).astype(np.int8)
        dx += [m["X"][idx], m["Xs"][idx]]
        dy += [y_dead, y_dead]
        alive = idx[lab_t[idx] < D]
        tx += [m["X"][alive], m["Xs"][alive]]
        ty += [(lab_t[alive] == H).astype(np.int8), (lab_t[alive] == A).astype(np.int8)]
    return (np.concatenate(dx), np.concatenate(dy)), (np.concatenate(tx), np.concatenate(ty))


def fit(matches, step=4, team_target="raw", params=None):
    """Fit the stage-2 dead and team models on ``matches`` (see training_rows)."""
    prm = dict(LGB_STAGE2, **(params or {}))
    (Xd, yd), (Xt, yt) = training_rows(matches, step, team_target)
    md = lgb.LGBMClassifier(**prm).fit(Xd, yd)
    mt = lgb.LGBMClassifier(**prm).fit(Xt, yt)
    return md, mt


def predict(models, m, p_stage1):
    """Stage-2 probabilities [T, 4] for one match (dict with X, Xs), team-symmetric by averaging.

    Column 3 (P(home restarts)) is copied from the stage-1 probabilities.
    """
    md, mt = models
    pd_ = 0.5 * (md.predict_proba(m["X"])[:, 1] + md.predict_proba(m["Xs"])[:, 1])
    ph = 0.5 * (mt.predict_proba(m["X"])[:, 1] + 1.0 - mt.predict_proba(m["Xs"])[:, 1])
    return np.stack([(1 - pd_) * ph, (1 - pd_) * (1 - ph), pd_,
                     np.asarray(p_stage1)[:, 3]], 1).astype(np.float32)


def blend(p1, p2, weight=0.5):
    """Geometric blend of two [T, 4] probability arrays on the H/A/D columns (renormalised)."""
    lp = (1.0 - weight) * np.log(np.clip(p1[:, :3], EPS, None)) \
        + weight * np.log(np.clip(p2[:, :3], EPS, None))
    q = np.exp(lp - lp.max(axis=1, keepdims=True))
    q /= q.sum(axis=1, keepdims=True)
    return np.concatenate([q, p1[:, 3:4]], axis=1).astype(np.float32)
