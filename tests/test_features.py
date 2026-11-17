"""Tests for possession.features (the base feature set), on a synthetic two-half match.

The properties the training pipeline relies on: features never read the label, temporal context
never crosses the half boundary (DFL stores the two halves back to back, so a frame difference
across that seam sees the teams swap ends), and swap_teams of the features equals the features of
the team-swapped match, for the base and the extra set together (scripts/train_eval.py augments
and symmetrises the team and restart models with swap_teams).
"""

import numpy as np

from possession import features
from possession.features import swap_teams
from possession.features_extra import build_extra

HALF = 900          # frames per half


def synthetic_match(seed=0):
    """Two back-to-back halves of random-walking players and ball; the teams change ends."""
    rng = np.random.default_rng(seed)
    T, n = 2 * HALF, 11
    period = np.repeat(np.array([1, 2], np.int8), HALF)
    walk = lambda shape, sd: np.cumsum(rng.normal(0.0, sd, shape), 0).astype(np.float32)
    end = np.where(period == 1, -1.0, 1.0).astype(np.float32)[:, None]   # home's own half
    home = walk((T, n, 2), 0.05)
    away = walk((T, n, 2), 0.05)
    home[:, :, 0] += 25.0 * end
    away[:, :, 0] -= 25.0 * end
    home[:, :, 1] += np.linspace(-25, 25, n, dtype=np.float32)
    away[:, :, 1] += np.linspace(-25, 25, n, dtype=np.float32)
    ball = walk((T, 2), 0.3)
    ball[300:420] = np.nan                       # a ball gap in each half
    ball[HALF + 200:HALF + 260] = np.nan
    home[500:540, 3] = np.nan                    # a player dropout
    label = rng.integers(0, 3, T).astype(np.int8)
    return {"home": home, "away": away, "ball": ball, "period": period, "label": label,
            "pitch": np.array([105.0, 68.0], np.float32)}


def all_features(m):
    f = features.build(m)
    f.update(build_extra(m))
    return f


def test_base_features_ignore_the_label():
    m = synthetic_match()
    f = features.build(m)
    g = features.build(dict(m, label=np.full_like(m["label"], 3)))
    assert all(np.array_equal(f[k], g[k], equal_nan=True) for k in f)


def test_base_features_do_not_cross_the_half_boundary():
    m = synthetic_match()
    f = features.build(m)
    pert = dict(m, home=m["home"].copy(), away=m["away"].copy(), ball=m["ball"].copy())
    pert["home"][HALF:] += np.float32(7.0)       # move everything in the second half
    pert["away"][HALF:] -= np.float32(5.0)
    pert["ball"][HALF:] = np.nan
    g = features.build(pert)
    changed = [k for k in f if not np.array_equal(f[k][:HALF], g[k][:HALF], equal_nan=True)]
    assert changed == []


def test_swap_teams_matches_the_swapped_match():
    m = synthetic_match()
    f = all_features(m)
    g = all_features(dict(m, home=m["away"], away=m["home"]))
    s = swap_teams(f)
    assert set(g) == set(s)
    for k in sorted(g):
        assert np.array_equal(np.isnan(g[k]), np.isnan(s[k])), k
        assert np.allclose(g[k], s[k], rtol=1e-5, atol=1e-5, equal_nan=True), k
