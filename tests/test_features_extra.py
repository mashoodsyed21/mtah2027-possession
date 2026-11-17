"""Tests for possession.features_extra.

The synthetic tests need no data. The DFL tests use one open DFL match (extdata/dfl/<ID>.npz,
written by scripts/build_dfl.py; override the folder with POSSESSION_DFL_DIR) and are skipped
when it is missing. Run with `pytest -s` to see the per-feature report: NaN share and univariate
ROC-AUC against the DFL labels (dead vs alive; home vs away on alive frames), on the clean match
and on a degraded copy that mimics the target tracking.
"""

import ast
import os
import time
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from possession import features
from possession.degrade import degrade
from possession.features import CAP_S, FPS, swap_teams
from possession.features_extra import build_extra

DFL_DIR = Path(os.environ.get("POSSESSION_DFL_DIR",
                              Path(__file__).resolve().parents[2] / "extdata" / "dfl"))
MATCH_ID = "J03WMX"
MAX_SECONDS = 30.0
GAP_MEDIAN_S, GAP_SIGMA, N_GAPS = 10.7, 0.9, 150   # lognormal ball gaps, p90 ~ 34 s


def same(a, b):
    return np.array_equal(a, b, equal_nan=True)


def swap_match(m):
    return dict(m, home=m["away"], away=m["home"])


def assert_swap_symmetric(m, f, max_mismatch=0):
    """build_extra on the team-swapped match == swap_teams of the original features."""
    g = build_extra(swap_match(m))
    s = swap_teams(f)
    assert set(g) == set(s)
    for k in sorted(g):
        assert np.array_equal(np.isnan(g[k]), np.isnan(s[k])), k
        bad = ~np.isclose(g[k], s[k], rtol=1e-5, atol=1e-5, equal_nan=True)
        assert bad.sum() <= max_mismatch, (k, int(bad.sum()))


# -- synthetic scenes ---------------------------------------------------------------------------

def static_match(home, away, ball, T=300):
    """A one-half match in which everybody (and the ball) stands still for T frames."""
    tile = lambda p: np.repeat(np.asarray(p, np.float32)[None], T, 0)
    return {"home": tile(home), "away": tile(away), "ball": tile(ball),
            "period": np.ones(T, np.int8), "label": np.zeros(T, np.int8),
            "pitch": np.array([105.0, 68.0], np.float32)}


def line_up(xs, ys):
    return np.stack([np.asarray(xs, float), np.asarray(ys, float)], 1)


# Home attacks +x (stands in the negative half), away attacks -x.
HOME = line_up([-50, -30, -30, -30, -30, -15, -15, -15, -15, -5, -5],
               [0, -25, -8, 8, 25, -20, -5, 5, 20, -10, 10])
AWAY = -HOME


def test_kickoff_and_clear_control():
    home = HOME.copy()
    home[10] = (-0.6, 0.0)                     # kicker next to the ball on the centre spot
    m = static_match(home, AWAY, (0.0, 0.0))
    m["ball"][200:] = np.nan                   # ball lost for the last 100 frames
    f = build_extra(m)
    assert np.all(f["sp_kickoff"] == 1)
    assert np.all(f["h_t_since_ball_ctrl"][:200] == 0) and np.all(f["a_t_since_ball_ctrl"] == CAP_S)
    assert f["h_ball_ctrl_past25"][100] == 1 and f["a_ball_ctrl_past25"][100] == 0
    assert np.all(f["walk_share_c51"] == 1) and np.all(f["fast_share_c51"][1:-1] == 0)
    # restart cues and the open gap at the end of the half
    assert f["ball_gap_len_s"][250] == pytest.approx(100 / FPS) and f["ball_gap_len_s"][150] == 0
    assert f["h_d_ball_last"][250] == pytest.approx(0.6, abs=1e-5)
    assert np.isnan(f["h_d_ball_next"][250]) and np.isnan(f["ball_gap_speed"][250])
    assert_swap_symmetric(m, f)


def test_corner_throw_in_and_goal_kick():
    home = HOME.copy()
    home[10] = (52.0, 33.5)                    # at the corner flag of the attacked end
    home[9] = (45.0, 5.0)                      # an attacker inside the away penalty area
    away = AWAY.copy()                         # away keeper (50, 0) in its six-yard box
    away[1:5] = ((40, -3), (40, 3), (38, -10), (38, 10))   # defenders: >= 6 players in the zone
    m = static_match(home, away, (52.3, 33.8))
    f = build_extra(m)
    assert np.all(f["h_sp_corner"] == 1) and np.all(f["a_sp_corner"] == 0)
    assert np.all(f["a_sp_goalkick_ball"] == 0)     # an opponent is inside that penalty area
    assert_swap_symmetric(m, f)

    away = AWAY.copy()
    away[3] = (10.0, -34.2)                    # away player on the touchline with the ball
    m = static_match(HOME, away, (10.0, -34.5))
    m["ball"][:100] = np.nan                   # ball only seen from frame 100 on
    f = build_extra(m)
    assert np.all(f["a_sp_throw_ball"] == 1) and np.all(f["h_sp_throw_ball"] == 0)
    assert_swap_symmetric(m, f)

    # home keeper (-50, 0) in its six-yard box, no away player in that penalty area, ball next
    # seen on the six-yard line
    m = static_match(HOME, AWAY, (-48.0, 3.0))
    m["ball"][:100] = np.nan
    f = build_extra(m)
    assert np.all(f["h_sp_goalkick_ball"] == 1) and np.all(f["a_sp_goalkick_ball"] == 0)
    assert_swap_symmetric(m, f)


# -- DFL match ----------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dfl():
    path = DFL_DIR / (MATCH_ID + ".npz")
    if not path.is_file():
        pytest.skip("DFL match %s not available" % path)
    z = np.load(path)
    return {k: z[k] for k in z.files}


def degraded_copy(m, seed):
    rng = np.random.default_rng(seed)
    gaps = np.sort(rng.lognormal(np.log(GAP_MEDIAN_S * FPS), GAP_SIGMA, N_GAPS).astype(np.int64))
    ball, home, away = degrade(m, np.maximum(gaps, 1), rng)
    return dict(m, ball=ball, home=home, away=away)


@pytest.fixture(scope="module")
def matches(dfl):
    """{'clean': (match, features, seconds), 'degraded': (...)}."""
    out = {}
    for name, m in (("clean", dfl), ("degraded", degraded_copy(dfl, seed=0))):
        t0 = time.perf_counter()
        f = build_extra(m)
        out[name] = (m, f, time.perf_counter() - t0)
    return out


def test_names_and_dtypes(dfl, matches):
    _, f, _ = matches["clean"]
    T = len(dfl["period"])
    for k, v in f.items():
        assert v.dtype == np.float32 and v.shape == (T,), k
        if k.startswith("h_"):
            assert "a_" + k[2:] in f, k
        elif k.startswith("a_"):
            assert "h_" + k[2:] in f, k
        else:
            assert "_h" not in k and "_a" not in k and "home" not in k, k
    # no clash with the base features (names do not depend on the match length)
    short = {k: (v[:3000] if k != "pitch" else v) for k, v in dfl.items()}
    assert not set(f) & set(features.build(short))


@pytest.mark.parametrize("name", ["clean", "degraded"])
def test_swap_symmetry(matches, name):
    m, f, _ = matches[name]
    assert_swap_symmetric(m, f, max_mismatch=2)    # float rounding at exact ties only


def test_determinism(dfl, matches):
    m, f, _ = matches["degraded"]
    again = degraded_copy(dfl, seed=0)
    assert same(again["ball"], m["ball"]) and same(again["home"], m["home"])
    g = build_extra(again)
    assert all(same(f[k], g[k]) for k in f)


def ball_tokens():
    """BALL_TOKENS of scripts/train_eval.py (its ball-free dead model drops features by name)."""
    path = Path(__file__).resolve().parents[1] / "scripts" / "train_eval.py"
    if path.is_file():
        for node in ast.parse(path.read_text(encoding="utf-8")).body:
            if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", "") == "BALL_TOKENS":
                return tuple(ast.literal_eval(node.value))
    return ("ball",)


def window(m, a, b):
    return {k: (v[a:b] if k != "pitch" else v) for k, v in m.items()}


def test_ball_dependence_is_named(matches):
    """Every feature that changes with the ball track carries a ball token in its name."""
    m, _, _ = matches["degraded"]
    w = window(m, 20000, 35000)
    ball = w["ball"].copy()
    ball[2000:6000] = np.nan                       # remove a block, shift the rest
    ball[6000:] += np.float32(3.0)
    f, g = build_extra(w), build_extra(w, ball=ball)
    tokens = ball_tokens()
    changed = [k for k in f if not same(f[k], g[k])]
    assert changed
    assert [k for k in changed if not any(t in k for t in tokens)] == []


def test_no_label_or_cross_half_dependence(dfl, matches):
    m, f, _ = matches["degraded"]
    # the label is never read
    g = build_extra(dict(m, label=np.full_like(m["label"], 3)))
    assert all(same(f[k], g[k]) for k in f)
    # nothing crosses the half boundary: perturbing the first half leaves the second unchanged
    b = int(np.flatnonzero(np.diff(m["period"]) != 0)[0]) + 1      # first frame of half 2
    w = window(m, b - 6000, b + 6000)
    pert = dict(w, home=w["home"].copy(), ball=w["ball"].copy())
    pert["home"][:6000] += np.float32(4.0)
    pert["ball"][:6000] = np.nan
    f1, f2 = build_extra(w), build_extra(pert)
    assert [k for k in f1 if not same(f1[k][6000:], f2[k][6000:])] == []
    # ungraded frames are NaN everywhere
    w["period"] = w["period"].copy()
    w["period"][:500] = 0
    f3 = build_extra(w)
    assert all(np.isnan(v[:500]).all() for v in f3.values())


def test_runtime(matches):
    for name, (m, f, seconds) in matches.items():
        print("\n%s: %d features on %d frames in %.1f s" % (name, len(f), len(m["period"]), seconds))
        assert seconds < MAX_SECONDS


def univariate_auc(x, y):
    ok = ~np.isnan(x)
    if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
        return np.nan
    return roc_auc_score(y[ok], x[ok])


def test_univariate_report(matches):
    best = {}
    for name, (m, f, _) in matches.items():
        lab = m["label"]
        graded = (m["period"] > 0) & (lab < 3)
        alive = graded & (lab < 2)
        rows = []
        for k in sorted(f):
            v = f[k]
            rows.append((k, np.isnan(v[graded]).mean(),
                         univariate_auc(v[graded], (lab[graded] == 2).astype(int)),
                         univariate_auc(v[alive], (lab[alive] == 0).astype(int))))
        print("\n== %s: NaN share, ROC-AUC dead-vs-alive, ROC-AUC home-vs-away (alive) ==" % name)
        for k, nan, a_dead, a_team in rows:
            extra = ""
            if "sp_" in k:          # sparse 0/1 templates: AUC says little, precision more
                on = graded & (f[k] == 1)
                precision = (lab[on] == 2).mean() if on.any() else np.nan
                extra = "  fires %.3f  P(dead|fires) %.3f" % (on.sum() / graded.sum(), precision)
            print("%-22s nan %.3f  dead %.3f  home %.3f%s" % (k, nan, a_dead, a_team, extra))
        for col, task in ((2, "dead-vs-alive"), (3, "home-vs-away")):
            strength = [(max(r[col], 1 - r[col]), r[0]) for r in rows if not np.isnan(r[col])]
            strength.sort(reverse=True)
            print("-- %s top 15 (%s, max(AUC, 1-AUC)):" % (name, task))
            print("   " + ", ".join("%s %.3f" % (k, s) for s, k in strength[:15]))
            best[(name, task)] = strength[0][0]
    # the new evidence must carry real signal even on the degraded match
    assert best[("degraded", "dead-vs-alive")] > 0.75
    assert best[("degraded", "home-vs-away")] > 0.75
