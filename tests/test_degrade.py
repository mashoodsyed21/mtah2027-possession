"""Tests for possession.degrade.

The statistics tests use two DFL open-data matches from extdata/dfl and are skipped when that
folder is absent. The target match itself is never loaded here: a synthetic gap pool with the
target's measured aggregates (median 10.7 s, ~150 gaps, longest ~220 s) stands in for
gap_profile(target), and TARGET below holds the target's measured aggregate statistics.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from possession import degrade as dg
from possession.features import FPS, _segments, speed

DFL = Path(__file__).resolve().parents[2] / "extdata" / "dfl"
MATCH_IDS = ("J03WMX", "J03WN1")

# Aggregates of the target tracking, graded frames. The live / dead ball shares come from a
# labelled 5-minute window only; the overall share from the whole match.
TARGET = {
    "ball_miss": 0.41, "ball_miss_live": 0.08, "ball_miss_dead": 0.34,
    "gap_median_s": 10.7, "gaps": 150,
    "player_speed_p50": 1.36, "player_speed_p99": 6.6, "player_acc_p99": 5.0,
    "ball_speed_p50": 5.1, "ball_speed_p99": 24.5,
    "players_p5": 14, "players_median": 22,
}
SHORT = {"ball_miss": "miss", "ball_miss_live": "miss_lv", "ball_miss_dead": "miss_dd",
         "gap_median_s": "gap_med", "gaps": "n_gaps", "player_speed_p50": "pl_v50",
         "player_speed_p99": "pl_v99", "player_acc_p99": "pl_a99", "ball_speed_p50": "b_v50",
         "ball_speed_p99": "b_v99", "players_p5": "n_pl5", "players_median": "n_pl50"}

# Everything off except the ball smoothing and the stoppage rebuild.
NO_NOISE = {"p_dead": 0.0, "q_alive": 0.0, "blip_per_min": 0.0, "player_sd": 0.0,
            "ball_sd": 0.0, "dropout_per_min": 0.0, "blackout_p": 0.0,
            "burst_per_min": 0.0}


def target_like_pool(seed=7, n=150):
    """Gap lengths (frames) with the target's aggregates: lognormal, median 10.7 s."""
    rng = np.random.default_rng(seed)
    s = np.exp(np.log(10.7) + 0.8 * rng.standard_normal(n))
    return np.round(np.clip(s, 0.3, 220.0) * FPS).astype(np.int64)


def load(mid):
    path = DFL / (mid + ".npz")
    if not path.is_file():
        pytest.skip("DFL open data not available: " + str(path))
    return dict(np.load(path))


@pytest.fixture(scope="module")
def pool():
    return target_like_pool()


@pytest.fixture(scope="module")
def match():
    return load(MATCH_IDS[0])


def measure(m, ball, home, away):
    """Aggregate statistics over graded frames, as measured on the target."""
    period, label = m["period"], m["label"]
    segs = _segments(period)
    g = period > 0
    miss = np.isnan(ball[:, 0])
    _, gap_len = dg.runs_in(miss & g, segs)
    xy = np.concatenate([home, away], 1)
    sp = speed(xy, segs)[g]
    v = sp[~np.isnan(sp)]
    acc = np.abs(np.diff(sp, axis=0)) * FPS
    acc = acc[~np.isnan(acc)]
    bs = speed(ball, segs)[g]
    bs = bs[~np.isnan(bs)]
    n_on = (~np.isnan(xy[g][:, :, 0])).sum(1)
    return {
        "ball_miss": miss[g].mean(),
        "ball_miss_live": miss[g & (label < 2)].mean(),
        "ball_miss_dead": miss[g & (label == 2)].mean(),
        "gap_median_s": np.median(gap_len) / FPS if len(gap_len) else 0.0,
        "gaps": len(gap_len),
        "player_speed_p50": np.median(v), "player_speed_p99": np.percentile(v, 99),
        "player_acc_p99": np.percentile(acc, 99),
        "ball_speed_p50": np.median(bs), "ball_speed_p99": np.percentile(bs, 99),
        "players_p5": np.percentile(n_on, 5), "players_median": np.median(n_on),
    }


# -- parameters ---------------------------------------------------------------------------------

def test_draw_params_overrides():
    base = dg.draw_params(np.random.default_rng(1))
    for key, (lo, hi) in dg.MATCH_RANGES.items():
        assert lo <= base[key] <= hi
    # a fixed override leaves every other draw unchanged; a range override rescales the draw
    fixed = dg.draw_params(np.random.default_rng(1), {"p_dead": 0.5, "q_alive": (0.0, 0.0)})
    assert fixed["p_dead"] == 0.5 and fixed["q_alive"] == 0.0
    for key in dg.MATCH_RANGES:
        if key not in ("p_dead", "q_alive"):
            assert fixed[key] == base[key]
    assert dg.draw_params(np.random.default_rng(1), {"roll_decel": 3.0})["roll_decel"] == 3.0
    # a single number fixes a per-event range setting; a range for a single-number one is refused
    fixed = dg.draw_params(np.random.default_rng(1), {"roll_s": 1.0, "blip_frames": 8})
    assert fixed["roll_s"] == (1.0, 1.0) and fixed["blip_frames"] == (8, 8)
    with pytest.raises(TypeError):
        dg.draw_params(np.random.default_rng(1), {"roll_decel": (3.0, 4.0)})
    with pytest.raises(KeyError):
        dg.draw_params(np.random.default_rng(1), {"no_such_parameter": 1})


# -- stoppage rebuild on a hand-made half --------------------------------------------------------

def test_stoppage_rebuild_physics():
    """10 m/s ball, a 20 s stoppage frozen DFL-style, the restart spot 5 m behind the stop."""
    T, a, e = 40 * FPS, 10 * FPS, 30 * FPS
    t = np.arange(T) / FPS
    ball = np.stack([-65.0 + 10.0 * np.minimum(t, a / FPS), np.zeros(T)], 1).astype(np.float32)
    ball[a:e] = ball[a - 1]                    # frozen where play stopped
    restart = ball[a - 1] + np.array([-5.0, 0.0], np.float32)
    ball[e:] = restart                         # the jump, then a still ball (kick-off like)
    label = np.zeros(T, np.int8)
    label[a:e] = 2
    period = np.ones(T, np.int8)
    rows = dg.stoppage_runs(ball, label, _segments(period), FPS, 45.0 / FPS)
    assert rows.tolist() == [[0, T, a, e, a, e]]
    cfg = {"roll_s": (1.0, 1.0), "walk_mps": (1.5, 1.5)}
    P = dg.draw_params(np.random.default_rng(0), cfg)
    out, rebuilt = dg.rebuild_stoppages(ball, rows, (105.0, 68.0), P, np.random.default_rng(0))
    assert rebuilt[a:e].all() and not rebuilt[:a].any() and not rebuilt[e:].any()
    v = np.diff(out[:, 0]) * FPS               # x velocity, m/s
    # rolls on with the incoming velocity, decelerating at 4 m/s^2 for 1 s
    assert v[a - 1] == pytest.approx(10.0 - 4.0 / FPS, abs=0.2)
    assert v[a + FPS - 2] == pytest.approx(6.0, abs=0.3)
    # then carried back at 1.5 m/s and waiting on the spot >= 0.5 s before the restart
    carry = v[a + FPS + 5:a + FPS + 30]
    assert np.allclose(carry, -1.5, atol=0.05)
    ready = int(P["ready_s"] * FPS)
    assert np.abs(out[e - ready:e] - restart).max() < 1e-4
    # no position jump anywhere
    assert np.abs(np.diff(out, axis=0)).max() < 0.5


def test_stoppage_rebuild_unreachable_is_linear():
    """A 2 s stoppage with the restart 30 m away cannot be walked: straight line, on time."""
    T, a, e = 20 * FPS, 5 * FPS, 7 * FPS
    ball = np.zeros((T, 2), np.float32)
    ball[e:, 0] = 30.0
    label = np.zeros(T, np.int8)
    label[a:e] = 2
    rows = dg.stoppage_runs(ball, label, [(0, T)], FPS, 45.0 / FPS)
    P = dg.draw_params(np.random.default_rng(0))
    out, _ = dg.rebuild_stoppages(ball, rows, (105.0, 68.0), P, np.random.default_rng(0))
    ready = int(P["ready_s"] * FPS)
    assert np.abs(out[e - ready:e, 0] - 30.0).max() < 1e-4
    step = np.diff(out[a - 1:e - ready, 0])   # up to the last frame before arrival
    assert step.min() > 0 and np.ptp(step) < 1e-3   # constant speed


def test_stoppage_runs_never_overlap():
    """A ball frozen between two close stoppages, and a late jump into the next stoppage, must
    not let one rebuilt stretch reach into the other."""
    T = 30 * FPS
    ball = np.zeros((T, 2), np.float32)       # frozen throughout
    label = np.zeros(T, np.int8)
    a1, e1 = 5 * FPS, 10 * FPS                 # stoppage 1
    a2, e2 = e1 + 10, 15 * FPS                 # stoppage 2, 0.4 s of frozen live play later
    a3, e3 = e2 + 1, 20 * FPS                  # stoppage 3 right after, DFL jump inside it
    label[a1:e1] = label[a2:e2] = label[a3:e3] = 2
    ball[e2 + 2:, 0] = 30.0                    # a jump 2 frames after stoppage 2 ends
    rows = dg.stoppage_runs(ball, label, [(0, T)], FPS, 45.0 / FPS)
    assert rows[:, 2].tolist() == [a1, a2, a3]
    assert (rows[1:, 4] >= rows[:-1, 5]).all()        # r of a run >= re of the one before
    assert (rows[:-1, 5] <= rows[1:, 2]).all()        # re of a run <= a of the next one


def test_blips_never_touch_the_real_ball():
    """False detections stay inside their gap with a missing frame on either side."""
    T, a, e = 60 * FPS, 20 * FPS, 23 * FPS
    ball = np.zeros((T, 2), np.float32)       # the real ball stays at the centre spot
    home = np.zeros((T, 11, 2), np.float32)
    home[:, :, 0] = 20.0 + np.arange(11)       # players far from it
    away = -home
    label = np.zeros(T, np.int8)
    label[a:e] = 2
    m = {"ball": ball, "home": home, "away": away, "label": label,
         "period": np.ones(T, np.int8), "pitch": np.array([105.0, 68.0], np.float32)}
    cfg = dict(NO_NOISE, p_dead=1.0, dead_gap_start_s=0.0, dead_gap_end_s=0.0,
               blip_per_min=100.0, ball_sigma=0.0, player_sigma=0.0)
    out, _, _ = dg.degrade(m, None, np.random.default_rng(0), cfg)
    dist = np.hypot(*out.T)
    real, blip = dist < 1.0, dist > 5.0        # NaN compares False to both
    assert blip.sum() > 30 and np.isnan(out[a]).all() and np.isnan(out[e - 1]).all()
    assert not (blip[1:] & real[:-1]).any() and not (blip[:-1] & real[1:]).any()


# -- degrade on DFL ------------------------------------------------------------------------------

def test_deterministic(match, pool):
    one = dg.degrade(match, pool, np.random.default_rng(3))
    two = dg.degrade(match, pool, np.random.default_rng(3))
    other = dg.degrade(match, pool, np.random.default_rng(4))
    for x, y, z in zip(one, two, other):
        assert np.array_equal(x, y, equal_nan=True)
        assert not np.array_equal(x, z, equal_nan=True)


def test_shapes_dtypes_and_input_untouched(match, pool):
    before = {k: v.copy() for k, v in match.items()}
    ball, home, away = dg.degrade(match, pool, np.random.default_rng(0))
    for k, v in before.items():
        assert np.array_equal(match[k], v, equal_nan=True), k   # label and tracking untouched
    for out, key in ((ball, "ball"), (home, "home"), (away, "away")):
        assert out.shape == match[key].shape and out.dtype == np.float32
    assert np.array_equal(np.isnan(ball[:, 0]), np.isnan(ball[:, 1]))


def test_switches_off(match, pool):
    ball, home, away = dg.degrade(match, pool, np.random.default_rng(0), NO_NOISE)
    assert not np.isnan(ball).any()
    assert np.array_equal(np.isnan(home), np.isnan(match["home"]))
    assert np.array_equal(np.isnan(away), np.isnan(match["away"]))


def test_nan_safe_with_ungraded_frames(match, pool):
    """Missing ball stretches, untracked players and ungraded padding do not break anything."""
    rng = np.random.default_rng(5)
    n, pad = 3 * 60 * FPS, 500
    m = {k: match[k][:n].copy() for k in ("home", "away", "ball", "label", "period")}
    m["pitch"] = match["pitch"]
    for start in rng.integers(0, n - 100, 20):
        m["ball"][start:start + rng.integers(1, 100)] = np.nan
    m["home"][:, 3] = np.nan                   # a player never on the pitch
    m["away"][1000:2000, 5] = np.nan
    for k, fill in (("ball", np.nan), ("home", np.nan), ("away", np.nan), ("label", 3), ("period", 0)):
        widths = [(pad, pad)] + [(0, 0)] * (m[k].ndim - 1)
        m[k] = np.pad(m[k], widths, constant_values=fill)
    ball, home, away = dg.degrade(m, pool, np.random.default_rng(0))
    for out in (ball, home, away):
        assert not np.isinf(out).any()
    assert np.isnan(ball[:pad]).all() and np.isnan(ball[-pad:]).all()
    assert np.isnan(home[:, 3]).all()
    assert not np.isnan(home[pad:-pad, 0]).all()


def test_blackouts_do_not_depend_on_the_label(match, pool):
    """Given a missing ball, a blackout is as likely in live play as in a stoppage: otherwise
    'players vanish with the ball' would be a label cue the target does not have."""
    cfg = {"dropout_per_min": 0.0, "blackout_p": 0.5}
    ball, home, away = dg.degrade(match, pool, np.random.default_rng(2), cfg)
    n_clean = (~np.isnan(np.concatenate([match["home"], match["away"]], 1)[:, :, 0])).sum(1)
    n_deg = (~np.isnan(np.concatenate([home, away], 1)[:, :, 0])).sum(1)
    out = n_deg < n_clean                      # only blackouts remove players here
    miss = np.isnan(ball[:, 0]) & (match["period"] > 0)
    p_live = out[miss & (match["label"] < 2)].mean()
    p_dead = out[miss & (match["label"] == 2)].mean()
    print("\nblackout share of missing-ball frames: live %.3f, dead %.3f" % (p_live, p_dead))
    assert 0.5 < p_live / p_dead < 2.0


def test_no_frozen_ball_or_jumps(match, pool):
    """DFL's frozen stoppage ball and restart jumps are gone; the ball is on the spot early."""
    rng = np.random.default_rng
    rebuilt, _, _ = dg.degrade(match, pool, rng(0), dict(NO_NOISE, ball_sigma=0.0))
    ball, _, _ = dg.degrade(match, pool, rng(0), NO_NOISE)          # rebuilt and filtered
    segs = _segments(match["period"])
    rows = dg.stoppage_runs(match["ball"], match["label"], segs, FPS, 45.0 / FPS)

    def steps(xy):                             # m/s moved into each frame
        return np.r_[0.0, np.hypot(*np.diff(xy, axis=0).T)] * FPS

    restarts = rows[(rows[:, 5] < rows[:, 1]) & (rows[:, 3] - rows[:, 2] > FPS), 5]
    print("\nball step into the restart frame, m/s: raw max %.0f, rebuilt %.1f, filtered %.1f"
          % (steps(match["ball"])[restarts].max(), steps(rebuilt)[restarts].max(),
             steps(ball)[restarts].max()))
    assert steps(match["ball"])[restarts].max() > 100   # the artefact is there in the raw data
    assert steps(rebuilt)[restarts].max() < 1e-3
    assert steps(ball)[restarts].max() < 30.0
    long_runs = rows[(rows[:, 3] - rows[:, 2] > 2 * FPS) & (rows[:, 5] < rows[:, 1])
                     & (rows[:, 4] > rows[:, 0] + 6)]
    r, re = long_runs[:, 4], long_runs[:, 5]
    # waiting on the restart spot for the last 0.5 s
    for lag in range(1, 13):
        assert np.abs(rebuilt[re - lag] - rebuilt[re]).max() < 1e-4
    # a ball that arrived fast keeps rolling: no longer frozen where play stopped
    incoming = np.hypot(*(match["ball"][r - 1] - match["ball"][r - 6]).T) * FPS / 5
    moved = np.hypot(*(ball[r + 7] - ball[r - 1]).T)       # over the first 8 frames
    fast = incoming > 3.0
    print("stoppages with an incoming ball > 3 m/s: %d, moved > 0.4 m in 0.3 s: %d"
          % (fast.sum(), (moved[fast] > 0.4).sum()))
    assert np.mean(moved[fast] > 0.4) > 0.8


def test_statistics_against_target(pool):
    rows, t_run = [], []
    for mid in MATCH_IDS:
        m = load(mid)
        rows.append((mid + " clean", measure(m, m["ball"], m["home"], m["away"])))
        for seed in (0, 1):
            t0 = time.perf_counter()
            ball, home, away = dg.degrade(m, pool, np.random.default_rng(seed))
            t_run.append(time.perf_counter() - t0)
            rows.append(("%s s%d" % (mid, seed), measure(m, ball, home, away)))
    keys = list(TARGET)
    print("\n%-13s" % "" + "".join("%8s" % SHORT[k] for k in keys))
    print("%-13s" % "TARGET" + "".join("%8.2f" % TARGET[k] for k in keys))
    for name, s in rows:
        print("%-13s" % name + "".join("%8.2f" % s[k] for k in keys))
    print("degrade runtime per match: %s s" % ", ".join("%.1f" % x for x in t_run))

    for name, s in rows:
        if name.endswith("clean"):
            continue
        assert 0.25 <= s["ball_miss"] <= 0.55, name
        assert 0.06 <= s["ball_miss_live"] <= 0.30, name
        assert 0.40 <= s["ball_miss_dead"] <= 0.95, name
        assert 4.0 <= s["gap_median_s"] <= 15.0, name
        assert 80 <= s["gaps"] <= 400, name
        assert 1.2 <= s["player_speed_p50"] <= 1.8, name
        assert 5.8 <= s["player_speed_p99"] <= 7.5, name
        assert s["player_acc_p99"] <= 10.0, name
        assert 2.0 <= s["ball_speed_p50"] <= 8.0, name
        assert 15.0 <= s["ball_speed_p99"] <= 32.0, name
        assert 8 <= s["players_p5"] <= 21, name
        assert 20 <= s["players_median"] <= 22, name
    assert max(t_run) < 20.0
