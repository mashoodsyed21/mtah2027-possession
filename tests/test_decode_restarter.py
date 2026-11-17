"""Tests for the restarter-aware decoder in possession.decode (synthetic data only)."""

import itertools
import time

import numpy as np
import pytest

from possession import decode as dc
from possession import metrics
from synthetic import FPS, H, A, D, X, make_half, make_match, noisy_probs

DH, DA = 2, 3
PARAMS = dict(dc.RESTARTER_PARAMS)


# -- synthetic data ------------------------------------------------------------------------------

def make_restart_match(seed, half_len=80_000, pad=2_000, gap=5_000,
                       mislead=(1 * FPS, 4 * FPS), restart_signal=1.0, live_restart="possessor"):
    """(gt, probs, p_restart_home, runs) for a two-half match where restarts are hard to read.

    Ground truth as in synthetic.make_match. The frame model's team evidence is misleading for the
    first 1-4 s of every possession that follows a stoppage (it points at the other team), while a
    restart model sees, noisily, which team will take the restart during the whole stoppage
    (logit +/- restart_signal plus unit Gaussian noise).

    live_restart sets what the restart model outputs while the ball is alive, where it has no
    restart to predict: "possessor" (default) points at the team the frame model sees on the ball,
    "random" at a random team per possession, both with the same confidence as during stoppages.
    "noise" gives pure noise around 0.5; it is NOT a fair test, because the restart model's
    confidence alone then tells dead frames from live ones -- label information the real model
    does not have.
    """
    rng = np.random.default_rng(seed)
    gt = np.concatenate([
        np.full(pad, X, np.int8), make_half(rng, half_len),
        np.full(gap, X, np.int8), make_half(rng, half_len),
        np.full(pad, X, np.int8),
    ])
    runs = [(pad, pad + half_len), (pad + half_len + gap, pad + 2 * half_len + gap)]
    seen = gt.copy()                               # the classes the frame model "sees"
    restarter = np.full(len(gt), -1, np.int8)
    for a, b in runs:
        w = gt[a:b]
        bounds = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)])) + a
        for i in range(len(bounds) - 2):
            s, e, next_end = bounds[i], bounds[i + 1], bounds[i + 2]
            if gt[s] == D:                         # make_half always follows D with a team
                team = gt[e]
                restarter[s:e] = team
                m = min(int(rng.integers(mislead[0], mislead[1] + 1)), next_end - e)
                seen[e:e + m] = 1 - team
    probs = noisy_probs(rng, seen)
    sign = np.where(restarter == H, 1.0, np.where(restarter == A, -1.0, 0.0))
    alive = (gt == H) | (gt == A)
    if live_restart == "possessor":
        sign[alive] = np.where(seen[alive] == H, 1.0, -1.0)
    elif live_restart == "random":
        seg_id = np.cumsum(np.concatenate(([0], gt[1:] != gt[:-1])))
        sign[alive] = rng.choice([-1.0, 1.0], seg_id[-1] + 1)[seg_id[alive]]
    elif live_restart != "noise":
        raise ValueError(live_restart)
    p_restart_home = 1.0 / (1.0 + np.exp(-(restart_signal * sign + rng.normal(0, 1, len(gt)))))
    return gt, probs, p_restart_home, runs


def logp_pair(probs, p_restart_home, runs, window=25, confidence_bonus=0.0):
    """Matching inputs for the two decoders from the same smoothed probabilities."""
    q = dc.moving_average(probs, window, runs)
    pr = dc.moving_average(p_restart_home, window, runs)
    logp3 = np.log(np.clip(q, 1e-6, None))
    logp4 = dc.restarter_logp(q[:, D], q[:, H] / (q[:, H] + q[:, A]), pr,
                              confidence_bonus=confidence_bonus)
    return logp3, logp4


def segments(labels, runs):
    """[(start, stop, code, touches_run_edge)] inside each run."""
    out = []
    for a, b in runs:
        w = labels[a:b]
        bounds = np.concatenate(([0], np.flatnonzero(w[1:] != w[:-1]) + 1, [len(w)]))
        for i in range(len(bounds) - 1):
            s, e = int(bounds[i]), int(bounds[i + 1])
            out.append((a + s, a + e, int(w[s]), s == 0 or e == len(w)))
    return out


@pytest.fixture(scope="module")
def restart_match():
    return make_restart_match(seed=0)


@pytest.fixture(scope="module")
def restart_logp(restart_match):
    gt, probs, pr, runs = restart_match
    return logp_pair(probs, pr, runs)


@pytest.fixture(scope="module", autouse=True)
def warm_jit():
    """Compile the numba kernels once so timing tests measure the decoder, not the compiler."""
    lp = np.log(np.full((300, 4), 0.25))
    dc.viterbi_restarter(lp, [(0, 300)], PARAMS)
    dc.viterbi_restarter(lp, [(0, 300)], dict(PARAMS, min_len_D=1))
    dc.viterbi_hsmm_lite(lp[:, :3], [(0, 300)], None)


# -- helper: restarter_logp ----------------------------------------------------------------------

def test_restarter_logp_columns():
    p_dead = np.array([0.1, 0.5, 0.9, np.nan])
    p_home = np.array([0.8, 0.5, 0.3, 0.6])
    p_rest = np.array([0.9, np.nan, 0.2, 0.5])
    lp = dc.restarter_logp(p_dead, p_home, p_rest)
    assert lp.shape == (4, 4) and lp.dtype == np.float32
    q = np.exp(lp.astype(np.float64))
    # The better restarter carries exactly the 3-class dead evidence: restart confidence never
    # makes a frame look more (or less) dead.
    assert np.allclose(q[:, H] + q[:, A] + np.maximum(q[:, DH], q[:, DA]), 1.0, atol=1e-5)
    assert q[0, H] == pytest.approx(0.9 * 0.8, rel=1e-5)
    assert q[0, DH] == pytest.approx(0.1, rel=1e-5)
    assert q[0, DA] == pytest.approx(0.1 * 0.1 / 0.9, rel=1e-5)
    assert q[2, DA] == pytest.approx(0.9, rel=1e-5)
    assert q[2, DH] == pytest.approx(0.9 * 0.25, rel=1e-5)
    assert q[1, DH] == pytest.approx(q[1, DA], rel=1e-6)        # NaN restart -> no information
    assert q[3, H] == pytest.approx(0.5 * 0.6, rel=1e-5)        # NaN dead -> 0.5
    none = dc.restarter_logp(p_dead, p_home)
    assert np.array_equal(none[:, DH], none[:, DA])
    assert np.allclose(np.exp(none[:, DH].astype(np.float64)), np.nan_to_num(p_dead, nan=0.5),
                       rtol=1e-5)
    # confidence_bonus=1: the calibrated likelihood ratio p / 0.5, a proper mixture over restarters.
    cal = np.exp(dc.restarter_logp(p_dead, p_home, p_rest, confidence_bonus=1.0).astype(np.float64))
    assert np.allclose(cal[:, H] + cal[:, A] + 0.5 * (cal[:, DH] + cal[:, DA]), 1.0, atol=1e-5)
    assert cal[0, DH] == pytest.approx(0.1 * 0.9 * 2, rel=1e-5)
    assert np.array_equal(dc.restarter_logp(p_dead, p_home, None, confidence_bonus=1.0), none)


def test_uninformative_restart_equals_hsmm_lite():
    gt, probs, runs = make_match(seed=0)
    lp = np.log(probs)
    # Exactly the 3-class model: both dead columns carry the dead evidence.
    lp4 = np.stack([lp[:, H], lp[:, A], lp[:, D], lp[:, D]], axis=1)
    for params in (dict(PARAMS), dict(PARAMS, min_len_D=1, mean_extra_HA=200.0, free_edges=False)):
        p3 = {k: v for k, v in params.items() if k != "cost_restarter"}
        assert np.array_equal(dc.viterbi_restarter(lp4, runs, params),
                              dc.viterbi_hsmm_lite(lp, runs, p3))
    # Same through the helper (float32, clipping): agreement up to rounding-level ties.
    via_helper = dc.restarter_logp(probs[:, D], probs[:, H] / (probs[:, H] + probs[:, A]))
    agree = dc.viterbi_restarter(via_helper, runs, PARAMS) == dc.viterbi_hsmm_lite(lp, runs, None)
    assert agree.mean() > 0.999


# -- decoder structure ---------------------------------------------------------------------------

def test_restart_team_consistency(restart_match, restart_logp):
    gt, _, _, runs = restart_match
    _, lp4 = restart_logp
    labels, states = dc.viterbi_restarter(lp4, runs, PARAMS, return_states=True)
    inside = labels != X
    assert np.array_equal(inside, gt != X)
    assert np.all(states[~inside] == -1)
    assert np.array_equal(labels[inside], np.minimum(states[inside], D))
    n_restarts = 0
    for a, b in runs:
        st = states[a:b]
        prev, nxt = st[:-1], st[1:]
        assert not np.any((prev == DH) & (nxt == A))
        assert not np.any((prev == DA) & (nxt == H))
        ends = np.flatnonzero((prev >= DH) & (nxt <= A))
        assert np.array_equal(nxt[ends], prev[ends] - 2)          # the restarter restarts
        n_restarts += len(ends)
    assert n_restarts > 50


@pytest.mark.parametrize("free_edges", [True, False])
def test_min_durations_and_restarter_changes(restart_match, free_edges):
    gt, probs, pr, runs = restart_match
    # Raw (unsmoothed) inputs and cheap switches: many segments and many restarter changes.
    _, lp4 = logp_pair(probs, pr, runs, window=1)
    params = dict(PARAMS, min_len_H=60, min_len_A=40, min_len_D=130, free_edges=free_edges,
                  cost_switch_team=2.0, cost_to_dead=2.0, cost_from_dead=2.0, cost_restarter=0.1)
    labels, states = dc.viterbi_restarter(lp4, runs, params, return_states=True)
    mins = {H: 60, A: 40, D: 130}
    segs = segments(labels, runs)
    assert len(segs) > 100
    for s, e, c, edge in segs:
        if free_edges and edge:
            continue
        assert e - s >= mins[c], (s, e, c)
    # Restarter changes inside a stoppage do not restart its minimum: there are stoppages whose
    # last restarter stretch is shorter than min_len_D although the stoppage itself is not.
    short_tail = 0
    for s, e, c, edge in segs:
        if c == D and not edge:
            st = states[s:e]
            changes = np.flatnonzero(st[1:] != st[:-1])
            if len(changes) and e - s - (changes[-1] + 1) < mins[D]:
                short_tail += 1
    assert short_tail > 0


def test_determinism(restart_match, restart_logp):
    gt, _, _, runs = restart_match
    _, lp4 = restart_logp
    rng = np.random.default_rng(5)
    mod = {"switch_team": rng.uniform(0, 20, len(gt)).astype(np.float32)}
    first = dc.viterbi_restarter(lp4, runs, PARAMS, mod, return_states=True)
    second = dc.viterbi_restarter(lp4.copy(), runs, dict(PARAMS),
                                  {"switch_team": mod["switch_team"].copy()}, return_states=True)
    assert np.array_equal(first[0], second[0]) and np.array_equal(first[1], second[1])


def test_runtime_160k_frames(restart_match, restart_logp):
    gt, _, _, runs = restart_match
    _, lp4 = restart_logp
    one_run = [(runs[0][0], runs[0][0] + 160_000)]
    times = []
    for params in (PARAMS, dict(PARAMS, min_len_HA=100, min_len_D=250)):
        t0 = time.perf_counter()
        dc.viterbi_restarter(lp4, one_run, params, {"switch_team": np.ones(len(gt))})
        times.append(time.perf_counter() - t0)
    print("\nrestarter 160k frames: {:.3f} s (k=50/50/125), {:.3f} s (k=100/100/250)".format(
        *times))
    assert max(times) < 2.0


# -- exactness against brute force ---------------------------------------------------------------

def _path_score(states, logp, params, mods):
    """Score of a 4-state path under the restarter model (-inf if the path is not allowed)."""
    p = dc._resolve_params(params, dc.RESTARTER_PARAMS)
    k, _, stay, exitc = dc._model_arrays(p)
    msw, mtd, mfd = mods
    n = len(states)
    score = float(logp[np.arange(n), states].sum())
    for t in range(1, n):
        a, b = states[t - 1], states[t]
        if a == b:
            continue
        if a >= DH and b >= DH:
            score -= p["cost_restarter"]
        elif a >= DH:
            if b != a - 2:
                return -np.inf                       # DH -> A or DA -> H
            score -= exitc[D] + p["cost_from_dead"] + mfd[t]
        elif b >= DH:
            score -= exitc[a] + p["cost_to_dead"] + mtd[t]
        else:
            score -= exitc[a] + p["cost_switch_team"] + msw[t]
    cls = np.minimum(states, D)                      # DH and DA form one stoppage segment
    bounds = np.concatenate(([0], np.flatnonzero(cls[1:] != cls[:-1]) + 1, [n]))
    nseg = len(bounds) - 1
    for i in range(nseg):
        length, c = bounds[i + 1] - bounds[i], cls[bounds[i]]
        if length < k[c] and not (p["free_edges"] and i in (0, nseg - 1)):
            return -np.inf
        score -= max(0, length - k[c]) * stay[c]
    return score


@pytest.mark.parametrize("seed,free_edges,min_d,extra", [(1, True, 2, None), (2, False, 3, None),
                                                         (3, True, 1, 3.0), (4, False, 1, 2.0)])
def test_viterbi_is_exact_on_tiny_problems(seed, free_edges, min_d, extra):
    rng = np.random.default_rng(seed)
    n = 7
    logp = np.log(rng.dirichlet(np.ones(4) * 0.5, size=n))
    mods = [rng.uniform(-0.3, 1.0, n) for _ in range(3)]
    params = dict(min_len_H=2, min_len_A=3, min_len_D=min_d, cost_switch_team=0.7,
                  cost_to_dead=0.4, cost_from_dead=0.9, cost_restarter=0.3,
                  mean_extra_H=extra, mean_extra_A=extra, mean_extra_D=extra,
                  free_edges=free_edges, logp_floor=-1e9)
    cost_mod = dict(zip(dc.COST_MOD_KEYS, mods))
    labels, states = dc.viterbi_restarter(logp, None, params, cost_mod, return_states=True)
    best = max(_path_score(np.array(seq), logp, params, mods)
               for seq in itertools.product(range(4), repeat=n))
    assert np.isfinite(best)
    assert _path_score(states, logp, params, mods) == pytest.approx(best)
    assert np.array_equal(labels, np.minimum(states, D))


# -- cost modifiers ------------------------------------------------------------------------------

def test_cost_mod(restart_match, restart_logp):
    gt, _, _, runs = restart_match
    _, lp4 = restart_logp
    (a0, b0), (a1, b1) = runs

    def team_switches(labels, a, b):
        w = labels[a:b]
        return int(np.sum(((w[:-1] == H) & (w[1:] == A)) | ((w[:-1] == A) & (w[1:] == H))))

    base = dc.viterbi_restarter(lp4, runs, PARAMS)
    assert team_switches(base, a0, b0) > 20 and team_switches(base, a1, b1) > 20
    block = np.zeros(len(gt), np.float32)
    block[a0:b0] = np.inf                            # forbid turnovers with the ball alive, half 1
    out = dc.viterbi_restarter(lp4, runs, PARAMS, {"switch_team": block})
    assert team_switches(out, a0, b0) == 0
    assert np.array_equal(out[a1:b1], base[a1:b1])   # the other half is untouched
    # A scalar modifier applies everywhere; NaN counts as zero.
    no_dead = dc.viterbi_restarter(lp4, runs, PARAMS, {"to_dead": np.inf})
    for a, b in runs:
        w = no_dead[a:b]
        assert not np.any((w[:-1] != D) & (w[1:] == D))
    nan_mod = np.full(len(gt), np.nan)
    assert np.array_equal(dc.viterbi_restarter(lp4, runs, PARAMS, {"from_dead": nan_mod}), base)


def test_input_validation():
    lp = np.zeros((10, 4))
    with pytest.raises(ValueError):
        dc.viterbi_restarter(np.zeros((10, 3)))
    with pytest.raises(KeyError):
        dc.viterbi_restarter(lp, cost_mod={"switch": np.zeros(10)})
    with pytest.raises(ValueError):
        dc.viterbi_restarter(lp, cost_mod={"switch_team": np.zeros(9)})
    with pytest.raises(ValueError):
        dc.viterbi_restarter(lp, cost_mod={"to_dead": -np.inf})
    with pytest.raises(KeyError):
        dc.viterbi_restarter(lp, params={"cost_nonsense": 1.0})


# -- the point of the exercise -------------------------------------------------------------------

@pytest.mark.parametrize("live_restart,seed", [("possessor", 0), ("possessor", 1),
                                               ("possessor", 2), ("random", 0)])
def test_restarter_beats_hsmm_lite_when_restarts_are_hard(live_restart, seed):
    gt, probs, pr, runs = make_restart_match(seed, live_restart=live_restart)
    lp3, lp4 = logp_pair(probs, pr, runs)
    hsmm_pred = dc.viterbi_hsmm_lite(lp3, runs, None)
    rest_pred = dc.viterbi_restarter(lp4, runs, PARAMS)
    hsmm, rest = metrics.score(gt, hsmm_pred), metrics.score(gt, rest_pred)
    fixed = metrics.score(gt, dc.viterbi_restarter(lp4, runs, dict(PARAMS, cost_restarter=100.0)))
    # Without restart information the restarter decoder is no better (it is the same model).
    blind = dc.restarter_logp(np.exp(lp3[:, D]), np.exp(lp3[:, H]) /
                              (np.exp(lp3[:, H]) + np.exp(lp3[:, A])))
    blind = metrics.score(gt, dc.viterbi_restarter(blind, runs, PARAMS))
    inside = gt != X
    d_share = [float(np.mean(x[inside] == D)) for x in (gt, hsmm_pred, rest_pred)]
    print("\n{} seed {}: spellF1 hsmm {:.4f}  restarter {:.4f}  (cost_restarter 100: {:.4f})  "
          "blind {:.4f}  D share gt/hsmm/restarter {:.3f}/{:.3f}/{:.3f}".format(
              live_restart, seed, hsmm["spell_f1_mean"], rest["spell_f1_mean"],
              fixed["spell_f1_mean"], blind["spell_f1_mean"], *d_share))
    assert rest["spell_f1_mean"] > hsmm["spell_f1_mean"]
    assert fixed["spell_f1_mean"] > hsmm["spell_f1_mean"] + 0.02
    assert abs(blind["spell_f1_mean"] - hsmm["spell_f1_mean"]) < 0.01
    assert rest["macro_f1"] >= hsmm["macro_f1"] - 0.005
    assert d_share[2] <= d_share[1] + 0.005             # restart evidence does not inflate D


def test_restart_confidence_is_not_dead_evidence(restart_match):
    """With a restart model as confident on live frames as on dead ones, the calibrated ratio
    (confidence_bonus=1) turns that confidence into dead evidence and over-predicts stoppages; the
    default form keeps the share of dead frames where the 3-class decoder has it."""
    gt, probs, pr, runs = restart_match
    lp3, lp4 = logp_pair(probs, pr, runs)
    _, lp4_cal = logp_pair(probs, pr, runs, confidence_bonus=1.0)
    inside = gt != X
    share = [float(np.mean(lab[inside] == D)) for lab in (
        dc.viterbi_hsmm_lite(lp3, runs, None), dc.viterbi_restarter(lp4, runs, PARAMS),
        dc.viterbi_restarter(lp4_cal, runs, PARAMS))]
    assert abs(share[1] - share[0]) < 0.01
    assert share[2] > share[0] + 0.03


# -- shift_dead_edges ----------------------------------------------------------------------------

def test_shift_dead_edges_basic():
    labels = np.array([H] * 20 + [D] * 10 + [A] * 20 + [H] * 10, np.int8)
    out = dc.shift_dead_edges(labels, None, start_shift=4, end_shift=2)
    assert out.tolist() == [H] * 16 + [D] * 12 + [A] * 22 + [H] * 10   # H<->A edge unchanged
    later = dc.shift_dead_edges(labels, None, start_shift=-3, end_shift=-5)
    assert later.tolist() == [H] * 23 + [D] * 12 + [A] * 15 + [H] * 10
    assert np.array_equal(dc.shift_dead_edges(labels, None, 0, 0), labels)


def test_shift_dead_edges_caps_at_half_length():
    labels = np.array([H] * 6 + [D] * 10 + [A] * 30, np.int8)
    out = dc.shift_dead_edges(labels, None, start_shift=5, end_shift=8)
    assert out.tolist() == [H] * 3 + [D] * 8 + [A] * 35      # H keeps 3 of 6, D keeps 5 of 10
    # Cut from both ends: the half-length allowance (4 of 8) is shared in proportion (2 + 2).
    labels = np.array([D] * 10 + [H] * 8 + [D] * 10, np.int8)
    out = dc.shift_dead_edges(labels, None, start_shift=6, end_shift=-6)
    assert out.tolist() == [D] * 12 + [H] * 4 + [D] * 12


def test_shift_dead_edges_is_run_aware(restart_match, restart_logp):
    gt, _, _, runs = restart_match
    _, lp4 = restart_logp
    labels = dc.viterbi_restarter(lp4, runs, PARAMS)
    out = dc.shift_dead_edges(labels, runs, start_shift=12, end_shift=6)
    assert np.array_equal(out == X, labels == X)
    for a, b in runs:                                  # run edges and segment order unchanged
        assert out[a] == labels[a] and out[b - 1] == labels[b - 1]
    seq = lambda lab: [c for _, _, c, _ in segments(lab, runs)]  # noqa: E731
    assert seq(out) == seq(labels)
    lens_in = np.array([e - s for s, e, _, _ in segments(labels, runs)])
    lens_out = np.array([e - s for s, e, _, _ in segments(out, runs)])
    assert np.all(2 * lens_out >= lens_in)
    assert np.sum(out == D) != np.sum(labels == D)    # it did move something


# -- tuning --------------------------------------------------------------------------------------

def test_tune_restarter(restart_match, restart_logp):
    gt, _, _, runs = restart_match
    _, lp4 = restart_logp
    grid = {"cost_restarter": [2.0, 50.0], "start_shift": [0, 10]}
    best, table = dc.tune_restarter([(gt, lp4, None), (gt, lp4, runs, {"switch_team": 1.0})],
                                    grid, base_params=PARAMS)
    assert len(table) == 4
    assert table["spell_f1_mean"].is_monotonic_decreasing
    top = table.iloc[0]
    assert best["cost_restarter"] == top["cost_restarter"]
    assert best["start_shift"] == top["start_shift"] and isinstance(best["start_shift"], int)
    # The returned params reproduce the table's best score through decode_restarter.
    scores = [metrics.score(gt, dc.decode_restarter(lp4, runs, best))["spell_f1_mean"],
              metrics.score(gt, dc.decode_restarter(lp4, runs, best,
                                                    {"switch_team": 1.0}))["spell_f1_mean"]]
    assert np.mean(scores) == pytest.approx(top["spell_f1_mean"])
