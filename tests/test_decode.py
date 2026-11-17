"""Tests for possession.decode on synthetic data (no real match data anywhere)."""

import itertools
import time

import numpy as np
import pytest

from possession import decode as dc
from possession import metrics
from synthetic import FPS, H, A, D, X, make_match

PARAMS = dict(dc.DEFAULT_PARAMS)


def argmax_labels(probs, runs):
    out = np.full(len(probs), X, np.int8)
    for a, b in runs:
        out[a:b] = probs[a:b].argmax(axis=1)
    return out


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
def match():
    return make_match(seed=0)


@pytest.fixture(scope="module", autouse=True)
def warm_jit():
    """Compile the numba kernels once so timing tests measure the decoder, not the compiler."""
    lp = np.log(np.full((300, 3), 1 / 3))
    dc.viterbi_hsmm_lite(lp, [(0, 300)], PARAMS)
    dc.prune_short_segments(np.zeros(300, np.int8), [(0, 300)], 10, np.full((300, 3), 1 / 3))


# -- metrics wrapper -----------------------------------------------------------------------------

def test_metrics_wrapper_matches_official(match):
    gt, probs, runs = match
    assert metrics.valid_runs(gt) == runs
    perfect = metrics.score(gt, gt)
    assert perfect["spell_f1_mean"] == 1.0 and perfect["macro_f1"] == 1.0

    pred = dc.viterbi_hsmm_lite(np.log(probs), runs, PARAMS)
    mine = metrics.score(gt, pred)
    full = metrics.evaluate(gt, pred)
    for key in ("spell_f1_mean", "macro_f1", "n_gt_spells", "n_pred_spells"):
        assert mine[key] == full[key]


# -- Viterbi -------------------------------------------------------------------------------------

def test_decoding_beats_argmax(match):
    gt, probs, runs = match
    base = metrics.score(gt, argmax_labels(probs, runs))
    dec = metrics.score(gt, dc.viterbi_hsmm_lite(np.log(probs), runs, PARAMS))
    smoothed = dc.smooth_logits(probs, temperature=1.0, window=25, runs=runs)
    dec_s = metrics.score(gt, dc.viterbi_hsmm_lite(smoothed, runs, PARAMS))
    print("\nargmax   ", base)
    print("viterbi  ", dec)
    print("smooth+v ", dec_s)
    assert dec["spell_f1_mean"] > base["spell_f1_mean"] + 0.5
    assert dec["spell_f1_mean"] > 0.7
    assert dec["macro_f1"] > base["macro_f1"]
    # A short moving average before decoding removes most of the remaining over-segmentation.
    assert dec_s["spell_f1_mean"] > 0.85
    assert 0.8 < dec_s["n_pred_spells"] / dec_s["n_gt_spells"] < 1.25


def test_outside_runs_is_x(match):
    gt, probs, runs = match
    pred = dc.viterbi_hsmm_lite(np.log(probs), runs, PARAMS)
    assert np.array_equal(pred == X, gt == X)


@pytest.mark.parametrize("free_edges", [True, False])
def test_min_durations_respected(match, free_edges):
    gt, probs, runs = match
    params = dict(PARAMS, min_len_H=60, min_len_A=40, min_len_D=130, free_edges=free_edges,
                  cost_switch_team=2.0, cost_to_dead=2.0, cost_from_dead=2.0)  # cheap switches
    pred = dc.viterbi_hsmm_lite(np.log(probs), runs, params)
    mins = {H: 60, A: 40, D: 130}
    segs = segments(pred, runs)
    assert len(segs) > 100  # cheap switches really do produce many segments
    for s, e, c, edge in segs:
        if free_edges and edge:
            continue
        assert e - s >= mins[c], (s, e, c)


def test_determinism(match):
    gt, probs, runs = match
    lp = np.log(probs)
    first = dc.viterbi_hsmm_lite(lp, runs, PARAMS)
    second = dc.viterbi_hsmm_lite(lp.copy(), runs, dict(PARAMS))
    assert np.array_equal(first, second)


def test_runtime_160k_frames(match):
    gt, probs, runs = match
    lp = np.log(probs)
    assert sum(b - a for a, b in runs) >= 160_000
    t0 = time.perf_counter()
    dc.viterbi_hsmm_lite(lp, runs, PARAMS)
    elapsed = time.perf_counter() - t0
    # Also a single 160k-frame run with long minimum durations (more expanded states).
    one_run = [(runs[0][0], runs[0][0] + 160_000)]
    t0 = time.perf_counter()
    dc.viterbi_hsmm_lite(lp, one_run, dict(PARAMS, min_len_HA=100, min_len_D=250))
    elapsed_big = time.perf_counter() - t0
    print("\nviterbi 160k frames: {:.3f} s (k=50/50/125), {:.3f} s (k=100/100/250)".format(
        elapsed, elapsed_big))
    assert elapsed < 2.0 and elapsed_big < 2.0


def _brute_force_best(logp, params):
    """Exhaustive search over all label sequences of the exact model the kernel decodes."""
    k, trans, stay, exitc = dc._model_arrays(dc._resolve_params(params))
    n = len(logp)
    best = -np.inf
    for seq in itertools.product(range(3), repeat=n):
        seq = np.array(seq)
        bounds = np.concatenate(([0], np.flatnonzero(seq[1:] != seq[:-1]) + 1, [n]))
        ok, score = True, float(logp[np.arange(n), seq].sum())
        for i in range(len(bounds) - 1):
            length, c = bounds[i + 1] - bounds[i], seq[bounds[i]]
            first, last = i == 0, i == len(bounds) - 2
            edge_ok = params["free_edges"] and (first or last)
            if length < k[c] and not edge_ok:
                ok = False
                break
            score -= max(0, length - k[c]) * stay[c]
            if not last:
                score -= exitc[c] + trans[c, seq[bounds[i + 1]]]
        if ok:
            best = max(best, score)
    return best


def _path_score(labels, logp, params):
    k, trans, stay, exitc = dc._model_arrays(dc._resolve_params(params))
    n = len(labels)
    bounds = np.concatenate(([0], np.flatnonzero(labels[1:] != labels[:-1]) + 1, [n]))
    score = float(logp[np.arange(n), labels].sum())
    for i in range(len(bounds) - 1):
        c = labels[bounds[i]]
        score -= max(0, bounds[i + 1] - bounds[i] - k[c]) * stay[c]
        if i < len(bounds) - 2:
            score -= exitc[c] + trans[c, labels[bounds[i + 1]]]
    return score


@pytest.mark.parametrize("seed,free_edges,extra", [(1, True, None), (2, False, None),
                                                   (3, True, 3.0), (4, False, 2.0)])
def test_viterbi_is_exact_on_tiny_problems(seed, free_edges, extra):
    rng = np.random.default_rng(seed)
    logp = np.log(rng.dirichlet(np.ones(3) * 0.5, size=9))
    params = dict(min_len_H=2, min_len_A=3, min_len_D=1, cost_switch_team=0.7,
                  cost_to_dead=0.4, cost_from_dead=0.9, mean_extra_H=extra,
                  mean_extra_A=extra, mean_extra_D=extra, free_edges=free_edges,
                  logp_floor=-1e9)
    labels = dc.viterbi_hsmm_lite(logp, None, params)
    assert _path_score(labels, logp, params) == pytest.approx(_brute_force_best(logp, params))


# -- smoothing -----------------------------------------------------------------------------------

def test_moving_average_is_run_aware():
    p = np.zeros((20, 3))
    p[:10, H] = 1.0
    p[10:, A] = 1.0
    out = dc.moving_average(p, 5, runs=[(0, 10), (10, 20)])
    assert np.array_equal(out, p)                    # never mixes across the run boundary
    mixed = dc.moving_average(p, 5, runs=[(0, 20)])
    assert np.allclose(mixed.sum(axis=1), 1.0)
    assert mixed[9, H] == pytest.approx(0.6)          # frames 7..11: three H, two A


def test_temperature_scaling():
    p = np.array([[0.7, 0.2, 0.1]])
    sharp = dc.temperature_scale(p, 0.5)
    flat = dc.temperature_scale(p, 2.0)
    assert sharp[0, 0] > 0.7 > flat[0, 0]
    assert np.allclose(dc.temperature_scale(p, 1.0), p)
    lp = dc.smooth_logits(p, temperature=1.0)
    assert np.allclose(np.exp(lp), p, atol=1e-6)


# -- pruning -------------------------------------------------------------------------------------

def test_prune_absorbs_into_more_probable_neighbour():
    labels = np.array([H] * 10 + [D] * 3 + [A] * 10, np.int8)
    probs = np.full((23, 3), 1 / 3)
    probs[10:13] = [0.2, 0.5, 0.3]                    # the short D looks more like A
    out = dc.prune_short_segments(labels, None, min_len=5, probs=probs)
    assert out.tolist() == [H] * 10 + [A] * 13
    probs[10:13] = [0.5, 0.2, 0.3]
    out = dc.prune_short_segments(labels, None, min_len=5, probs=probs)
    assert out.tolist() == [H] * 13 + [A] * 10


def test_prune_flicker_and_per_class_minimum(match):
    gt, probs, runs = match
    noisy = argmax_labels(probs, runs)
    out = dc.prune_short_segments(noisy, runs, min_len=(50, 125), probs=probs)
    mins = {H: 50, A: 50, D: 125}
    for s, e, c, _ in segments(out, runs):
        assert e - s >= mins[c]
    assert np.array_equal(out == X, gt == X)
    pruned = metrics.score(gt, out)
    assert pruned["spell_f1_mean"] > metrics.score(gt, noisy)["spell_f1_mean"]
    print("\nargmax+prune", pruned)


# -- boundary refinement -------------------------------------------------------------------------

def test_refine_moves_to_evidence_peak_within_radius():
    labels = np.array([H] * 20 + [A] * 20 + [D] * 20, np.int8)
    ev = np.zeros(60)
    ev[23] = 5.0       # H->A boundary truly at 23 (within radius)
    ev[50] = 9.0       # A->D peak at 50 is outside radius 4 from 40 ...
    ev[37] = 1.0       # ... so the best in-window point (37) wins
    out = dc.refine_boundaries(labels, ev, radius=4)
    assert out.tolist() == [H] * 23 + [A] * 14 + [D] * 23
    team_only = dc.refine_boundaries(labels, ev, radius=4, kinds="team")
    assert team_only.tolist() == [H] * 23 + [A] * 17 + [D] * 20


def test_refine_keeps_segment_order():
    labels = np.array([H] * 5 + [A] * 2 + [H] * 5, np.int8)
    ev = np.zeros(12)
    ev[9] = 10.0       # would pull both boundaries past each other without the order guard
    out = dc.refine_boundaries(labels, ev, radius=6)
    bounds = np.flatnonzero(out[1:] != out[:-1]) + 1
    assert len(bounds) == 2 and bounds[0] < bounds[1]
    assert [out[0], out[bounds[0]], out[bounds[1]]] == [H, A, H]


# -- tuning --------------------------------------------------------------------------------------

def test_tune_decoder(match):
    gt, probs, runs = match
    grid = {"cost_switch_team": [5.0, 40.0], "min_len_HA": [10, 50]}
    best, table = dc.tune_decoder([(gt, np.log(probs), None)], grid, base_params=PARAMS)
    assert len(table) == 4
    assert table["spell_f1_mean"].is_monotonic_decreasing
    top = table.iloc[0]
    assert best["cost_switch_team"] == top["cost_switch_team"]
    assert best["min_len_HA"] == top["min_len_HA"] and isinstance(best["min_len_HA"], int)
    # The returned params really reproduce the table's best score.
    pred = dc.viterbi_hsmm_lite(np.log(probs), runs, best)
    assert metrics.score(gt, pred)["spell_f1_mean"] == pytest.approx(top["spell_f1_mean"])
