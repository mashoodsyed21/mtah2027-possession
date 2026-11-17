"""Tests for the post-processing in scripts/final_predict.py on synthetic labels (no match data)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import final_predict as fp  # noqa: E402
from possession import target  # noqa: E402

H, A, D, X = 0, 1, 2, 3


def lab_of(*spans):
    """Labels from (code, length) pairs."""
    return np.concatenate([np.full(n, c, np.int8) for c, n in spans])


def runs_of(lab):
    """(values, starts) of the constant runs."""
    starts = np.concatenate(([0], np.flatnonzero(np.diff(lab) != 0) + 1))
    return lab[starts], starts


# -- shift_team_edges ----------------------------------------------------------------------------

def test_shift_zero_is_a_copy():
    lab = lab_of((H, 10), (A, 10))
    out = fp.shift_team_edges(lab, 0)
    assert out is not lab and np.array_equal(out, lab)


@pytest.mark.parametrize("shift, edge", [(3, 7), (-3, 13), (1, 9), (-1, 11)])
def test_shift_moves_team_edge(shift, edge):
    out = fp.shift_team_edges(lab_of((H, 10), (A, 10)), shift)
    assert np.array_equal(out, lab_of((H, edge), (A, 20 - edge)))


def test_shift_leaves_dead_edges_alone():
    lab = lab_of((H, 10), (D, 10), (A, 10))
    for s in (-4, 4):
        assert np.array_equal(fp.shift_team_edges(lab, s), lab)


def test_shift_clipped_at_neighbours_and_array_ends():
    # first segment (2 frames) and last segment (2 frames) keep one frame each
    lab = lab_of((H, 2), (A, 10), (H, 2))
    assert np.array_equal(fp.shift_team_edges(lab, 5), lab_of((H, 1), (A, 6), (H, 7)))
    assert np.array_equal(fp.shift_team_edges(lab, -5), lab_of((H, 7), (A, 6), (H, 1)))
    # a dead neighbour is never overwritten
    lab = lab_of((D, 5), (H, 3), (A, 5))
    assert np.array_equal(fp.shift_team_edges(lab, 10), lab_of((D, 5), (H, 1), (A, 7)))


def test_shift_stays_inside_runs():
    # two abutting runs (DFL halves) plus an X gap: the H|A change at the run edge is not moved
    lab = lab_of((H, 10), (A, 10), (X, 5), (H, 10))
    runs = [(0, 10), (10, 20), (25, 35)]
    assert np.array_equal(fp.shift_team_edges(lab, 4, runs), lab)
    assert not np.array_equal(fp.shift_team_edges(lab, 4), lab)   # without runs it would be


def test_shift_random_properties():
    rng = np.random.default_rng(0)
    for _ in range(300):
        n_seg = int(rng.integers(1, 12))
        codes = [int(rng.integers(0, 3))]
        for _ in range(n_seg - 1):
            codes.append(int(rng.choice([c for c in (H, A, D) if c != codes[-1]])))
        lab = lab_of(*[(c, int(rng.integers(1, 15))) for c in codes])
        shift = int(rng.integers(-20, 21))
        out = fp.shift_team_edges(lab, shift)
        v0, b0 = runs_of(lab)
        v1, b1 = runs_of(out)
        assert np.array_equal(v0, v1)                      # same segments, none removed or added
        for i in range(1, len(b0)):
            moved = b0[i] - b1[i]
            if v0[i - 1] < D and v0[i] < D:                # team edge: towards the shift, at most |shift|
                assert 0 <= moved * np.sign(shift) <= abs(shift)
            else:                                          # dead edge: unchanged
                assert moved == 0


# -- forced pre-kick-off window and final labels -------------------------------------------------

def synthetic_period():
    period = np.zeros(target.N_FRAMES, np.int8)
    for half, a, b in target.HALVES:
        period[a:b + 1] = half
    return period


def test_prekick_window_bounds():
    # FAQ.md: second half graded from 110808, kick-off at 113742 -> dead frames 110808..113741
    assert fp.PREKICK == (110808, 113742)
    assert target.HALVES[1][1] == fp.PREKICK[0]


def test_force_prekickoff_only_touches_window():
    rng = np.random.default_rng(1)
    q = rng.dirichlet(np.ones(3), size=target.N_FRAMES)
    q = np.concatenate([q, rng.random((target.N_FRAMES, 1))], 1)
    before = q.copy()
    fp.force_prekickoff(q)
    a, b = fp.PREKICK
    assert np.all(q[a:b, :3].argmax(1) == D) and np.allclose(q[a:b, :3].sum(1), 1)
    assert np.array_equal(q[a:b, 3], before[a:b, 3])       # restarter column untouched
    assert np.array_equal(q[:a], before[:a]) and np.array_equal(q[b:], before[b:])


def test_final_labels():
    period = synthetic_period()
    lab = np.full(target.N_FRAMES, H, np.int8)
    lab[period == 0] = X
    out = fp.final_labels(lab, period)
    assert out is not lab and lab[fp.PREKICK[0]] == H      # input not modified
    assert np.all(out[110808:113742] == D)
    assert out[110807] == X and out[113742] == H
    assert np.all(out[period == 0] == X) and not np.any(out[period > 0] == X)
    assert out[11184] == X and out[11185] == H and out[90651] == H and out[90652] == X
    assert out[190197] == H and out[190198] == X
    assert np.sum(out != X) == (90651 - 11185 + 1) + (190197 - 110808 + 1)
