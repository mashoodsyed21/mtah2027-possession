"""Post-decoding refinement experiments on saved out-of-fold DFL predictions (no retraining).

    python scripts/lab/refine_lab.py [--tag v2b_ball] [--feat-dir features_v2] [--top 3]

The leave-one-match-out probabilities of a training run are decoded with that run's best
3-state decoder (smooth_logits + viterbi_hsmm_lite). On top of that decode the lab tests
post-processing aimed at boundary precision and at short possessions the decoder misses:

  a  snap boundaries to touch evidence of the team that takes over (touch / control / touch
     kernel features, or the raw frame probabilities), within +-radius frames; as a control,
     a constant shift of every H <-> A boundary, alone and before snapping
  b  gap rule: an H <-> A boundary inside a ball-missing stretch moves to the new team's first
     touch / control after the ball reappears (or to the start / middle / end of the gap)
  c  split long spells where P(opponent | ball alive) stays above tau for at least L frames
  d  shift stoppage edges by a constant (decode.shift_dead_edges)

then the cross product of the best few settings of each family, applied in the order
c -> b -> a -> d. Every variant is scored with the official spellF1 (metrics.score) per match.
Because the best of hundreds of settings is optimistic on seven matches, every table also gets
a nested leave-one-match-out estimate: pick the setting on six matches, score it on the
seventh.

DFL open data only (CC BY 4.0). Writes aggregates to results/refine_lab_dfl.csv.
"""
import os

# Share the machine with long training jobs: at most two threads for any numeric library.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "2")

import argparse  # noqa: E402
import itertools  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from possession.decode import (A, D, H, moving_average, shift_dead_edges,  # noqa: E402
                               smooth_logits, viterbi_hsmm_lite)
from possession.features import _segments  # noqa: E402
from possession.metrics import score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))

# Best decoder of run v2b_ball (3-state viterbi_hsmm_lite after smooth_logits).
DECODER = {"window": 50, "temperature": 0.5, "cost_switch_team": 5.0, "cost_to_dead": 50.0,
           "cost_from_dead": 5.0, "min_len_HA": 75, "min_len_D": 1}


# -- data ----------------------------------------------------------------------------------------

def _lag1(x):
    """x[t - 1] (0 at t = 0), NaN -> 0. Turns 'mean over the next 6 frames' into 'from t on'."""
    out = np.zeros(len(x), np.float64)
    out[1:] = np.nan_to_num(np.asarray(x, np.float64)[:-1])
    return out


def _gap_bounds(vis, runs):
    """For every frame: start and end (exclusive) of the ball-missing stretch it lies in.

    Visible frames get start = end = t. Stretches never cross a run edge.
    """
    n = len(vis)
    gs = np.arange(n)
    ge = np.arange(n)
    for a, b in runs:
        idx = np.arange(a, b)
        last = np.maximum.accumulate(np.where(vis[a:b], idx, a - 1))           # last visible <= t
        nxt = np.minimum.accumulate(np.where(vis[a:b], idx, b)[::-1])[::-1]     # next visible >= t
        miss = ~vis[a:b]
        gs[a:b] = np.where(miss, last + 1, idx)
        ge[a:b] = np.where(miss, nxt, idx)
    return gs, ge


class Match:
    """One DFL match: OOF probabilities, ground truth, runs and the boundary evidence used here."""

    def __init__(self, mid, p, g, feat):
        self.mid = mid
        self.p = np.asarray(p, np.float64)
        self.g = np.asarray(g).astype(np.int8)
        self.runs = [(int(a), int(b)) for a, b in _segments(feat["period"])]
        self.vis = np.asarray(feat["ball_vis"]) > 0.5
        self.gap_start, self.gap_end = _gap_bounds(self.vis, self.runs)
        n = len(self.p)

        def by_class(h, a):
            # Evidence that a segment of class c (H, A, D) starts at frame t. Touch-like signals
            # carry no evidence for D (column of zeros).
            return np.stack([np.nan_to_num(np.asarray(h, np.float64)),
                             np.nan_to_num(np.asarray(a, np.float64)), np.zeros(n)], 1)

        self.ev = {
            # a player within 1.5 m of the visible ball
            "touch": by_class(feat["touch_h"], feat["touch_a"]),
            # stricter control: within 1 m, ball slow, no opponent within 2.5 m (0 s since control)
            "ctrl": by_class(feat["h_t_since_ball_ctrl"] == 0, feat["a_t_since_ball_ctrl"] == 0),
            # soft touch kernel, mean over frames [t, t + 6)
            "kern": by_class(_lag1(feat["h_ball_tk_fut6"]), _lag1(feat["a_ball_tk_fut6"])),
            # the frame model's own (unsmoothed) log-probabilities
            "prob": np.log(np.clip(self.p[:, :3], 1e-6, None)),
        }
        self.base = viterbi_hsmm_lite(
            smooth_logits(self.p[:, :3], temperature=DECODER["temperature"],
                          window=DECODER["window"], runs=self.runs),
            self.runs, {k: v for k, v in DECODER.items() if k not in ("window", "temperature")})
        self._smooth = {}

    def smoothed(self, win):
        """Moving average of the 3-class probabilities (cached per window)."""
        if win not in self._smooth:
            self._smooth[win] = moving_average(self.p[:, :3], win, self.runs)
        return self._smooth[win]


def load(tag, feat_dir):
    z = np.load(os.path.join(ROOT, "extdata", "oof_%s.npz" % tag))
    out = []
    for mid in sorted({k[:-2] for k in z.files}):
        feat = np.load(os.path.join(ROOT, "extdata", feat_dir, mid + "_s0.npz"))
        out.append(Match(mid, z[mid + "_p"], z[mid + "_g"], feat))
    return out


# -- boundary moves ------------------------------------------------------------------------------

def move_boundaries(labels, runs, choose, min_seg=1):
    """Re-place boundaries one at a time, left to right, without changing segment order or labels.

    choose(t0, b, old, new, lo, hi) returns a new run-relative position or None (keep b). t0 is the
    run's first frame, b the boundary (first frame of the new segment), old / new the classes that
    meet. lo / hi confine the move between the already moved previous boundary and the original
    next one, keeping min_seg frames per segment -- the same rule as decode.refine_boundaries.
    """
    out = labels.copy()
    for start, stop in runs:
        lab = labels[start:stop]
        n = len(lab)
        bounds = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        if not len(bounds):
            continue
        seg_lab = lab[np.concatenate(([0], bounds))]
        new = bounds.copy()
        for i, b in enumerate(bounds):
            lo = (new[i - 1] if i else 0) + min_seg
            hi = (bounds[i + 1] if i + 1 < len(bounds) else n) - min_seg
            if lo > hi:
                continue
            pos = choose(start, int(b), int(lab[b - 1]), int(lab[b]), lo, hi)
            if pos is not None:
                new[i] = min(max(int(pos), lo), hi)
        out[start:stop] = np.repeat(seg_lab, np.diff(np.concatenate(([0], new, [n]))))
    return out


def _is_team(old, new):
    return {old, new} == {H, A}


def snap_boundaries(m, labels, signal, radius, mode, kinds):
    """(a) Move boundaries within +-radius frames to the evidence of the class that takes over.

    Like decode.refine_boundaries, which takes one evidence array for every boundary, but the
    evidence here depends on which classes meet (class-indexed [T, 3] arrays of Match.ev):
      snap  argmax of E[t, new]: the frame the new segment most plausibly starts at
      cp    change point: argmax over b of sum_{t < b} (E_old - E_new) + sum_{t >= b} (E_new - E_old)
            in the window, i.e. the argmin of the running sum of E_new - E_old
    kinds: "team" moves only H <-> A boundaries, "all" every boundary (with touch signals a stoppage
    has zero evidence, so 'cp' keeps a stoppage from starting before the old team's last touch and
    a restart from starting after the new team's first touch). Ties go to the closest position.
    """
    E = m.ev[signal]

    def choose(t0, b, old, new, lo, hi):
        if kinds == "team" and not _is_team(old, new):
            return None
        lo, hi = max(lo, b - radius), min(hi, b + radius)
        e = E[t0 + lo:t0 + hi + 1]
        if mode == "snap":
            s = e[:, new]
        else:
            d = e[:, new] - e[:, old]
            s = -np.concatenate(([0.0], np.cumsum(d)[:-1]))
        cand = np.flatnonzero(s == s.max()) + lo
        return cand[np.argmin(np.abs(cand - b))]

    return move_boundaries(labels, m.runs, choose)


def shift_team_edges(m, labels, shift):
    """(a, control) Move every H <-> A boundary earlier by `shift` frames (negative: later)."""
    def choose(t0, b, old, new, lo, hi):
        return b - shift if _is_team(old, new) else None
    return move_boundaries(labels, m.runs, choose)


def gap_rule(m, labels, target, min_gap, max_wait=50, fallback="end", kinds="team"):
    """(b) Re-place boundaries that fall inside a ball-missing stretch.

    A boundary b is inside a gap when the ball is missing at b - 1 or b. Targets:
      touch / ctrl  first frame in [gap end, gap end + max_wait] at which the new team touches /
                    controls the ball; if there is none, the gap end (fallback "end") or b ("keep")
      start / mid / end   the first missing frame, the middle, the first visible frame again
    Only gaps of at least min_gap frames count. kinds: "team" (H <-> A) or "all" (also restarts
    D -> H / A; boundaries into D are never moved).
    """
    vis = m.vis

    def choose(t0, b, old, new, lo, hi):
        if new == D or (kinds == "team" and not _is_team(old, new)):
            return None
        a = t0 + b
        g = a if not vis[a] else (a - 1 if not vis[a - 1] else -1)
        if g < 0:
            return None
        gs, ge = m.gap_start[g], m.gap_end[g]
        if ge - gs < min_gap:
            return None
        if target == "start":
            pos = gs
        elif target == "mid":
            pos = (gs + ge) // 2
        elif target == "end":
            pos = ge
        else:
            hit = np.flatnonzero(m.ev[target][ge:ge + max_wait + 1, new] > 0)
            if len(hit):
                pos = ge + hit[0]
            elif fallback == "end":
                pos = ge
            else:
                return None
        return pos - t0

    return move_boundaries(labels, m.runs, choose)


def split_spells(m, labels, tau, min_len, win, min_parent, edge, dmax):
    """(c) Insert an opponent spell inside a long H / A spell where the opponent clearly has the ball.

    q = P(opponent | ball alive) from the probabilities averaged over `win` frames. Inside every
    H / A segment of at least min_parent frames, each run of q > tau lasting min_len frames or more
    becomes the opponent's -- only in the segment's interior, `edge` frames away from both of its
    ends, and (dmax not None) only where the averaged P(dead) is below dmax.
    """
    q = m.smoothed(win)
    ph = q[:, 0] / np.maximum(q[:, 0] + q[:, 1], 1e-9)
    out = labels.copy()
    for start, stop in m.runs:
        lab = labels[start:stop]
        bounds = np.concatenate(([0], np.flatnonzero(lab[1:] != lab[:-1]) + 1, [len(lab)]))
        for s, e in zip(bounds[:-1], bounds[1:]):
            c = lab[s]
            if c not in (H, A) or e - s < min_parent:
                continue
            a, b = start + s + edge, start + e - edge
            if b - a < min_len:
                continue
            po = 1.0 - ph[a:b] if c == H else ph[a:b]
            mask = po > tau
            if dmax is not None:
                mask &= q[a:b, 2] < dmax
            edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
            for r0, r1 in zip(edges[::2], edges[1::2]):
                if r1 - r0 >= min_len:
                    out[a + r0:a + r1] = A if c == H else H
    return out


# -- pipeline and scoring ------------------------------------------------------------------------

def apply(m, cfg):
    """Run the post-processing steps named in cfg on the match's base decode: c -> b -> a -> d."""
    lab = m.base
    if cfg.get("c"):
        lab = split_spells(m, lab, **cfg["c"])
    if cfg.get("b"):
        lab = gap_rule(m, lab, **cfg["b"])
    if cfg.get("a"):
        a = dict(cfg["a"])
        if "team_shift" in a:                       # constant shift first, then (optional) snap
            lab = shift_team_edges(m, lab, a.pop("team_shift"))
        if a:
            lab = snap_boundaries(m, lab, **a)
    if cfg.get("d"):
        lab = shift_dead_edges(lab, m.runs, cfg["d"]["start_shift"], cfg["d"]["end_shift"])
    return lab


def evaluate(matches, cfg):
    spell, macro, npred = [], [], 0
    for m in matches:
        s = score(m.g, apply(m, cfg))
        spell.append(s["spell_f1_mean"])
        macro.append(s["macro_f1"])
        npred += s["n_pred_spells"]
    return {"spell": np.array(spell), "macro": float(np.mean(macro)), "n_pred": npred}


def nested_lomo(per_match):
    """Pick the best row on six matches, score it on the seventh; mean over held-out matches.

    per_match: [n_configs, n_matches]. Ties go to the earlier row (row 0 is the baseline).
    """
    per_match = np.asarray(per_match)
    held = []
    for j in range(per_match.shape[1]):
        rest = np.delete(per_match, j, axis=1).mean(1)
        held.append(per_match[int(np.argmax(rest)), j])
    return float(np.mean(held))


def team_boundary_error(m, labels, tol=50):
    """Signed error (pred - truth, frames) of predicted H <-> A boundaries matched to the nearest
    true boundary of the same direction within tol frames. Returns (matched share, median, MAE)."""
    def team_bounds(x):
        out = []
        for a, b in m.runs:
            lab = x[a:b]
            t = np.flatnonzero((lab[1:] != lab[:-1]) & (lab[1:] < 2) & (lab[:-1] < 2)) + 1
            out += [(a + i, int(lab[i])) for i in t]
        return out
    truth = team_bounds(m.g)
    errs, n = [], 0
    for direction in (H, A):
        tb = np.array([t for t, c in truth if c == direction])
        pb = [t for t, c in team_bounds(labels) if c == direction]
        n += len(pb)
        if not len(tb):
            continue
        for t in pb:
            k = np.argmin(np.abs(tb - t))
            if abs(tb[k] - t) <= tol:
                errs.append(t - tb[k])
    return errs, n


LENGTH_BINS = (0, 50, 75, 125, 250, 500, 10 ** 9)   # frames


def spell_recall_by_length(matches, preds, iou=0.5):
    """Share of true H / A spells per length bin that some predicted spell of the same team
    overlaps with temporal IoU >= iou. A diagnostic only; the ranking metric is metrics.score."""
    def spells(x, runs):
        out = []
        for a, b in runs:
            lab = x[a:b]
            bd = np.concatenate(([0], np.flatnonzero(lab[1:] != lab[:-1]) + 1, [len(lab)]))
            out += [(a + s, a + e, lab[s]) for s, e in zip(bd[:-1], bd[1:]) if lab[s] in (H, A)]
        return out
    hit = np.zeros(len(LENGTH_BINS) - 1)
    tot = np.zeros(len(LENGTH_BINS) - 1)
    for m, pred in zip(matches, preds):
        ps = spells(pred, m.runs)
        for s0, s1, c in spells(m.g, m.runs):
            k = np.searchsorted(LENGTH_BINS, s1 - s0, side="right") - 1
            tot[k] += 1
            hit[k] += any(c == pc and min(s1, p1) - max(s0, p0) >= iou * (max(s1, p1) - min(s0, p0))
                          for p0, p1, pc in ps if p1 > s0 and p0 < s1)
    return tot, hit / np.maximum(tot, 1)


# -- experiment grids ----------------------------------------------------------------------------

def grid_a():
    out = []
    for signal, mode, radius, kinds in itertools.product(
            ("touch", "ctrl", "kern", "prob"), ("snap", "cp"), (5, 10, 25, 50), ("team", "all")):
        if signal == "prob" and mode == "snap":
            continue  # the argmax of a probability is not a boundary
        out.append({"a": dict(signal=signal, mode=mode, radius=radius, kinds=kinds)})
    # control: a constant shift of every H <-> A boundary (the base decode is a few frames late)
    shifts = (-25, -15, -10, -5, 2, 3, 4, 5, 6, 8, 10, 15, 25)
    out += [{"a": {"team_shift": s}} for s in shifts]
    # shift, then snap with the least harmful snapping settings
    snaps = (dict(signal="ctrl", mode="cp", radius=5, kinds="all"),
             dict(signal="ctrl", mode="cp", radius=5, kinds="team"),
             dict(signal="touch", mode="snap", radius=5, kinds="team"))
    for s, snap in itertools.product((3, 4, 5), snaps):
        out.append({"a": dict(snap, team_shift=s)})
    return out


def grid_b():
    out = []
    for target, min_gap, kinds in itertools.product(
            ("touch", "ctrl", "start", "mid", "end"), (1, 25), ("team", "all")):
        if target in ("touch", "ctrl"):
            for max_wait, fallback in itertools.product((25, 75), ("end", "keep")):
                out.append({"b": dict(target=target, min_gap=min_gap, max_wait=max_wait,
                                      fallback=fallback, kinds=kinds)})
        else:
            out.append({"b": dict(target=target, min_gap=min_gap, kinds=kinds)})
    return out


def grid_c():
    return [{"c": dict(tau=tau, min_len=L, win=win, min_parent=mp, edge=edge, dmax=dmax)}
            for tau, L, win, mp, edge, dmax in itertools.product(
                (0.6, 0.7, 0.8, 0.9), (13, 25, 50, 75), (1, 25), (0, 250), (0, 25), (None, 0.5))]


def grid_d():
    return [{"d": dict(start_shift=s, end_shift=e)}
            for s, e in itertools.product(range(-25, 26, 5), repeat=2) if s or e]


def run_family(matches, name, cfgs, base, rows, log):
    """Score cfgs (baseline first), print the best few, return them ordered best first."""
    res = [(dict(), base)] + [(c, evaluate(matches, c)) for c in cfgs]
    table = np.array([r["spell"] for _, r in res])
    for c, r in res:
        rows.append({"family": name, "config": json.dumps(c, sort_keys=True),
                     "spell_f1_mean": float(r["spell"].mean()), "macro_f1": r["macro"],
                     "n_pred_spells": r["n_pred"],
                     **{m.mid: float(x) for m, x in zip(matches, r["spell"])}})
    # best spellF1, then macro-F1 (official tie-break), then the fewest post-processing steps
    order = sorted(range(1, len(res)), key=lambda i: (-res[i][1]["spell"].mean(),
                                                       -res[i][1]["macro"], len(res[i][0])))
    log("\n[%s] %d configs, nested-LOMO %.4f (baseline %.4f)"
        % (name, len(cfgs), nested_lomo(table), base["spell"].mean()))
    for i in order[:5]:
        c, r = res[i]
        d = r["spell"] - base["spell"]
        log("  %.4f (%+.4f) improved %d/%d  n_pred %d  %s"
            % (r["spell"].mean(), d.mean(), int((d > 0).sum()), len(d), r["n_pred"],
               json.dumps(c, sort_keys=True)))
    return [res[i][0] for i in order]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2b_ball")
    ap.add_argument("--feat-dir", default="features_v2")
    ap.add_argument("--top", type=int, default=3, help="settings per family in the combinations")
    args = ap.parse_args()

    lines = []

    def log(s):
        print(s, flush=True)
        lines.append(s)

    t0 = time.time()
    matches = load(args.tag, args.feat_dir)
    base = evaluate(matches, {})
    n_gt = sum(score(m.g, m.base)["n_gt_spells"] for m in matches)
    log("baseline spellF1 %.4f  per match %s  spells pred %d / true %d"
        % (base["spell"].mean(), " ".join("%.3f" % x for x in base["spell"]), base["n_pred"],
           n_gt))
    errs = sum((team_boundary_error(m, m.base)[0] for m in matches), [])
    log("baseline H<->A boundary error: matched %d, median %+.1f, MAE %.1f frames"
        % (len(errs), np.median(errs), np.mean(np.abs(errs))))
    tot, rec = spell_recall_by_length(matches, [m.base for m in matches])
    log("baseline recall@IoU0.5 by true spell length: " + "  ".join(
        "%d-%s: %d %.2f" % (LENGTH_BINS[i], LENGTH_BINS[i + 1] if i + 2 < len(LENGTH_BINS)
                            else "inf", tot[i], rec[i]) for i in range(len(tot))))

    rows = []
    best = {}
    for name, grid in (("a", grid_a()), ("b", grid_b()), ("c", grid_c()), ("d", grid_d())):
        best[name] = run_family(matches, name, grid, base, rows, log)

    # Combinations: the top settings of each family (or nothing), in the order c -> b -> a -> d.
    choices = [[None] + [c[f] for c in best[f][:args.top]] for f in "cbad"]
    combos = []
    for picks in itertools.product(*choices):
        cfg = {f: p for f, p in zip("cbad", picks) if p is not None}
        if cfg:
            combos.append(cfg)
    final = run_family(matches, "combo", combos, base, rows, log)

    winner = final[0]
    pred = [apply(m, winner) for m in matches]
    r = evaluate(matches, winner)
    errs = sum((team_boundary_error(m, p)[0] for m, p in zip(matches, pred)), [])
    log("\nbest %s\n  spellF1 %.4f (%+.4f)  per match %s  spells pred %d"
        % (json.dumps(winner, sort_keys=True), r["spell"].mean(),
           r["spell"].mean() - base["spell"].mean(),
           " ".join("%+.3f" % x for x in r["spell"] - base["spell"]), r["n_pred"]))
    log("  H<->A boundary error: matched %d, median %+.1f, MAE %.1f frames"
        % (len(errs), np.median(errs), np.mean(np.abs(errs))))
    log("done in %.0f s" % (time.time() - t0))

    import pandas as pd
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS, "refine_lab_dfl.csv"), index=False)


if __name__ == "__main__":
    main()
