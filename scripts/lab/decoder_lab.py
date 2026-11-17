"""Decoder experiments on saved out-of-fold probabilities (no retraining).

    python scripts/lab/decoder_lab.py --tag v2a_noball [--trials 300]

Compares, on leave-one-match-out DFL predictions (relabelled like the training run):
  A  3-state decoder with the run's best params (reproduces train_eval)
  B  4-state restarter decoder (H, A, dead-home-restarts, dead-away-restarts)
  C  B + higher team-switch cost while the ball is not visible
  D  C + dead-edge shifts
Then applies the winner to the competition match (optionally with broadcast fusion) and
writes results/<tag>_<variant>.csv. Aggregates only.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from possession import target  # noqa: E402
from possession.decode import (decode_restarter, restarter_logp, smooth_logits,  # noqa: E402
                               viterbi_hsmm_lite)
from possession.features import _segments, roll_mean  # noqa: E402
from possession.metrics import score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))


def runs_of(period):
    return list(_segments(period))


def logp4(p, window, temperature, runs):
    """Smooth the 3-class probabilities, then split dead by the (smoothed) restarter."""
    lp3 = smooth_logits(p[:, :3], temperature=temperature, window=window, runs=runs)
    q = np.exp(lp3)
    pr = roll_mean(p[:, 3].astype(np.float32), max(window, 1), runs)
    pr = np.where(np.isnan(pr), 0.5, pr)
    ph = q[:, 0] / np.maximum(q[:, 0] + q[:, 1], 1e-9)
    return restarter_logp(q[:, 2], ph, pr)


def dec_restarter(p, runs, prm, vis=None):
    lp = logp4(p, prm.get("window", 25), prm.get("temperature", 1.0), runs)
    keys = ("cost_switch_team", "cost_to_dead", "cost_from_dead", "min_len_HA", "min_len_D",
            "cost_restarter", "start_shift", "end_shift")
    dp = {k: prm[k] for k in keys if k in prm}
    cm = None
    if vis is not None and prm.get("miss_switch", 0) > 0:
        cm = {"switch_team": (prm["miss_switch"] * (1.0 - vis)).astype(np.float32)}
    return decode_restarter(lp, runs, dp, cm)


def space(rng, n, base):
    for _ in range(n):
        prm = dict(base)
        prm.update({
            "window": int(rng.choice([13, 25, 50, 75])),
            "temperature": float(rng.choice([0.5, 1.0, 1.5])),
            "cost_switch_team": float(rng.choice([2, 5, 10, 20, 40])),
            "cost_to_dead": float(rng.choice([10, 25, 50, 100])),
            "cost_from_dead": float(rng.choice([5, 10, 25, 50])),
            "min_len_HA": int(rng.choice([13, 25, 50, 75])),
            "min_len_D": int(rng.choice([50, 125, 250])),
            "cost_restarter": float(rng.choice([5, 10, 30, 100])),
            "miss_switch": float(rng.choice([0, 0, 2, 5, 10, 20])),
            "start_shift": int(rng.choice([0, 0, 12, 25, -12])),
            "end_shift": int(rng.choice([0, 0, 12, 25, -12])),
        })
        yield prm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trials", type=int, default=250)
    args = ap.parse_args()
    exp = [json.loads(l) for l in open(os.path.join(RESULTS, "experiments.jsonl"))]
    run = [r for r in exp if r["tag"] == args.tag][-1]
    feat_dir = os.path.join(ROOT, "extdata", run["feat_dir"])
    z = np.load(os.path.join(ROOT, "extdata", "oof_%s.npz" % args.tag))
    mids = sorted({k[:-2] for k in z.files})
    seqs = []
    for mid in mids:
        d = np.load(os.path.join(feat_dir, mid + "_s0.npz"))
        r = runs_of(d["period"])
        vis = roll_mean(d["ball_vis"].astype(np.float32), 25, r)
        seqs.append((z[mid + "_p"], z[mid + "_g"], r, np.nan_to_num(vis)))

    def evaluate(fn):
        s = [score(g, fn(p, r, v))["spell_f1_mean"] for p, g, r, v in seqs]
        return float(np.mean(s)), s

    base = run["decoder"]
    a = evaluate(lambda p, r, v: viterbi_hsmm_lite(
        smooth_logits(p[:, :3], temperature=base["temperature"], window=base["window"], runs=r), r,
        {k: base[k] for k in base if k not in ("window", "temperature")}))
    print("A 3-state best-of-run: %.4f" % a[0])

    rng = np.random.default_rng(1)
    best = None
    for prm in space(rng, args.trials, base):
        m, s = evaluate(lambda p, r, v: dec_restarter(p, r, prm, v))
        if best is None or m > best[0]:
            best = (m, prm, s)
    print("B-D restarter search: %.4f  per match %s" % (best[0], " ".join("%.3f" % x for x in best[2])))
    print("  params %s" % best[1])

    # apply to the competition match
    tfp = np.load(os.path.join(ROOT, "extdata", "probs_%s.npy" % args.tag))
    tf = np.load(os.path.join(target.DATA_DIR, "derived", "BAR-ATH_%s.npz" % run["feat_dir"]))
    period, gt = tf["period"], tf["label"]
    r = runs_of(period)
    vis = np.nan_to_num(roll_mean(tf["ball_vis"].astype(np.float32), 25, r))
    q = tfp.copy()
    q[110808:113742, :3] = [1e-6, 1e-6, 1 - 2e-6]        # FAQ: graded but before the 2nd-half kick-off
    lab = dec_restarter(q, r, best[1], vis)
    lab[period == 0] = 3
    pub = score(gt, lab)
    g = period > 0
    print("PUBLIC spellF1 %.4f macroF1 %.4f spells %d/%d | full D share %.3f changes %d" % (
        pub["spell_f1_mean"], pub["macro_f1"], pub["n_pred_spells"], pub["n_gt_spells"],
        np.mean(lab[g] == 2), int(np.sum(np.diff(lab[g]) != 0))))
    name = args.tag + "_restarter"
    with open(os.path.join(RESULTS, name + ".csv"), "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, "HADX"[c]) for i, c in enumerate(lab))
    with open(os.path.join(RESULTS, "decodes.jsonl"), "a") as fh:
        fh.write(json.dumps({"name": name, "tag": args.tag, "dfl_3state": a[0], "dfl_restarter": best[0],
                             "params": best[1], "public_spellf1": pub["spell_f1_mean"],
                             "public_macrof1": pub["macro_f1"], "pred_spells": pub["n_pred_spells"]}) + "\n")


if __name__ == "__main__":
    main()
