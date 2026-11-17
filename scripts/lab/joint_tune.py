"""Tune the decoder of one run on external data at the competition's label definition.

CHALLENGE.md: the competition labels are derived mechanically from corrected tracking (dead-ball
flag plus the team the tracking names as owner), nothing smoothed. The external labels closest to
that are the RAW ones, not coarsened: DFL BallStatus / BallPossession per frame, and the Metrica
sample games' event-derived labels. So the objective is the mean of two scores, both at k = 0:
    DFL out-of-fold (7 matches)   and   Metrica (3 games x all degraded copies).
Nothing from the competition match enters this choice (no public window, no published anchors).

    python scripts/lab/joint_tune.py --tag v4_k05 [--trials 200]

Appends the winner to results/joint.jsonl (objective "raw"), where final_predict.py --joint reads it.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # scripts/: train_eval
from train_eval import RESULTS, ROOT, decode, gt_codes, runs_of_period, score  # noqa: E402

OBJECTIVE = "raw"


def space(rng, n):
    for _ in range(n):
        yield {
            "window": int(rng.choice([1, 5, 13, 25, 50])),
            "temperature": float(rng.choice([0.5, 0.75, 1.0, 1.5])),
            "cost_switch_team": float(rng.choice([1, 2, 4, 8, 16, 32])),
            "cost_to_dead": float(rng.choice([5, 10, 25, 50, 100])),
            "cost_from_dead": float(rng.choice([5, 10, 25, 50, 100])),
            "min_len_HA": int(rng.choice([1, 8, 13, 19, 25, 38, 50, 75])),
            "min_len_D": int(rng.choice([1, 13, 25, 50, 125])),
        }


def load_sets(tag, run):
    z = np.load(os.path.join(ROOT, "extdata", "oof_%s.npz" % tag))
    feat_dir = os.path.join(ROOT, "extdata", run["feat_dir"])
    dfl = []
    for mid in sorted({k[:-2] for k in z.files}):
        d = np.load(os.path.join(feat_dir, mid + "_s0.npz"))
        dfl.append((z[mid + "_p"], gt_codes(d["label"], d["period"]), runs_of_period(d["period"])))
    v = np.load(os.path.join(ROOT, "extdata", "valp_%s.npz" % tag))
    met = [(v[n + "_p"], gt_codes(v[n + "_label"], v[n + "_period"]), runs_of_period(v[n + "_period"]))
           for n in sorted({k.rsplit("_", 1)[0] for k in v.files})]
    return dfl, met


def evaluate(prm, dfl, met):
    parts = [float(np.mean([score(g, decode(p, r, prm))["spell_f1_mean"] for p, g, r in s]))
             for s in (dfl, met)]
    return float(np.mean(parts)), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--trials", type=int, default=200)
    args = ap.parse_args()
    runs = {json.loads(l)["tag"]: json.loads(l) for l in open(os.path.join(RESULTS, "experiments.jsonl"))}
    run = runs[args.tag]
    dfl, met = load_sets(args.tag, run)
    own = evaluate(run["decoder"], dfl, met)
    rng = np.random.default_rng(0)
    best = (own[0], run["decoder"], own[1])
    for prm in space(rng, args.trials):
        j, parts = evaluate(prm, dfl, met)
        if j > best[0]:
            best = (j, prm, parts)
    print("%s  [DFL k=0 | Metrica k=0]" % args.tag)
    print("  own decoder : J %.4f  %.4f %.4f" % (own[0], *own[1]))
    print("  joint tuned : J %.4f  %.4f %.4f" % (best[0], *best[2]))
    print("  params      : %s" % json.dumps(best[1]))
    with open(os.path.join(RESULTS, "joint.jsonl"), "a") as fh:
        fh.write(json.dumps({"tag": args.tag, "objective": OBJECTIVE, "J": best[0], "parts": best[2],
                             "decoder": best[1], "own_J": own[0], "own_parts": own[1]}) + "\n")


if __name__ == "__main__":
    main()
