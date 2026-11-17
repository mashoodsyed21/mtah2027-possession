"""Refit the full stage-1 models of an earlier run and save its Metrica probabilities
(../extdata/valp_<tag>.npz), which train_eval.py only saves for newer runs.

    python scripts/lab/metrica_backfill.py --tags v2d_k1 v3_k25

The refit uses today's training_rows (per-copy stride offset), so it is a close stand-in for the
run's own full model, not a byte copy. Prints aggregates only.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # scripts/: train_eval
from train_eval import (FPS, LGB, META, RESULTS, ROOT, Spec, decode, fit_all, gt_codes,  # noqa: E402
                        is_ball_feature, load, predict, runs_of_period, score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    runs = {json.loads(l)["tag"]: json.loads(l) for l in open(os.path.join(RESULTS, "experiments.jsonl"))}
    for tag in args.tags:
        run = runs[tag]
        feat_dir = os.path.join(ROOT, "extdata", run["feat_dir"])
        files = sorted(glob.glob(os.path.join(feat_dir, "*_s*.npz")))
        names = [n for n in load(files[0]) if n not in META
                 and not any(n.startswith(d) for d in run.get("drop", []))]
        dead_names = [n for n in names if not is_ball_feature(n)] if run.get("dead_noball") else names
        k = int(run["relabel"] * FPS)
        k_team = int(run["team_relabel"] * FPS)
        params = dict(LGB, n_estimators=run.get("trees", 400))
        models = fit_all(files, Spec(names, dead_names, k, k_team, 4), params)
        keep, s0 = {}, []
        for vp in sorted(glob.glob(os.path.join(feat_dir + "_val", "*_s*.npz"))):
            d = load(vp)
            name = os.path.basename(vp)[:-4]
            p = predict(models, {n: d[n] for n in names}, Spec(names, dead_names, k, k_team, 4))
            keep[name + "_p"], keep[name + "_label"], keep[name + "_period"] = p, d["label"], d["period"]
            if name.endswith("_s0"):
                dec = decode(p, runs_of_period(d["period"]), run["decoder"])
                s0.append(score(gt_codes(d["label"], d["period"]), dec)["spell_f1_mean"])
        np.savez(os.path.join(ROOT, "extdata", "valp_%s.npz" % tag), **keep)
        print("%s: Metrica raw labels (k=0), own decoder: %s  mean %.4f" % (
            tag, " ".join("%.3f" % v for v in s0), np.mean(s0)), flush=True)


if __name__ == "__main__":
    main()
