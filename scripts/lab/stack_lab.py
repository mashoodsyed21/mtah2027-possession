"""Stage-2 stacking experiment on saved out-of-fold DFL probabilities (no stage-1 retraining).

    python scripts/lab/stack_lab.py [--tag v2b_ball] [--feat-dir features_v2] [--trials 60]

A second LightGBM stage (src/possession/stack.py) re-predicts P(dead) and P(home | alive) from the
stage-1 out-of-fold probabilities in temporal context plus a few raw tracking features. It is
validated with a nested leave-one-match-out: for each of the 7 DFL matches, stage 2 is trained on
the other 6 matches' stage-1 OOF predictions and applied to the held-out one.

Caveat: the stage-1 OOF predictions of the 6 training matches come from stage-1 models whose
training set contained the held-out match, so the nesting is not perfectly clean (re-running
stage 1 for every outer fold would be). The held-out match's own stage-2 input is clean.

Reports, per match and on average, the official spellF1 / macro-F1 / spell counts for
  stage1   the saved OOF probabilities (reproduces the train_eval number with its best decoder)
  stage2   the nested stage-2 probabilities
  blend    geometric 50/50 blend of stage1 and stage2 (H/A/D columns)
each with (a) the stage-1 best decoder, (b) the best of a shared random decoder search tuned on all
7 matches (in-sample, like train_eval), and (c) the same search tuned leave-one-match-out.
Aggregates only. Writes stage-2 OOF probabilities to extdata/oof_<tag>_stack2.npz and one summary
line to results/stack.jsonl.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from possession import stack  # noqa: E402
from possession.decode import smooth_logits, viterbi_hsmm_lite  # noqa: E402
from possession.features import _segments  # noqa: E402
from possession.metrics import score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))

# Best 3-state decoder of the v2b_ball run (results/experiments.jsonl).
BASE_DECODER = {"window": 50, "temperature": 0.5, "cost_switch_team": 5.0, "cost_to_dead": 50.0,
                "cost_from_dead": 5.0, "min_len_HA": 75, "min_len_D": 1}


def decode(p, runs, prm):
    """Same decoder as scripts/train_eval.py: smooth_logits, then viterbi_hsmm_lite."""
    logp = smooth_logits(p[:, :3], temperature=prm.get("temperature", 1.0),
                         window=prm.get("window", 25), runs=runs)
    dec = {k: v for k, v in prm.items() if k not in ("temperature", "window")}
    return viterbi_hsmm_lite(logp, runs=runs, params=dec)


def search_space(rng, n):
    """The decoder grid of scripts/train_eval.py (same parameter names and values)."""
    for _ in range(n):
        yield {
            "window": int(rng.choice([1, 13, 25, 50, 75])),
            "temperature": float(rng.choice([0.5, 1.0, 1.5, 2.0])),
            "cost_switch_team": float(rng.choice([2, 5, 10, 20, 40, 80])),
            "cost_to_dead": float(rng.choice([5, 10, 25, 50, 100])),
            "cost_from_dead": float(rng.choice([5, 10, 25, 50, 100])),
            "min_len_HA": int(rng.choice([1, 13, 25, 50, 75, 100])),
            "min_len_D": int(rng.choice([1, 25, 50, 125, 250])),
        }


def frame_stats(p, g):
    """Frame-level diagnostics: dead accuracy and log loss, team accuracy on live frames."""
    graded = g < 3
    pd_ = np.clip(p[graded, 2], 1e-6, 1 - 1e-6)
    yd = g[graded] == 2
    alive = g < 2
    return {"dead_acc": float(np.mean((pd_ > 0.5) == yd)),
            "dead_ll": float(-np.mean(np.where(yd, np.log(pd_), np.log(1 - pd_)))),
            "team_acc": float(np.mean((p[alive, 0] > p[alive, 1]) == (g[alive] == 0)))}


def scores(probs, seqs, prm):
    """Official headline numbers per match for one decoder setting."""
    return [score(s["g"], decode(probs[s["mid"]], s["runs"], prm)) for s in seqs]


def summarise(rows):
    f1 = [r["spell_f1_mean"] for r in rows]
    return {"spellf1": float(np.mean(f1)), "per_match": [float(x) for x in f1],
            "macro_f1": float(np.mean([r["macro_f1"] for r in rows])),
            "pred_spells": int(sum(r["n_pred_spells"] for r in rows)),
            "gt_spells": int(sum(r["n_gt_spells"] for r in rows))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v2b_ball", help="stage-1 run whose OOF predictions to stack")
    ap.add_argument("--feat-dir", default="features_v2")
    ap.add_argument("--step", type=int, default=4, help="use every step-th graded frame for training")
    ap.add_argument("--trials", type=int, default=60, help="random decoder-search trials")
    ap.add_argument("--team-target", choices=("raw", "relabelled"), default="raw")
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--leaves", type=int, default=31)
    ap.add_argument("--seed", type=int, default=0, help="LightGBM seed (bagging / feature sampling)")
    args = ap.parse_args()
    t0 = time.time()

    z = np.load(os.path.join(ROOT, "extdata", "oof_%s.npz" % args.tag))
    mids = sorted({k[:-2] for k in z.files})
    seqs = []
    for mid in mids:
        d = np.load(os.path.join(ROOT, "extdata", args.feat_dir, mid + "_s0.npz"))
        p1 = z[mid + "_p"].astype(np.float32)
        raw = {n: d[n] for n in stack.raw_feature_names(d.files)}
        names, X_, Xs = stack.match_matrices(p1, raw, d["period"])
        seqs.append({"mid": mid, "p1": p1, "g": z[mid + "_g"], "label": d["label"].astype(np.int8),
                     "runs": _segments(d["period"]), "X": X_, "Xs": Xs})
    print("%s: %d matches, %d stage-2 features (%d raw: %s)  [%.0fs]" % (
        args.tag, len(seqs), len(names), len(raw), " ".join(sorted(raw)), time.time() - t0))

    # -- nested leave-one-match-out stage 2 --------------------------------------------------------
    params = {"n_estimators": args.trees, "num_leaves": args.leaves, "seed": args.seed}
    p2 = {}
    for s in seqs:
        train = [m for m in seqs if m["mid"] != s["mid"]]
        models = stack.fit(train, step=args.step, team_target=args.team_target, params=params)
        p2[s["mid"]] = stack.predict(models, s, s["p1"])
        a, b = frame_stats(s["p1"], s["g"]), frame_stats(p2[s["mid"]], s["g"])
        print("  fold %s: dead acc %.3f -> %.3f  dead ll %.3f -> %.3f  team acc %.3f -> %.3f  [%.0fs]" % (
            s["mid"], a["dead_acc"], b["dead_acc"], a["dead_ll"], b["dead_ll"], a["team_acc"],
            b["team_acc"], time.time() - t0))
    suffix = "" if args.seed == 0 else "_seed%d" % args.seed
    out_npz = os.path.join(ROOT, "extdata", "oof_%s_stack2%s.npz" % (args.tag, suffix))
    np.savez(out_npz, **{m + "_p": v for m, v in p2.items()}, **{s["mid"] + "_g": s["g"] for s in seqs})

    inputs = {"stage1": {s["mid"]: s["p1"] for s in seqs}, "stage2": p2,
              "blend": {s["mid"]: stack.blend(s["p1"], p2[s["mid"]]) for s in seqs}}

    # -- decoding -----------------------------------------------------------------------------------
    trials = [dict(BASE_DECODER)] + list(search_space(np.random.default_rng(0), args.trials))
    grid = {k: [scores(probs, seqs, prm) for prm in trials] for k, probs in inputs.items()}
    print("decoded %d inputs x %d decoder settings  [%.0fs]" % (len(inputs), len(trials), time.time() - t0))

    res = {}
    for k in inputs:
        means = np.array([np.mean([r["spell_f1_mean"] for r in rows]) for rows in grid[k]])
        best = int(np.argmax(means))   # first maximum: ties go to the earlier trial (base first)
        nested = []                    # decoder chosen on the other 6 matches, scored on this one
        for i in range(len(seqs)):
            loo = [np.mean([r["spell_f1_mean"] for j, r in enumerate(rows) if j != i]) for rows in grid[k]]
            nested.append(grid[k][int(np.argmax(loo))][i])
        res[k] = {"base": summarise(grid[k][0]), "tuned": summarise(grid[k][best]),
                  "tuned_params": trials[best], "nested": summarise(nested)}

    print("\nmean DFL spellF1 / macroF1 / spells pred (gt %d)" % res["stage1"]["base"]["gt_spells"])
    print("%-8s %-7s %7s %7s %6s   per match (%s)" % ("input", "decoder", "spellF1", "macroF1", "pred",
                                                       " ".join(m[-3:] for m in mids)))
    for k in inputs:
        for dk in ("base", "tuned", "nested"):
            r = res[k][dk]
            print("%-8s %-7s %7.4f %7.4f %6d   %s" % (k, dk, r["spellf1"], r["macro_f1"], r["pred_spells"],
                                                     " ".join("%.3f" % x for x in r["per_match"])))
    for k in ("stage2", "blend"):
        print("  %s tuned decoder: %s" % (k, res[k]["tuned_params"]))

    print("\npaired differences vs stage1 (same decoder setting), per match:")
    for k in ("stage2", "blend"):
        for dk in ("base", "tuned", "nested"):
            diff = np.array(res[k][dk]["per_match"]) - np.array(res["stage1"][dk]["per_match"])
            print("  %-6s %-6s mean %+.4f  sd %.4f  improved %d/%d   %s" % (
                k, dk, diff.mean(), diff.std(ddof=1), int(np.sum(diff > 0)), len(diff),
                " ".join("%+.3f" % x for x in diff)))

    # Per-trial paired view: over all decoder settings, how often does stage 2 beat stage 1?
    m1 = np.array([np.mean([r["spell_f1_mean"] for r in rows]) for rows in grid["stage1"]])
    for k in ("stage2", "blend"):
        mk = np.array([np.mean([r["spell_f1_mean"] for r in rows]) for rows in grid[k]])
        print("  %s beats stage1 on %d/%d decoder settings (median gain %+.4f)" % (
            k, int(np.sum(mk > m1)), len(m1), float(np.median(mk - m1))))

    with open(os.path.join(RESULTS, "stack.jsonl"), "a") as fh:
        fh.write(json.dumps({"tag": args.tag, "feat_dir": args.feat_dir, "step": args.step,
                             "team_target": args.team_target, "trees": args.trees, "leaves": args.leaves,
                             "seed": args.seed,
                             "trials": args.trials, "n_features": len(names), "results": res}) + "\n")
    print("done (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
