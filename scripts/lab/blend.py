"""Average the stage-1 probabilities of several finished runs into a new pseudo-run.

    python scripts/lab/blend.py --name b1 --tags v4_k05 v5_k05 [--weights 1 1]

Only runs whose features were computed on the SAME degraded DFL copies may be blended (otherwise
the out-of-fold average also averages over degradation noise and the DFL score is optimistic);
that is checked on a shared column. Writes ../extdata/oof_<name>.npz, valp_<name>.npz,
probs_<name>.npy and one line in results/experiments.jsonl (feat_dir and decoder of the first
tag), so joint_tune.py and final_predict.py work on it unchanged. Aggregates only.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # scripts/: train_eval
from train_eval import RESULTS, ROOT  # noqa: E402

CHECK_COL = "ball_gap_len_s"      # depends only on the degraded ball track


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--weights", type=float, nargs="*", default=None)
    args = ap.parse_args()
    w = np.array(args.weights or [1.0] * len(args.tags), np.float64)
    w /= w.sum()
    runs = {json.loads(l)["tag"]: json.loads(l) for l in open(os.path.join(RESULTS, "experiments.jsonl"))}
    first = runs[args.tags[0]]

    ref = None
    for t in args.tags:
        fd = os.path.join(ROOT, "extdata", runs[t]["feat_dir"])
        mid = sorted(f for f in os.listdir(fd) if f.endswith("_s0.npz"))[0]
        col = np.load(os.path.join(fd, mid))[CHECK_COL]
        if ref is not None and not np.array_equal(col, ref, equal_nan=True):
            raise SystemExit("%s: degraded inputs differ from %s, cannot blend" % (t, args.tags[0]))
        ref = col

    out = {}
    for kind, pattern in (("oof", "oof_%s.npz"), ("valp", "valp_%s.npz")):
        zs = [np.load(os.path.join(ROOT, "extdata", pattern % t)) for t in args.tags]
        res = {}
        for key in zs[0].files:
            if key.endswith("_p"):
                res[key] = sum(wi * z[key] for wi, z in zip(w, zs)).astype(np.float32)
            else:
                res[key] = zs[0][key]
        np.savez(os.path.join(ROOT, "extdata", pattern % args.name), **res)
        out[kind] = len(res)
    p = sum(wi * np.load(os.path.join(ROOT, "extdata", "probs_%s.npy" % t)) for wi, t in zip(w, args.tags))
    np.save(os.path.join(ROOT, "extdata", "probs_%s.npy" % args.name), p.astype(np.float32))
    with open(os.path.join(RESULTS, "experiments.jsonl"), "a") as fh:
        fh.write(json.dumps({"tag": args.name, "blend_of": args.tags, "weights": w.tolist(),
                             "feat_dir": first["feat_dir"], "relabel": first["relabel"],
                             "team_relabel": first["team_relabel"], "decoder": first["decoder"],
                             "dfl_spellf1": None}) + "\n")
    print("blend %s of %s written (%s)" % (args.name, args.tags, out))


if __name__ == "__main__":
    main()
