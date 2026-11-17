"""Which label granularity should we train for? Score each run's out-of-fold DFL predictions
(decoded with that run's decoder) against DFL labels coarsened at several k.

    python scripts/lab/cross_k.py --tags v2c_noacc v2e_k3 v2d_k1

The competition's own granularity is only known roughly (public window: 4.2 spells per live
minute, between DFL coarsened at 2 s and 3 s), so we prefer the run that is best across the
plausible range, not at one k.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from possession.decode import smooth_logits, viterbi_hsmm_lite  # noqa: E402
from possession.degrade import relabel_min_spell  # noqa: E402
from possession.features import _segments  # noqa: E402
from possession.metrics import score  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # scripts/: train_eval
from train_eval import gt_codes  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--ks", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0, 3.5])
    args = ap.parse_args()
    runs = {json.loads(l)["tag"]: json.loads(l) for l in open(os.path.join(RESULTS, "experiments.jsonl"))}
    print("%-12s " % "run" + " ".join("k=%-5.1f" % k for k in args.ks) + "  mean")
    for tag in args.tags:
        run = runs[tag]
        prm = run["decoder"]
        z = np.load(os.path.join(ROOT, "extdata", "oof_%s.npz" % tag))
        feat_dir = os.path.join(ROOT, "extdata", run["feat_dir"])
        mids = sorted({k[:-2] for k in z.files})
        row = []
        for k in args.ks:
            s = []
            for mid in mids:
                d = np.load(os.path.join(feat_dir, mid + "_s0.npz"))
                g = gt_codes(relabel_min_spell(d["label"], d["period"], int(k * 25)), d["period"])
                runs_ = list(_segments(d["period"]))
                p = z[mid + "_p"]
                lp = smooth_logits(p[:, :3], temperature=prm["temperature"], window=prm["window"], runs=runs_)
                lab = viterbi_hsmm_lite(lp, runs_, {a: b for a, b in prm.items() if a not in ("window", "temperature")})
                s.append(score(g, lab)["spell_f1_mean"])
            row.append(np.mean(s))
        print("%-12s " % tag + " ".join("%.4f " % v for v in row) + " %.4f" % np.mean(row))


if __name__ == "__main__":
    main()
