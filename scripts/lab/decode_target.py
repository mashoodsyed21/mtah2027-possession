"""Fuse target-only evidence into the frame model's probabilities, decode, write a submission.

    python scripts/lab/decode_target.py --tag v1_k2 [--broadcast] [--whistle] [--params '{...}']

The frame model is trained on DFL, which has no broadcast feed and no audio. Those two
signals exist only for the competition match, so they enter here, as naive-Bayes
log-likelihood ratios on the dead-ball probability:

    logit P(D | x, s) = logit P(D | x) + log P(s | D) / P(s | alive)

P(s | D) and P(s | alive) are estimated on the competition match itself from frames the
frame model is already confident about (P(D) > 0.9 or < 0.1): no ground truth is used,
so nothing is fitted to the public 5 minutes. The ratio is shrunk (x shrink) because the
signal and the frame model are not truly independent.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from possession import target  # noqa: E402
from possession.decode import smooth_logits, viterbi_hsmm_lite  # noqa: E402
from possession.features import _segments, roll_mean  # noqa: E402
from possession.fusion import llr_bins, llr_em  # noqa: E402
from possession.metrics import score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
KICKOFF_2ND = 113742          # FAQ.md: 2nd half graded from 110808 but kicks off here; all D before


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--broadcast", action="store_true")
    ap.add_argument("--whistle", action="store_true")
    ap.add_argument("--shrink", type=float, default=0.7)
    ap.add_argument("--em", action="store_true", help="estimate broadcast ratio by EM")
    ap.add_argument("--params", default=None, help="decoder params JSON; default: best for tag")
    ap.add_argument("--force-prekickoff", action="store_true", default=True)
    ap.add_argument("--feat-dir", default=None,
                    help="target feature cache written by build_features.py (only period / label are "
                         "read); default: the one the run was trained with")
    args = ap.parse_args()

    run = {}
    if not (args.params and args.feat_dir):
        with open(os.path.join(RESULTS, "experiments.jsonl")) as fh:
            run = [r for r in map(json.loads, fh) if r["tag"] == args.tag][-1]
    prm = json.loads(args.params) if args.params else run["decoder"]
    feat_dir = args.feat_dir or run["feat_dir"]

    p = np.load(os.path.join(ROOT, "extdata", "probs_%s.npy" % args.tag)).astype(np.float64)
    tf = np.load(os.path.join(target.DATA_DIR, "derived", "BAR-ATH_%s.npz" % feat_dir))
    period, gt = tf["period"], tf["label"]
    graded = period > 0
    segs = _segments(period)

    pd_ = np.clip(p[:, 2], 1e-4, 1 - 1e-4)
    logit = np.log(pd_ / (1 - pd_))
    notes = []
    if args.broadcast:
        from possession import broadcast
        bc = broadcast.build()
        inv = 1.0 - bc["bc_valid"].astype(np.float64)
        s = roll_mean(inv.astype(np.float32), 25, segs)
        if args.em:
            add, table = llr_em(s, logit, graded, np.array([0.05, 0.3, 0.7, 0.95]))
        else:
            add, table = llr_bins(s, pd_, graded, np.array([0.05, 0.3, 0.7, 0.95]))
        logit += args.shrink * add
        notes.append("broadcast LLR per bin %s" % np.round(table, 2).tolist())
    if args.whistle:
        w = np.load(os.path.join(target.DATA_DIR, "derived", "BAR-ATH_whistle.npz"))["recent"]
        add, table = llr_bins(w.astype(np.float32), pd_, graded, np.array([0.01, 0.5, 0.8]))
        logit += args.shrink * add
        notes.append("whistle LLR per bin %s" % np.round(table, 2).tolist())
    pd_new = 1 / (1 + np.exp(-logit))
    alive_share = 1 - pd_new
    ph = p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9)
    q = np.stack([alive_share * ph, alive_share * (1 - ph), pd_new], 1).astype(np.float32)
    if args.force_prekickoff:
        q[110808:KICKOFF_2ND] = [1e-6, 1e-6, 1 - 2e-6]

    logp = smooth_logits(q, temperature=prm.get("temperature", 1.0), window=prm.get("window", 25), runs=segs)
    dec = {k: v for k, v in prm.items() if k not in ("temperature", "window")}
    lab = viterbi_hsmm_lite(logp, runs=segs, params=dec)
    lab[~graded] = 3

    pub = score(gt, lab)
    sh = [float(np.mean(lab[graded] == c)) for c in range(3)]
    changes = int(np.sum(np.diff(lab[graded]) != 0))
    for n in notes:
        print(n)
    print("PUBLIC spellF1 %.4f macroF1 %.4f spells %d/%d | full: H %.3f A %.3f D %.3f changes %d" % (
        pub["spell_f1_mean"], pub["macro_f1"], pub["n_pred_spells"], pub["n_gt_spells"], *sh, changes))
    name = args.out or args.tag + ("_bc" if args.broadcast else "") + ("_wh" if args.whistle else "")
    csv = os.path.join(RESULTS, name + ".csv")
    with open(csv, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, "HADX"[c]) for i, c in enumerate(lab))
    with open(os.path.join(RESULTS, "decodes.jsonl"), "a") as fh:
        fh.write(json.dumps({"name": name, "tag": args.tag, "broadcast": args.broadcast, "whistle": args.whistle,
                             "shrink": args.shrink, "params": prm, "public_spellf1": pub["spell_f1_mean"],
                             "public_macrof1": pub["macro_f1"], "pred_spells": pub["n_pred_spells"],
                             "shares": sh, "changes": changes}) + "\n")


if __name__ == "__main__":
    main()
