"""Final prediction for the competition match from a finished train_eval run.

    python scripts/final_predict.py --tag <run> [--seeds 0 1 2 3 4] [--shrink 0.5] [--out NAME]

Steps (every choice validated on DFL leave-one-match-out, none tuned on the public window):
  1. stage 2 (src/possession/stack.py): refits on the run's out-of-fold DFL probabilities, one
     model per seed, averaged; the nested DFL score is reported (stage 2 never sees the held-out
     match), then the same models are refitted on all 7 matches and applied to the target;
  2. broadcast evidence on the dead probability (naive Bayes, ratio estimated by EM on the target
     itself, x shrink);
  3. the graded frames before the second-half kick-off are dead (FAQ.md);
  4. decoding with the run's decoder;
  5. H <-> A boundaries moved by the median timing error measured on the DFL out-of-fold decode.
Writes the submission results/final_<tag>.csv (results/<out>.csv with --out) and appends to
results/final.jsonl. Aggregates only.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from possession import stack, target  # noqa: E402
from possession.decode import smooth_logits, viterbi_hsmm_lite  # noqa: E402
from possession.degrade import relabel_min_spell  # noqa: E402
from possession.features import _segments, roll_mean  # noqa: E402
from possession.fusion import llr_em  # noqa: E402
from possession.metrics import score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
# [first graded frame of the second half, its kick-off): graded but the ball is dead (FAQ.md)
PREKICK = (target.HALVES[1][1], 113742)
JOINT_KS = (0.0,)          # raw labels: the competition labels are not smoothed (CHALLENGE.md)


def force_prekickoff(q, eps=1e-6):
    """Make the pre-kick-off frames (almost) certainly dead in probabilities q [T, >=3], in place."""
    q[PREKICK[0]:PREKICK[1], :3] = [eps, eps, 1 - 2 * eps]
    return q


def final_labels(lab, period):
    """Submission codes: pre-kick-off frames D, frames outside the graded halves X."""
    out = np.asarray(lab, dtype=np.int8).copy()
    out[PREKICK[0]:PREKICK[1]] = stack.D
    out[period == 0] = stack.X
    return out


def decode(p, runs, prm):
    lp = smooth_logits(p[:, :3], temperature=prm["temperature"], window=prm["window"], runs=runs)
    return viterbi_hsmm_lite(lp, runs, {k: v for k, v in prm.items() if k not in ("window", "temperature")})


def multi_score(gs, lab):
    return float(np.mean([score(g, lab)["spell_f1_mean"] for g in gs]))


def gt_codes(label, period):
    """As train_eval.gt_codes: X outside halves and on the first frame of each later half."""
    g = label.astype(np.int8).copy()
    g[period == 0] = 3
    for a, b in list(_segments(period))[1:]:
        g[a] = 3
    return g


def team_boundaries(lab):
    b = np.flatnonzero(np.diff(lab) != 0) + 1
    return [x for x in b if lab[x - 1] < stack.D and lab[x] < stack.D]


def timing_error(gt, lab, tol=25):
    """Median signed offset (pred - true, frames) of matched H<->A boundaries."""
    lab = np.where(gt == stack.X, stack.X, lab)   # DFL halves abut: no boundary across their edge
    pb = np.array(team_boundaries(lab))
    errs = []
    for b in team_boundaries(gt):
        if len(pb):
            d = pb - b
            j = np.argmin(np.abs(d))
            if abs(d[j]) <= tol:
                errs.append(d[j])
    return float(np.median(errs)) if errs else 0.0, len(errs)


def shift_team_edges(lab, shift, runs=None):
    """Move every H <-> A boundary `shift` frames earlier (negative: later) inside each run
    (default: the whole array). A boundary never passes a neighbouring boundary or the edge of
    its run, so every segment keeps at least one frame; D and X edges stay where they are."""
    out = lab.copy()
    if shift == 0:
        return out
    for a, b in [(0, len(lab))] if runs is None else runs:
        seg = lab[a:b]
        bs = np.flatnonzero(np.diff(seg) != 0) + 1
        for i, e in enumerate(bs):
            if not (seg[e - 1] < stack.D and seg[e] < stack.D):
                continue
            lo = bs[i - 1] + 1 if i > 0 else 1
            hi = bs[i + 1] - 1 if i + 1 < len(bs) else len(seg) - 1
            ne = int(np.clip(e - shift, lo, hi))
            if ne < e:
                out[a + ne:a + e] = seg[e]
            elif ne > e:
                out[a + e:a + ne] = seg[e - 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="weight of the broadcast evidence; 0 = off. Its ratio would be estimated on "
                         "the competition match itself, not on external data, so it stays off")
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--joint", action="store_true",
                    help="decoder from scripts/lab/joint_tune.py (results/joint.jsonl); DFL scored on "
                         "the raw labels instead of the run's own k")
    args = ap.parse_args()
    t0 = time.time()
    run = [json.loads(l) for l in open(os.path.join(RESULTS, "experiments.jsonl"))
           if json.loads(l)["tag"] == args.tag][-1]
    prm = run["decoder"]
    if args.joint:
        prm = [json.loads(l) for l in open(os.path.join(RESULTS, "joint.jsonl"))
               if json.loads(l)["tag"] == args.tag and json.loads(l).get("objective") == "raw"][-1]["decoder"]
        print("joint decoder: %s" % prm)
    feat_dir = os.path.join(ROOT, "extdata", run["feat_dir"])
    z = np.load(os.path.join(ROOT, "extdata", "oof_%s.npz" % args.tag))
    mids = sorted({x[:-2] for x in z.files})

    # DFL matches as stage-2 inputs
    ms, dfl_names = {}, None
    for mid in mids:
        d = np.load(os.path.join(feat_dir, mid + "_s0.npz"))
        p1 = z[mid + "_p"]
        raw = {n: d[n] for n in stack.raw_feature_names(d.files)}
        names, X, Xs = stack.match_matrices(p1, raw, d["period"])
        if dfl_names is not None and names != dfl_names:
            raise ValueError("%s: stage-2 columns differ from the other DFL matches" % mid)
        dfl_names = names
        ms[mid] = dict(X=X, Xs=Xs, g=z[mid + "_g"], label=d["label"], p1=p1,
                       runs=list(_segments(d["period"])))
        # labels the post-model chain is scored (and the edge shift estimated) against
        ms[mid]["gs"] = ([gt_codes(relabel_min_spell(d["label"], d["period"], int(k * 25)), d["period"])
                          for k in JOINT_KS] if args.joint else [ms[mid]["g"]])

    # nested DFL evaluation of the whole post-model chain
    s1, s2, p2_oof = [], [], {}
    for mid in mids:
        m = ms[mid]
        if args.no_stack:
            p2 = m["p1"]
        else:
            tr = [ms[o] for o in mids if o != mid]
            p2 = np.mean([stack.predict(stack.fit(tr, params={"seed": s}), m, m["p1"])
                          for s in args.seeds], 0)
        p2_oof[mid] = p2
        s1.append(multi_score(m["gs"], decode(m["p1"], m["runs"], prm)))
        s2.append(multi_score(m["gs"], decode(p2, m["runs"], prm)))
    # timing correction estimated leave-one-match-out too
    s3, shifts = [], []
    for mid in mids:
        errs = [timing_error(ms[o]["gs"][0], decode(p2_oof[o], ms[o]["runs"], prm))[0] for o in mids if o != mid]
        sh = int(round(np.median(errs)))
        shifts.append(sh)
        lab = shift_team_edges(decode(p2_oof[mid], ms[mid]["runs"], prm), sh, ms[mid]["runs"])
        s3.append(multi_score(ms[mid]["gs"], lab))
    print("DFL nested: stage1 %.4f | +stage2 %.4f | +shift %.4f (shifts %s)  [%.0fs]" % (
        np.mean(s1), np.mean(s2), np.mean(s3), shifts, time.time() - t0))
    use_shift = np.mean(s3) > np.mean(s2)
    final_shift = int(round(np.median([timing_error(ms[o]["gs"][0], decode(p2_oof[o], ms[o]["runs"], prm))[0]
                                       for o in mids]))) if use_shift else 0

    # target
    tf = np.load(os.path.join(target.DATA_DIR, "derived", "BAR-ATH_%s.npz" % run["feat_dir"]))
    period, gt = tf["period"], tf["label"]
    runs = list(_segments(period))
    p1 = np.load(os.path.join(ROOT, "extdata", "probs_%s.npy" % args.tag))
    if args.no_stack:
        p2 = p1
    else:
        raw = {n: tf[n] for n in stack.raw_feature_names(tf.files)}
        names, X, Xs = stack.match_matrices(p1, raw, period)
        if names != dfl_names:   # same columns in the same order as the DFL training matrices
            raise ValueError("target stage-2 columns differ from DFL: %s" % sorted(set(names) ^ set(dfl_names)))
        mt = dict(X=X, Xs=Xs)
        allm = [ms[o] for o in mids]
        p2 = np.mean([stack.predict(stack.fit(allm, params={"seed": s}), mt, p1)
                      for s in args.seeds], 0)
    q = p2.astype(np.float64).copy()
    graded = period > 0
    if args.shrink > 0:
        from possession import broadcast
        bc = broadcast.build()
        sig = roll_mean((1.0 - bc["bc_valid"]).astype(np.float32), 25, runs)
        pd_ = np.clip(q[:, 2], 1e-4, 1 - 1e-4)
        logit = np.log(pd_ / (1 - pd_))
        add, table = llr_em(sig, logit, graded, np.array([0.05, 0.3, 0.7, 0.95]))
        pdn = 1 / (1 + np.exp(-(logit + args.shrink * add)))
        ph = q[:, 0] / np.maximum(q[:, 0] + q[:, 1], 1e-9)
        q[:, 0], q[:, 1], q[:, 2] = (1 - pdn) * ph, (1 - pdn) * (1 - ph), pdn
        print("broadcast LLR per bin %s" % np.round(table, 2).tolist())
    force_prekickoff(q)
    lab = final_labels(shift_team_edges(decode(q.astype(np.float32), runs, prm), final_shift, runs), period)
    pub = score(gt, lab)
    shares = [float(np.mean(lab[graded] == c)) for c in range(3)]
    changes = sum(int(np.count_nonzero(np.diff(lab[a:b]))) for a, b in runs)   # within each half
    print("PUBLIC spellF1 %.4f macroF1 %.4f spells %d/%d | full H %.3f A %.3f D %.3f changes %d | shift %d" % (
        pub["spell_f1_mean"], pub["macro_f1"], pub["n_pred_spells"], pub["n_gt_spells"], *shares, changes,
        final_shift))
    name = args.out or "final_" + args.tag
    with open(os.path.join(RESULTS, name + ".csv"), "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, "HADX"[c]) for i, c in enumerate(lab))
    with open(os.path.join(RESULTS, "final.jsonl"), "a") as fh:
        fh.write(json.dumps({"name": name, "tag": args.tag, "joint": args.joint, "decoder": prm, "stack_seeds": [] if args.no_stack else args.seeds,
                             "shrink": args.shrink, "dfl_stage1": float(np.mean(s1)),
                             "dfl_stage2": float(np.mean(s2)), "dfl_shift": float(np.mean(s3)),
                             "shift": final_shift, "public_spellf1": pub["spell_f1_mean"],
                             "public_macrof1": pub["macro_f1"], "pred_spells": pub["n_pred_spells"],
                             "shares": shares, "changes": changes}) + "\n")
    print("wrote results/%s.csv (%.0fs)" % (name, time.time() - t0))


if __name__ == "__main__":
    main()
