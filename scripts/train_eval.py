"""Train the frame models on degraded DFL, validate leave-one-match-out, tune the decoder,
then predict the competition match and write a submission.

    python scripts/train_eval.py --tag final --feat-dir features_v3 --relabel 2.5 --team-relabel 0 --drop ball_acc

Three LightGBM models, all team-symmetric where it matters:
  dead     P(ball dead)                       on all graded frames
  team     P(home | alive)                    on live frames, trained on both team orientations
  restart  P(home restarts | dead)            on dead frames: the team owning the next live spell

Prints aggregates only. Saves out-of-fold and target probabilities as ../extdata/oof_<tag>.npz and
../extdata/probs_<tag>.npy (never committed) for scripts/final_predict.py, writes the stage-1
decode to results/<tag>.csv and appends one line per run to results/experiments.jsonl.
"""
import argparse
import glob
import json
import os
import sys
import time

import lightgbm as lgb
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from possession import FEAT_DIR, LGB_THREADS, target  # noqa: E402
from possession.decode import smooth_logits, viterbi_hsmm_lite  # noqa: E402
from possession.degrade import relabel_min_spell  # noqa: E402
from possession.features import FPS, _segments, swap_teams  # noqa: E402
from possession.metrics import score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "results"))
META = ("period", "label")
BALL_TOKENS = ("ball", "held", "touch", "near_home", "margin_h", "dball", "nball", "anchor")

LGB = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_child_samples=200,
           feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
           n_estimators=400, deterministic=True, force_col_wise=True, seed=0,
           num_threads=LGB_THREADS, verbose=-1)


def is_ball_feature(name):
    return any(t in name for t in BALL_TOKENS)


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def matrix(f, names, idx=None):
    cols = [f[n] if idx is None else f[n][idx] for n in names]
    return np.stack(cols, 1).astype(np.float32)


def next_alive_team(lab, period):
    """For every frame, the team of the next live frame in the same half (-1 if none)."""
    out = np.full(len(lab), -1, np.int8)
    for a, b in _segments(period):
        seg = lab[a:b]
        nxt = -1
        res = np.full(b - a, -1, np.int8)
        for i in range(b - a - 1, -1, -1):     # one pass per half; cheap enough (numpy-free loop ~0.1 s)
            if seg[i] < 2:
                nxt = seg[i]
            res[i] = nxt
        out[a:b] = res
    return out


class Spec:
    def __init__(self, names, dead_names, k_dead, k_team, step):
        self.names, self.dead_names = names, dead_names
        self.k_dead, self.k_team, self.step = k_dead, k_team, step


def training_rows(files, spec):
    D, T, R = [[], []], [[], []], [[], []]
    for i, path in enumerate(files):
        d = load(path)
        lab_d = relabel_min_spell(d["label"], d["period"], spec.k_dead)
        lab_t = relabel_min_spell(d["label"], d["period"], spec.k_team)
        idx = np.flatnonzero((d["period"] > 0) & (d["label"] < 3))
        idx = idx[(idx + i) % spec.step == 0]     # each degraded copy samples different frames
        f = {n: d[n] for n in spec.names}
        g = swap_teams(f)
        D[0].append(matrix(f, spec.dead_names, idx))
        D[1].append((lab_d[idx] == 2).astype(np.int8))
        alive = idx[lab_t[idx] < 2]
        T[0] += [matrix(f, spec.names, alive), matrix(g, spec.names, alive)]
        T[1] += [(lab_t[alive] == 0).astype(np.int8), (lab_t[alive] == 1).astype(np.int8)]
        nxt = next_alive_team(lab_d, d["period"])
        dead = idx[(lab_d[idx] == 2) & (nxt[idx] >= 0)]
        R[0] += [matrix(f, spec.names, dead), matrix(g, spec.names, dead)]
        R[1] += [(nxt[dead] == 0).astype(np.int8), (nxt[dead] == 1).astype(np.int8)]
    return [(np.concatenate(x), np.concatenate(y)) for x, y in (D, T, R)]


def fit_all(files, spec, params):
    return [lgb.LGBMClassifier(**params).fit(X, y) for X, y in training_rows(files, spec)]


def predict(models, f, spec):
    md, mt, mr = models
    g = swap_teams(f)
    # dead ball does not depend on which team is which: average both orientations
    pd_ = 0.5 * (md.predict_proba(matrix(f, spec.dead_names))[:, 1]
                 + md.predict_proba(matrix(g, spec.dead_names))[:, 1])
    X, Xs = matrix(f, spec.names), matrix(g, spec.names)
    ph = 0.5 * (mt.predict_proba(X)[:, 1] + 1.0 - mt.predict_proba(Xs)[:, 1])
    pr = 0.5 * (mr.predict_proba(X)[:, 1] + 1.0 - mr.predict_proba(Xs)[:, 1])
    return np.stack([(1 - pd_) * ph, (1 - pd_) * (1 - ph), pd_, pr], 1).astype(np.float32)


def runs_of_period(period):
    return [(a, b) for a, b in _segments(period)]


def gt_codes(label, period):
    """Frames outside halves are X; the first frame of each later half too, so the official
    scorer sees separate runs (DFL halves are stored back to back)."""
    g = label.astype(np.int8).copy()
    g[period == 0] = 3
    for a, b in runs_of_period(period)[1:]:
        g[a] = 3
    return g


def decode(p, runs, prm):
    logp = smooth_logits(p[:, :3], temperature=prm.get("temperature", 1.0), window=prm.get("window", 25), runs=runs)
    dec = {k: v for k, v in prm.items() if k not in ("temperature", "window")}
    return viterbi_hsmm_lite(logp, runs=runs, params=dec)


def search_space(rng, n):
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


def tune(oof, trials, seed=0):
    rng = np.random.default_rng(seed)
    best, rows = None, []
    for prm in search_space(rng, trials):
        s = [score(g, decode(p, r, prm))["spell_f1_mean"] for p, g, r in oof.values()]
        rows.append((float(np.mean(s)), prm, s))
        if best is None or rows[-1][0] > best[0]:
            best = rows[-1]
    rows.sort(key=lambda r: -r[0])
    return best, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--feat-dir", default=FEAT_DIR)
    ap.add_argument("--relabel", type=float, default=2.0, help="min spell s for dead model + decoder target")
    ap.add_argument("--team-relabel", type=float, default=None, help="min spell s for team labels")
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--trials", type=int, default=150)
    ap.add_argument("--drop", nargs="*", default=[])
    ap.add_argument("--trees", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--leaves", type=int, default=63)
    ap.add_argument("--dead-noball", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=None, help="degradation seeds used for training")
    args = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    k = int(args.relabel * FPS)
    k_team = k if args.team_relabel is None else int(args.team_relabel * FPS)
    params = dict(LGB, n_estimators=args.trees, learning_rate=args.lr, num_leaves=args.leaves)
    feat_dir = os.path.join(ROOT, "extdata", args.feat_dir)

    files = sorted(glob.glob(os.path.join(feat_dir, "*_s*.npz")))
    if args.seeds is not None:
        files = [p for p in files if int(p.rsplit("_s", 1)[1][:-4]) in args.seeds]
    mids = sorted({os.path.basename(p).rsplit("_s", 1)[0] for p in files})
    names = [n for n in load(files[0]) if n not in META and not any(n.startswith(d) for d in args.drop)]
    dead_names = [n for n in names if not is_ball_feature(n)] if args.dead_noball else names
    spec = Spec(names, dead_names, k, k_team, args.step)
    print("%s: %d matches, %d files, %d features (dead model %d), relabel %.1fs team %.1fs" % (
        args.tag, len(mids), len(files), len(names), len(dead_names), args.relabel, k_team / FPS))

    t0 = time.time()
    oof = {}
    for mid in mids:
        tr = [p for p in files if not os.path.basename(p).startswith(mid)]
        models = fit_all(tr, spec, params)
        d = load(os.path.join(feat_dir, mid + "_s0.npz"))
        p = predict(models, {n: d[n] for n in names}, spec)
        g = gt_codes(relabel_min_spell(d["label"], d["period"], k), d["period"])
        oof[mid] = (p, g, runs_of_period(d["period"]))
        alive = g < 2
        print("  fold %s: dead acc %.3f  team acc %.3f  [%.0fs]" % (
            mid, np.mean((p[:, 2] > 0.5) == (g == 2)), np.mean((p[alive, 0] > p[alive, 1]) == (g[alive] == 0)),
            time.time() - t0))
    np.savez(os.path.join(ROOT, "extdata", "oof_%s.npz" % args.tag),
             **{mid + "_p": v[0] for mid, v in oof.items()}, **{mid + "_g": v[1] for mid, v in oof.items()})

    best, rows = tune(oof, args.trials)
    print("decoder search: best DFL spellF1 %.4f  (per match %s)" % (best[0], " ".join("%.3f" % x for x in best[2])))
    print("  best params: %s" % best[1])
    full = [score(g, decode(p, r, best[1])) for p, g, r in oof.values()]
    print("  DFL macroF1 %.4f  pred/gt spells %d/%d" % (
        np.mean([x["macro_f1"] for x in full]), sum(x["n_pred_spells"] for x in full),
        sum(x["n_gt_spells"] for x in full)))

    models = fit_all(files, spec, params)

    # extra validation domain: Metrica sample games (event-derived labels), degraded
    # scored at the run's own k (comparable within a run) and on the raw event labels (k=0,
    # comparable across runs; raw labels are the external stand-in for the competition definition)
    val, val0, keep = {}, {}, {}
    for vp in sorted(glob.glob(os.path.join(feat_dir + "_val", "*_s*.npz"))):
        d = load(vp)
        name = os.path.basename(vp)[:-4]
        p = predict(models, {n: d[n] for n in names}, spec)
        keep[name + "_p"], keep[name + "_label"], keep[name + "_period"] = p, d["label"], d["period"]
        if not name.endswith("_s0"):
            continue
        dec = decode(p, runs_of_period(d["period"]), best[1])
        g = gt_codes(relabel_min_spell(d["label"], d["period"], k), d["period"])
        val[name[:-3]] = score(g, dec)["spell_f1_mean"]
        val0[name[:-3]] = score(gt_codes(d["label"], d["period"]), dec)["spell_f1_mean"]
    if val:
        np.savez(os.path.join(ROOT, "extdata", "valp_%s.npz" % args.tag), **keep)
        print("METRICA sample (event labels): spellF1 %s  mean %.4f   raw labels (k=0) mean %.4f" % (
            " ".join("%.3f" % v for v in val.values()), np.mean(list(val.values())),
            np.mean(list(val0.values()))))

    tf = load(os.path.join(target.DATA_DIR, "derived", "BAR-ATH_%s.npz" % args.feat_dir))
    p = predict(models, {n: tf[n] for n in names}, spec)
    runs = runs_of_period(tf["period"])
    lab = decode(p, runs, best[1])
    lab[tf["period"] == 0] = 3
    pub = score(tf["label"], lab)
    print("PUBLIC 5 min: spellF1 %.4f  macroF1 %.4f  spells pred/gt %d/%d" % (
        pub["spell_f1_mean"], pub["macro_f1"], pub["n_pred_spells"], pub["n_gt_spells"]))
    graded = tf["period"] > 0
    shares = [float(np.mean(lab[graded] == c)) for c in range(3)]
    changes = int(np.sum(np.diff(lab[graded]) != 0))
    print("FULL MATCH (unlabelled): share H %.3f A %.3f D %.3f, label changes %d" % (*shares, changes))

    with open(os.path.join(RESULTS, "%s.csv" % args.tag), "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("%d,%s\n" % (i, "HADX"[c]) for i, c in enumerate(lab))
    np.save(os.path.join(ROOT, "extdata", "probs_%s.npy" % args.tag), p)
    with open(os.path.join(RESULTS, "experiments.jsonl"), "a") as fh:
        fh.write(json.dumps({"tag": args.tag, "feat_dir": args.feat_dir, "relabel": args.relabel,
                             "team_relabel": k_team / FPS, "dead_noball": args.dead_noball,
                             "features": len(names), "drop": args.drop, "trees": args.trees,
                             "dfl_spellf1": best[0], "dfl_per_match": best[2], "decoder": best[1],
                             "metrica_spellf1": val, "metrica_k0": val0, "public_spellf1": pub["spell_f1_mean"],
                             "public_macrof1": pub["macro_f1"], "public_pred_spells": pub["n_pred_spells"],
                             "full_shares": shares, "full_changes": changes}) + "\n")
    print("done (%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
