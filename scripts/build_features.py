"""Build and cache features for the degraded DFL matches and the target match.

    python scripts/build_features.py [--seeds 0 1 2] [--feat-dir features_v3] [--force]

Every DFL match (and Metrica sample game, if built) is degraded once per seed, with a numpy
Generator seeded from the seed and the match id, so each copy is reproducible. Outputs (never
committed; existing files are kept unless --force):
    ../extdata/<feat-dir>/<MATCH>_s<seed>.npz                  features + raw DFL label + period
    ../extdata/<feat-dir>_val/metrica_game<g>_s<seed>.npz      validation domain only
    <MTAH data>/derived/BAR-ATH_<feat-dir>.npz                 the competition match, not degraded
"""
import argparse
import glob
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from possession import FEAT_DIR, features, target  # noqa: E402
from possession.degrade import degrade, gap_profile  # noqa: E402
from possession.features_extra import build_extra  # noqa: E402


def all_features(m):
    f = features.build(m)
    f.update(build_extra(m))
    return f


ROOT =os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXT = os.path.join(ROOT, "extdata")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--feat-dir", default=FEAT_DIR)
    args = ap.parse_args()

    t0 = time.time()
    tgt = target.load()
    gaps = gap_profile(tgt)
    print("target loaded (%.0fs); %d ball gaps in graded frames" % (time.time() - t0, len(gaps)))

    tpath = os.path.join(target.DATA_DIR, "derived", "BAR-ATH_%s.npz" % args.feat_dir)
    if args.force or not os.path.isfile(tpath):
        f = all_features(tgt)
        np.savez(tpath, period=tgt["period"], label=tgt["label"], **f)
        print("target features: %d features" % len(f))

    sources = sorted(glob.glob(os.path.join(EXT, "dfl", "*.npz")))
    sources += sorted(glob.glob(os.path.join(EXT, "metrica", "metrica_game*.npz")))
    for path in sources:
        mid = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.join(EXT, args.feat_dir + ("_val" if mid.startswith("metrica") else ""))
        os.makedirs(out_dir, exist_ok=True)
        m = dict(np.load(path))
        if mid.startswith("metrica"):
            # validation-only domain: label 3 frames stay unknown, the clean-ish ball is degraded too
            m["period"] = m["period"].astype(np.int8)
        for seed in args.seeds:
            out = os.path.join(out_dir, "%s_s%d.npz" % (mid, seed))
            if os.path.isfile(out) and not args.force:
                continue
            t1 = time.time()
            rng = np.random.default_rng(1000 * seed + sum(map(ord, mid)))
            ball, home, away = degrade(m, gaps, rng)
            md = dict(m, ball=ball, home=home, away=away)
            f = all_features(md)
            np.savez(out, period=m["period"], label=m["label"], **f)
            print("%s seed %d: %.1f%% ball missing, %d features, %.0fs" % (
                mid, seed, 100 * np.isnan(ball[:, 0]).mean(), len(f), time.time() - t1))


if __name__ == "__main__":
    main()
