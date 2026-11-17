"""Adversarial validation: which features tell degraded DFL apart from the competition match?

    python scripts/lab/domain_check.py [--feat-dir features_v3]

Per feature: ROC-AUC of "target vs DFL" on non-missing values (0.5 = indistinguishable) and
the difference in missing share. Compared within live-ish and dead-ish frames separately is
not possible on the target (no labels), so this is marginal. Aggregates only.
"""
import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from possession import FEAT_DIR, target  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def auc(a, b):
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 100 or len(b) < 100:
        return np.nan
    v = np.concatenate([a, b])
    order = v.argsort(kind="mergesort")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    # average ranks for ties
    sv = v[order]
    edges = np.flatnonzero(np.diff(sv) != 0) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(v)]))
    for s, e in zip(starts, ends):
        if e - s > 1:
            ranks[order[s:e]] = (s + e + 1) / 2.0
    return (ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2) / (len(a) * len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat-dir", default=FEAT_DIR)
    ap.add_argument("--n", type=int, default=30000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    t = np.load(os.path.join(target.DATA_DIR, "derived", "BAR-ATH_%s.npz" % args.feat_dir))
    names = [n for n in t.files if n not in ("period", "label")]
    ti = rng.choice(np.flatnonzero(t["period"] > 0), args.n, replace=False)
    dfl = {n: [] for n in names}
    files = sorted(glob.glob(os.path.join(ROOT, "extdata", args.feat_dir, "*_s0.npz")))
    for p in files:
        d = np.load(p)
        idx = rng.choice(np.flatnonzero(d["period"] > 0), args.n // len(files), replace=False)
        for n in names:
            dfl[n].append(d[n][idx])
    rows = []
    for n in names:
        a = t[n][ti].astype(np.float64)
        b = np.concatenate(dfl[n]).astype(np.float64)
        u = auc(a, b)
        rows.append((abs(u - 0.5) if not np.isnan(u) else 0, u, np.isnan(a).mean() - np.isnan(b).mean(),
                     np.nanmedian(a) if np.any(~np.isnan(a)) else np.nan,
                     np.nanmedian(b) if np.any(~np.isnan(b)) else np.nan, n))
    rows.sort(key=lambda r: -r[0])
    print("%-34s %6s %8s %10s %10s" % ("feature", "AUC", "dNaN", "med tgt", "med dfl"))
    for r in rows[:40]:
        print("%-34s %6.3f %+8.3f %10.3f %10.3f" % (r[5], r[1], r[2], r[3], r[4]))
    big = [r for r in rows if r[0] > 0.15]
    print("features with |AUC-0.5| > 0.15: %d of %d" % (len(big), len(rows)))


if __name__ == "__main__":
    main()
