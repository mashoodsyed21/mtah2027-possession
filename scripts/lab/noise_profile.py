"""Compare tracking noise of the target match with clean and degraded DFL. Aggregates only.

    python scripts/lab/noise_profile.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from possession import target  # noqa: E402
from possession.degrade import degrade, gap_profile, runs_of  # noqa: E402
from possession.features import _segments, speed  # noqa: E402

EXT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "extdata", "dfl")


def profile(name, m, ball=None, home=None, away=None):
    ball = m["ball"] if ball is None else ball
    home = m["home"] if home is None else home
    away = m["away"] if away is None else away
    segs = _segments(m["period"])
    g = m["period"] > 0
    xy = np.concatenate([home, away], 1)
    sp = speed(xy, segs)[g]
    on = ~np.isnan(xy[g][:, :, 0])
    v = sp[~np.isnan(sp)]
    n = on.sum(1)
    lives = []
    for k in range(on.shape[1]):
        _, ln = runs_of(on[:, k])
        lives.extend(ln.tolist())
    lives = np.array(lives) / 25
    bs = speed(ball, segs)[g]
    bs = bs[~np.isnan(bs)]
    miss = np.isnan(ball[g][:, 0])
    _, gl = runs_of(miss)
    acc = np.abs(np.diff(sp, axis=0))
    acc = acc[~np.isnan(acc)] * 25
    print("%-14s players/frame p5 %2.0f med %2.0f | speed p50 %.2f p90 %.2f p99 %.1f >12m/s %.3f%% | "
          "accel p99 %.0f | track life med %.0fs p90 %.0fs n/min %.1f | ball miss %.2f gaps med %.1fs | ball spd p50 %.1f p99 %.1f"
          % (name, np.percentile(n, 5), np.median(n), np.median(v), np.percentile(v, 90), np.percentile(v, 99),
             100 * np.mean(v > 12), np.percentile(acc, 99), np.median(lives), np.percentile(lives, 90),
             len(lives) / (g.sum() / 1500), miss.mean(), np.median(gl) / 25 if len(gl) else 0,
             np.median(bs), np.percentile(bs, 99)))


tgt = target.load()
profile("TARGET", tgt)
gaps = gap_profile(tgt)
for p in sorted(glob.glob(os.path.join(EXT, "*.npz")))[:3]:
    m = dict(np.load(p))
    mid = os.path.basename(p)[:-4]
    profile(mid + " clean", m)
    b, h, a = degrade(m, gaps, np.random.default_rng(0))
    profile(mid + " degr", m, b, h, a)
