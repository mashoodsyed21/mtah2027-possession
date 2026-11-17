"""Choose the broadcast-fusion strength on DFL, where we have labels but no broadcast feed.

    python scripts/lab/fusion_sim.py --tag v2b_ball

A synthetic "broadcast not showing the pitch" signal is simulated on the DFL out-of-fold
predictions with the structure of the real one (aggregates measured on the competition
match: the feed shows the pitch on ~96-97% of live frames and ~24% of dead frames; cut-aways
are blocks of seconds that start a little after the stoppage). The exact fusion used on the
competition match (EM-estimated likelihood ratio x shrink) is then applied and scored with
the official spellF1 — so the shrink factor is picked by validation, not by the public window.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from possession.decode import smooth_logits, viterbi_hsmm_lite  # noqa: E402
from possession.degrade import runs_of  # noqa: E402
from possession.features import _segments, roll_mean  # noqa: E402
from possession.fusion import llr_em  # noqa: E402
from possession.metrics import score  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RESULTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results"))
FPS = 25


def simulate_invalid(g, rng, p_dead_block=0.8, live_share=0.035):
    T = len(g)
    inv = np.zeros(T, np.float32)
    s, n = runs_of(g == 2)
    for a, ln in zip(s, n):
        if rng.random() < p_dead_block:
            st = a + int(rng.uniform(0.5, 4.0) * FPS)
            en = a + ln + int(rng.uniform(-3.0, 1.0) * FPS)
            if en > st:
                inv[st:en] = 1
    live = np.flatnonzero(g < 2)
    need = int(live_share * len(live))
    while need > 0 and len(live):
        st = live[rng.integers(len(live))]
        ln = int(rng.uniform(1, 5) * FPS)
        inv[st:st + ln] = 1
        need -= ln
    return inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    run = [json.loads(l) for l in open(os.path.join(RESULTS, "experiments.jsonl"))
           if json.loads(l)["tag"] == args.tag][-1]
    prm = run["decoder"]
    z = np.load(os.path.join(ROOT, "extdata", "oof_%s.npz" % args.tag))
    mids = sorted({k[:-2] for k in z.files})
    rng = np.random.default_rng(0)
    res = {}
    for shrink in (0.0, 0.3, 0.5, 0.7, 1.0):
        s_all = []
        for mid in mids:
            p, g = z[mid + "_p"].astype(np.float64), z[mid + "_g"]
            period = (g < 3).astype(np.int8)
            runs = list(_segments(period))
            inv = simulate_invalid(g, np.random.default_rng(sum(map(ord, mid))))
            sig = roll_mean(inv, 25, runs)
            pd_ = np.clip(p[:, 2], 1e-4, 1 - 1e-4)
            logit = np.log(pd_ / (1 - pd_))
            add, _ = llr_em(sig, logit, period > 0, np.array([0.05, 0.3, 0.7, 0.95]))
            pdn = 1 / (1 + np.exp(-(logit + shrink * add)))
            ph = p[:, 0] / np.maximum(p[:, 0] + p[:, 1], 1e-9)
            q = np.stack([(1 - pdn) * ph, (1 - pdn) * (1 - ph), pdn], 1).astype(np.float32)
            lp = smooth_logits(q, temperature=prm["temperature"], window=prm["window"], runs=runs)
            lab = viterbi_hsmm_lite(lp, runs, {k: v for k, v in prm.items() if k not in ("window", "temperature")})
            s_all.append(score(g, lab)["spell_f1_mean"])
        res[shrink] = float(np.mean(s_all))
        print("shrink %.1f: DFL spellF1 %.4f" % (shrink, res[shrink]))


if __name__ == "__main__":
    main()
