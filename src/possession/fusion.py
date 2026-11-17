"""Naive-Bayes fusion of target-only evidence into the frame model's dead-ball probability.

The frame model is trained on DFL, which has no broadcast feed and no audio. Signals that exist
only for the competition match enter as a log-likelihood ratio on the dead-ball logit:

    logit P(D | x, s) = logit P(D | x) + log P(s | D) / P(s | alive)

The signal is binned by ``edges``; the per-bin ratio is estimated on the competition match itself
without ground truth, either from frames the frame model is confident about (``llr_bins``) or by
EM on soft dead/alive posteriors (``llr_em``). Callers multiply the returned ratio by a shrink
factor, because the signal and the frame model are not truly independent.
"""
import numpy as np


def llr_bins(signal, p_dead, graded, edges, conf=0.9, prior=1.0):
    """Per-bin log P(bin | D) / P(bin | alive) from confident pseudo-labels."""
    b = np.digitize(signal, edges)
    dead = graded & (p_dead > conf) & ~np.isnan(signal)
    alive = graded & (p_dead < 1 - conf) & ~np.isnan(signal)
    nb = len(edges) + 1
    cd = np.bincount(b[dead], minlength=nb) + prior
    ca = np.bincount(b[alive], minlength=nb) + prior
    llr = np.log(cd / cd.sum()) - np.log(ca / ca.sum())
    out = np.where(np.isnan(signal), 0.0, llr[b])
    return out.astype(np.float32), llr


def llr_em(signal, logit_model, graded, edges, iters=20, prior=1.0):
    """Same ratio, estimated by EM: soft dead/alive posteriors (model prior + current ratio)
    weight the bin counts, which update the ratio. Unlike confident-frame pseudo-labels this
    does not inherit the frame model's bias towards 'alive' on frames it gets wrong."""
    ok = graded & ~np.isnan(signal)
    b = np.digitize(np.nan_to_num(signal), edges)
    nb = len(edges) + 1
    llr = np.zeros(nb)
    for _ in range(iters):
        q = 1 / (1 + np.exp(-(logit_model[ok] + llr[b[ok]])))
        cd = np.bincount(b[ok], weights=q, minlength=nb) + prior
        ca = np.bincount(b[ok], weights=1 - q, minlength=nb) + prior
        llr = np.log(cd / cd.sum()) - np.log(ca / ca.sum())
    out = np.where(np.isnan(signal), 0.0, llr[b])
    return out.astype(np.float32), llr
