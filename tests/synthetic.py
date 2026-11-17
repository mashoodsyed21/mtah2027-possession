"""Synthetic possession sequences for decoder tests. No real match data is involved.

Ground truth: alternating possessions (2-40 s) and dead-ball stretches (5-40 s) at 25 fps,
two halves separated by an X gap, X padding before and after (like a real match file).
Probabilities: a weak true-class signal buried in temporally correlated noise, plus flicker bursts
toward a wrong class and "missing evidence" blocks where the signal is absent.
"""

import numpy as np

FPS = 25
H, A, D, X = 0, 1, 2, 3


def make_half(rng, n):
    """Ground-truth codes for one half of n frames."""
    out = np.empty(n, dtype=np.int8)
    t = 0
    label = D
    length = rng.integers(5 * FPS, 15 * FPS)      # kick-off stoppage
    team = rng.integers(0, 2)
    while t < n:
        out[t:t + length] = label
        t += length
        if label == D:
            team = rng.integers(0, 2)
            label = team
            length = rng.integers(2 * FPS, 40 * FPS + 1)
        elif rng.random() < 0.3:
            label = D
            length = rng.integers(5 * FPS, 40 * FPS + 1)
        else:
            team = 1 - team
            label = team
            length = rng.integers(2 * FPS, 40 * FPS + 1)
    return out


def noisy_probs(rng, gt, signal=1.6, ar_phi=0.95, ar_sigma=0.35, white=0.9,
                flicker_rate=1 / 150, block_rate=1 / 1500):
    """Per-frame [T, 3] probabilities for codes gt (X frames get uniform-ish noise)."""
    n = len(gt)
    logits = np.zeros((n, 3))
    alive = gt != X
    logits[alive, gt[alive]] += signal

    # AR(1) noise: slow drifts that a per-frame argmax turns into multi-frame wrong runs.
    eps = rng.normal(0.0, ar_sigma, size=(n, 3))
    ar = np.empty_like(eps)
    ar[0] = eps[0]
    for t in range(1, n):
        ar[t] = ar_phi * ar[t - 1] + eps[t]
    logits += ar + rng.normal(0.0, white, size=(n, 3))

    # Flicker: short bursts (3-15 frames) strongly toward a random wrong class.
    for start in np.flatnonzero(rng.random(n) < flicker_rate):
        length = rng.integers(3, 16)
        logits[start:start + length, rng.integers(0, 3)] += 3.0

    # Missing evidence: 1-6 s blocks where the true-class signal is gone.
    for start in np.flatnonzero(rng.random(n) < block_rate):
        length = rng.integers(1 * FPS, 6 * FPS)
        stop = min(n, start + length)
        logits[start:stop] = rng.normal(0.0, white, size=(stop - start, 3))

    logits -= logits.max(axis=1, keepdims=True)
    p = np.exp(logits)
    return p / p.sum(axis=1, keepdims=True)


def make_match(seed, half_len=80_000, pad=2_000, gap=5_000):
    """(gt, probs, runs) for a two-half synthetic match with X padding and a halftime gap."""
    rng = np.random.default_rng(seed)
    gt = np.concatenate([
        np.full(pad, X, np.int8), make_half(rng, half_len),
        np.full(gap, X, np.int8), make_half(rng, half_len),
        np.full(pad, X, np.int8),
    ])
    runs = [(pad, pad + half_len), (pad + half_len + gap, pad + 2 * half_len + gap)]
    return gt, noisy_probs(rng, gt), runs
