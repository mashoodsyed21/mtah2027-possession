"""Dead-ball evidence from the broadcast recording.

The TV feed cuts away from the pitch (close-ups, replays, crowd shots) mostly while the
ball is dead. Its homography file says, per broadcast frame, whether a pitch transform
exists at all (`valid`), so "the broadcast is not showing the pitch" is a cheap, strong
dead-ball cue, independent of the tactical tracking.

The broadcast is not frame-aligned to the tactical video, and no mapping is provided.
We recover one constant offset per half by matching ball positions: both recordings track
the ball in the same pitch coordinates, so at the right offset the two balls coincide
wherever both are seen (`find_offset`: share of co-visible frames within 3 m, coarse search
over +/- 80,000 frames, then refined frame by frame). The offsets are computed from the
provided data at runtime, never hard-coded, and cached with the features in
<MTAH data>/derived/BAR-ATH_broadcast_feats.npz.
"""

import os

import numpy as np

from . import target

FPS = 25


def _paths():
    folder = os.path.join(target.DATA_DIR, "broadcast")
    return (os.path.join(folder, "BAR-ATH_broadcast_atd.xml"),
            os.path.join(folder, "BAR-ATH_broadcast_atd.txt"),
            os.path.join(folder, "BAR-ATH_broadcast_homography.csv"))


def _load_broadcast():
    meta, raw, homog = _paths()
    team = target._slot_teams(meta)
    n = sum(1 for _ in open(raw))
    cx = np.full(n, np.nan, np.float32)
    ball = np.full((n, 2), np.nan, np.float32)
    nplay = np.zeros(n, np.int16)
    with open(raw) as fh:
        for line in fh:
            frame_id, groups, ball_group = line.rstrip("\n").split(":")
            i = int(frame_id) - 1
            xs = []
            for k, g in enumerate(groups.split(";")):
                if team[k] == 0 and not g.startswith("N"):
                    xs.append(float(g.split(",", 1)[0]))
                if not g.startswith("N"):
                    nplay[i] += 1
            if xs:
                cx[i] = np.mean(xs)
            v = ball_group.split(",")
            if not v[0].startswith("N"):
                ball[i] = (float(v[0]) * target.PITCH[0] - target.PITCH[0] / 2,
                           float(v[1]) * target.PITCH[1] - target.PITCH[1] / 2)
    _load_broadcast.ball = ball
    valid = np.zeros(n, np.int8)
    err = np.full(n, np.nan, np.float32)
    with open(homog) as fh:
        header = next(fh).strip().split(",")
        iv, ie = header.index("valid"), header.index("reprojection_error")
        for line in fh:
            p = line.rstrip("\n").split(",")
            f = int(p[0])
            if f < n:
                valid[f] = int(float(p[iv])) if p[iv] else 0
                err[f] = float(p[ie]) if p[ie] not in ("", "nan", "NaN") else np.nan
    return cx, nplay, valid, err


def find_offset(tac_ball, bc_ball, lo, hi, search, stride=5):
    """Offset o (tactical frame = broadcast frame + o) for tactical frames [lo, hi).

    Both recordings track the ball in the same pitch coordinates, so at the right offset the
    two ball positions coincide wherever both see it. Score = share of co-visible frames
    (sampled every `stride`) where they are within 3 m; the best offset maximises it."""
    best = (-1.0, None)
    idx = np.arange(lo, hi, stride)
    tb = tac_ball[idx]
    tok = ~np.isnan(tb[:, 0])
    for o in search:
        bi = idx - o
        inside = (bi >= 0) & (bi < len(bc_ball))
        if inside.sum() < 0.5 * len(idx):
            continue
        bb = np.full_like(tb, np.nan)
        bb[inside] = bc_ball[bi[inside]]
        ok = tok & ~np.isnan(bb[:, 0])
        if ok.sum() < 500:
            continue
        d = np.hypot(*(tb[ok] - bb[ok]).T)
        s = float(np.mean(d < 3.0))
        if s > best[0]:
            best = (s, o)
    return best


def build(tgt=None, cache=True):
    out_path = os.path.join(target.DATA_DIR, "derived", "BAR-ATH_broadcast_feats.npz")
    if cache and os.path.isfile(out_path):
        z = np.load(out_path)
        return {k: z[k] for k in z.files}
    tgt = tgt or target.load()
    T = len(tgt["period"])
    bc_cx, bc_n, bc_valid, bc_err = _load_broadcast()
    bc_ball = _load_broadcast.ball

    feats = {"bc_valid": np.full(T, np.nan, np.float32), "bc_players": np.full(T, np.nan, np.float32),
             "bc_err": np.full(T, np.nan, np.float32)}
    offsets = []
    for half, a, b in target.HALVES:
        coarse = find_offset(tgt["ball"], bc_ball, a, b + 1, range(-80000, 80001, 10), stride=50)
        fine = find_offset(tgt["ball"], bc_ball, a, b + 1, range(coarse[1] - 15, coarse[1] + 16), stride=2)
        o = fine[1]
        offsets.append((half, o, fine[0]))
        idx = np.arange(a, b + 1)
        bi = idx - o
        ok = (bi >= 0) & (bi < len(bc_valid))
        feats["bc_valid"][idx[ok]] = bc_valid[bi[ok]]
        feats["bc_players"][idx[ok]] = bc_n[bi[ok]]
        feats["bc_err"][idx[ok]] = bc_err[bi[ok]]
    feats["offsets"] = np.array(offsets, np.float64)
    if cache:
        np.savez(out_path, **feats)
    return feats
