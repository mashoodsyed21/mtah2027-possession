"""Per-frame features from canonical tracking (see target.py / dfl.py for the format).

Everything here is computed identically for the DFL training matches and the
competition match, so a model trained on one applies to the other. Features never
depend on player identity: a team is a set of positions per frame.

Two families:

* team-indexed features, computed once per team (suffix _h / _a). Swapping the two
  teams swaps the suffixes, which is how training data is augmented and why the
  team model can be symmetric;
* match-level features that do not depend on which team is which.

Coordinates are rotated per team into its own attacking direction ("fwd" = towards
the goal it attacks), found per half from where the team's players stand on average.
All temporal context is centred or two-sided: decoding is offline, so the future is
as available as the past.
"""

import numpy as np

FPS = 25
TOUCH_R = 1.5                 # m, a player this close to the ball "touches" it
WINDOWS = (5, 25, 75, 250)    # frames: 0.2 s, 1 s, 3 s, 10 s
CAP_S = 30.0                  # cap for time-since / time-until features


# -- small numpy helpers ----------------------------------------------------------------------

def _segments(period):
    """[(start, stop)] contiguous runs of a non-zero period value."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], period, [0]))) != 0)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if period[a] != 0:
            out.append((a, b))
    return out


def roll_mean(x, w, segs):
    """Centred NaN-aware moving mean over w frames, never crossing a segment edge."""
    out = np.full(len(x), np.nan, np.float32)
    for a, b in segs:
        v = x[a:b].astype(np.float64)
        ok = ~np.isnan(v)
        c = np.concatenate(([0.0], np.cumsum(np.where(ok, v, 0.0))))
        n = np.concatenate(([0], np.cumsum(ok)))
        i = np.arange(b - a)
        lo = np.maximum(0, i - w // 2)
        hi = np.minimum(b - a, i + w // 2 + 1)
        cnt = n[hi] - n[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:b] = np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)
    return out


def side_mean(x, w, segs, future):
    """Mean over the w frames strictly before (future=False) or after (True) each frame."""
    out = np.full(len(x), np.nan, np.float32)
    for a, b in segs:
        v = x[a:b].astype(np.float64)
        ok = ~np.isnan(v)
        c = np.concatenate(([0.0], np.cumsum(np.where(ok, v, 0.0))))
        n = np.concatenate(([0], np.cumsum(ok)))
        i = np.arange(b - a)
        if future:
            lo, hi = np.minimum(b - a, i + 1), np.minimum(b - a, i + 1 + w)
        else:
            lo, hi = np.maximum(0, i - w), i
        cnt = n[hi] - n[lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out[a:b] = np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)
    return out


def ffill_index(mask, segs):
    """Index of the last frame <= t where mask is True (within segment), -1 if none."""
    out = np.full(len(mask), -1, np.int64)
    for a, b in segs:
        idx = np.where(mask[a:b], np.arange(a, b), -1)
        out[a:b] = np.maximum.accumulate(idx)
    return out


def bfill_index(mask, segs):
    """Index of the next frame >= t where mask is True (within segment), -1 if none."""
    out = np.full(len(mask), -1, np.int64)
    for a, b in segs:
        big = np.iinfo(np.int64).max
        idx = np.where(mask[a:b], np.arange(a, b), big)
        r = np.minimum.accumulate(idx[::-1])[::-1]
        out[a:b] = np.where(r == big, -1, r)
    return out


def speed(xy, segs):
    """Per-object speed in m/s from positions, central difference, NaN-safe. xy [T, ..., 2]."""
    v = np.full(xy.shape[:-1], np.nan, np.float32)
    for a, b in segs:
        if b - a < 3:
            continue
        d = (xy[a + 2:b] - xy[a:b - 2]) * (FPS / 2.0)
        v[a + 1:b - 1] = np.sqrt((d ** 2).sum(-1))
    return v


# -- the feature builder ----------------------------------------------------------------------

def attack_sign(team_xy, period, segs):
    """+1 if the team attacks towards +x in that half, -1 otherwise; per frame."""
    sign = np.zeros(len(period), np.float32)
    for a, b in segs:
        with np.errstate(all="ignore"):
            mean_x = np.nanmean(team_xy[a:b, :, 0])
        sign[a:b] = -1.0 if mean_x > 0 else 1.0   # a team stands in its own half on average
    return sign


def team_block(xy, ball, sign, segs, prefix):
    """Features of one team (xy [T, N, 2])."""
    f = {}
    ok = ~np.isnan(xy[:, :, 0])
    f[prefix + "n"] = ok.sum(1).astype(np.float32)
    with np.errstate(all="ignore"):
        cx = np.nanmean(xy[:, :, 0], 1)
        cy = np.nanmean(xy[:, :, 1], 1)
        f[prefix + "cx_fwd"] = cx * sign
        f[prefix + "width"] = np.nanstd(xy[:, :, 1], 1)
        f[prefix + "length"] = np.nanstd(xy[:, :, 0], 1)
        f[prefix + "spread"] = f[prefix + "width"] * f[prefix + "length"]
        f[prefix + "ymax_minus_ymin"] = np.nanmax(xy[:, :, 1], 1) - np.nanmin(xy[:, :, 1], 1)
        # how far up the pitch the last outfield line sits: 2nd smallest forward x
        fx = xy[:, :, 0] * sign[:, None]
        fx_sorted = np.sort(np.where(ok, fx, np.inf), 1)
        f[prefix + "deepest2_fwd"] = np.where(np.isfinite(fx_sorted[:, 1]), fx_sorted[:, 1], np.nan)
    sp = speed(xy, segs)
    with np.errstate(all="ignore"):
        f[prefix + "speed"] = np.nanmean(sp, 1)
    # central difference inside each half only (as speed()): DFL stores the halves back to
    # back, and a difference across that seam would see the teams swap ends (~300 m/s)
    vcx = np.full(len(cx), np.nan, np.float32)
    for a, b in segs:
        if b - a >= 3:
            vcx[a + 1:b - 1] = (cx[a + 2:b] - cx[a:b - 2]) * (FPS / 2.0)
    f[prefix + "vcx_fwd"] = vcx * sign

    d = np.sqrt(((xy - ball[:, None, :]) ** 2).sum(-1))       # [T, N], NaN if either missing
    with np.errstate(all="ignore"):
        dmin = np.nanmin(np.where(ok, d, np.nan), 1)
    f[prefix + "dball"] = dmin.astype(np.float32)
    for r in (3.0, 6.0):
        f[prefix + "nball%d" % r] = np.where(np.isnan(ball[:, 0]), np.nan,
                                             (d < r).sum(1)).astype(np.float32)
    return f, dmin


def build(m, ball=None):
    """Feature dict {name: float32[T]} for canonical match dict m.

    `ball` overrides m["ball"] (used to pass a degraded ball track).
    """
    period = m["period"].astype(np.int8)
    segs = _segments(period)
    T = len(period)
    ball = m["ball"] if ball is None else ball
    L, W = float(m["pitch"][0]), float(m["pitch"][1])
    vis = ~np.isnan(ball[:, 0])

    sh = attack_sign(m["home"], period, segs)
    sa = attack_sign(m["away"], period, segs)
    fh, dh = team_block(m["home"], ball, sh, segs, "h_")
    fa, da = team_block(m["away"], ball, sa, segs, "a_")
    f = {}
    f.update(fh)
    f.update(fa)

    # ball state
    f["ball_vis"] = vis.astype(np.float32)
    bsp = speed(ball, segs)
    f["ball_speed"] = bsp
    acc = np.full(T, np.nan, np.float32)
    acc[1:-1] = np.abs(bsp[2:] - bsp[:-2]) * (FPS / 2.0)
    f["ball_acc"] = acc
    f["ball_out"] = np.where(vis, ((np.abs(ball[:, 0]) > L / 2) | (np.abs(ball[:, 1]) > W / 2)),
                             np.nan).astype(np.float32)
    f["ball_line_dist"] = np.where(vis, np.minimum(L / 2 - np.abs(ball[:, 0]),
                                                   W / 2 - np.abs(ball[:, 1])), np.nan).astype(np.float32)
    f["ball_absx"] = np.abs(ball[:, 0])
    f["ball_absy"] = np.abs(ball[:, 1])
    # ball x in each team's attacking direction (team-indexed)
    f["h_ball_fwd"] = ball[:, 0] * sh
    f["a_ball_fwd"] = ball[:, 0] * sa

    # time since / until the ball was / will be seen, and where it was last seen
    last = ffill_index(vis, segs)
    nxt = bfill_index(vis, segs)
    idx = np.arange(T)
    f["t_since_ball"] = np.where(last >= 0, np.minimum((idx - last) / FPS, CAP_S), CAP_S).astype(np.float32)
    f["t_until_ball"] = np.where(nxt >= 0, np.minimum((nxt - idx) / FPS, CAP_S), CAP_S).astype(np.float32)
    held = np.where(last[:, None] >= 0, ball[np.maximum(last, 0)], np.nan)
    f["held_out"] = ((np.abs(held[:, 0]) > L / 2) | (np.abs(held[:, 1]) > W / 2)).astype(np.float32)
    f["held_line_dist"] = np.minimum(L / 2 - np.abs(held[:, 0]), W / 2 - np.abs(held[:, 1])).astype(np.float32)
    for name, xy, s in (("h_", m["home"], sh), ("a_", m["away"], sa)):
        dd = np.sqrt(((xy - held[:, None, :]) ** 2).sum(-1))
        with np.errstate(all="ignore"):
            f[name + "dheld"] = np.nanmin(dd, 1).astype(np.float32)
        f[name + "held_fwd"] = (held[:, 0] * s).astype(np.float32)

    # nearest-team evidence and touches
    near_h = np.where(np.isnan(dh) | np.isnan(da), np.nan, (dh < da).astype(np.float32))
    f["near_home"] = near_h
    f["margin_h"] = (da - dh).astype(np.float32)          # >0: home closer
    touch_h = vis & (dh <= TOUCH_R)
    touch_a = vis & (da <= TOUCH_R)
    f["touch_h"] = touch_h.astype(np.float32)
    f["touch_a"] = touch_a.astype(np.float32)
    for team, t in (("h", touch_h), ("a", touch_a)):
        lt = ffill_index(t, segs)
        nt = bfill_index(t, segs)
        f["t_since_touch_" + team] = np.where(lt >= 0, np.minimum((idx - lt) / FPS, CAP_S), CAP_S).astype(np.float32)
        f["t_until_touch_" + team] = np.where(nt >= 0, np.minimum((nt - idx) / FPS, CAP_S), CAP_S).astype(np.float32)

    # all-player speed: the strongest dead-ball cue
    with np.errstate(all="ignore"):
        f["all_speed"] = np.nanmean(np.stack([fh["h_speed"], fa["a_speed"]]), 0).astype(np.float32)

    # temporal context
    ctx = ["near_home", "margin_h", "touch_h", "touch_a", "ball_vis", "ball_speed", "all_speed",
           "h_speed", "a_speed", "h_width", "a_width", "h_spread", "a_spread", "h_cx_fwd", "a_cx_fwd",
           "h_vcx_fwd", "a_vcx_fwd", "ball_out", "h_dball", "a_dball"]
    for name in ctx:
        for w in WINDOWS:
            f["%s_m%d" % (name, w)] = roll_mean(f[name], w, segs)
    for name in ("near_home", "touch_h", "touch_a", "all_speed", "ball_vis"):
        for w in (25, 125):
            f["%s_past%d" % (name, w)] = side_mean(f[name], w, segs, future=False)
            f["%s_fut%d" % (name, w)] = side_mean(f[name], w, segs, future=True)
    f["all_speed_m750"] = roll_mean(f["all_speed"], 750, segs)
    f["ball_vis_m750"] = roll_mean(f["ball_vis"], 750, segs)

    # time within the half (restarts cluster at the start), as a fraction
    tin = np.zeros(T, np.float32)
    for a, b in segs:
        tin[a:b] = np.arange(b - a) / FPS
    f["t_in_half"] = np.minimum(tin, 120.0)
    return {k: np.asarray(v, np.float32) for k, v in f.items()}


def swap_teams(f):
    """Same features with the two teams exchanged (for symmetric training)."""
    g = {}
    for k, v in f.items():
        if k.startswith("h_"):
            g["a_" + k[2:]] = v
        elif k.startswith("a_"):
            g["h_" + k[2:]] = v
        else:
            g[k] = v
    flip = {"near_home": lambda v: 1.0 - v, "margin_h": lambda v: -v}
    for k in list(g):
        base = k.split("_m")[0].split("_past")[0].split("_fut")[0]
        if k.startswith("touch_h") or k.startswith("touch_a") or "_touch_" in k:
            continue
        if base in flip and (k == base or k[len(base):].startswith(("_m", "_past", "_fut"))):
            g[k] = flip[base](f[k])
    # touch_h <-> touch_a and their context / time features
    for k in f:
        if k.startswith("touch_h"):
            g["touch_a" + k[len("touch_h"):]] = f[k]
        elif k.startswith("touch_a"):
            g["touch_h" + k[len("touch_a"):]] = f[k]
        elif k.endswith("_touch_h"):
            g[k[:-1] + "a"] = f[k]
        elif k.endswith("_touch_a"):
            g[k[:-1] + "h"] = f[k]
    return g
