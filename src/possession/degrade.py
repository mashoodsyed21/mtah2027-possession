"""Make clean DFL tracking look like the automatic tracking of the target match.

The DFL open data is operator-corrected: the ball is present on every frame and every
player is tracked throughout. The target tracking loses the ball for long blocks (most
of them during stoppages), filters the ball path, jitters and loses players for short
stretches. A model trained on clean data would lean on signals that do not exist at test
time, so the training data is degraded first. `degrade` does, in this order:

1. stoppages: during a stoppage DFL freezes the ball where play stopped and jumps it to the
   restart spot on the restart frame. That path is rebuilt physically: the ball rolls on
   with its incoming velocity and decelerates, is carried at walking pace to the restart
   spot and waits there;
2. smoothing: a Gaussian filter (sigma ~5 frames) on the ball inside each half, as the
   target tracker filters the ball, before any gap or noise is added. Players get a much
   lighter filter (~1 frame): the target's frame-to-frame player acceleration is lower than
   DFL's;
3. ball gaps: most stoppages lose the ball; live play loses it in crowded moments and on
   fast balls until a target share of live frames is missing; very short gaps are closed
   and a few false ball detections ("blips") appear next to players inside gaps;
4. jitter: smooth, slowly drifting position errors on players and ball;
5. player dropouts: short per-player losses, and occasional blackouts in which a large part
   of the players vanish together with the ball.

The live ball-gap lengths are partly drawn from the target match's own gaps, measured at
runtime (`gap_profile`), so the repository regenerates the same degradation from the
provided data. Everything else is randomised per match within broad ranges (domain
randomisation, `MATCH_RANGES`) so the model does not fit one exact noise level. All
randomness comes from the numpy Generator passed in, so a seed reproduces a copy exactly.
"""

import numpy as np
from scipy.ndimage import correlate1d, gaussian_filter1d
from scipy.signal import lfilter

from .features import _segments, FPS, speed

# Drawn once per match, uniformly from (lo, hi). `cfg` may replace any of them with a fixed
# number or with another (lo, hi) range.
MATCH_RANGES = {
    "ball_sigma": (4.0, 6.0),          # frames, Gaussian filter on the ball path
    "player_sigma": (0.5, 1.5),        # frames, Gaussian filter on player paths
    "p_dead": (0.6, 0.9),              # probability that a stoppage loses the ball
    "q_alive": (0.08, 0.24),           # share of live frames without a ball, at least
    "pool_share": (0.3, 0.7),          # share of live-gap lengths drawn from the target's gaps
    "player_sd": (0.1, 0.3),           # m, player position jitter
    "ball_sd": (0.15, 0.4),            # m, ball position jitter
    "dropout_per_min": (0.2, 0.8),     # player dropouts per tracked minute, per player
}

# Constants and per-event ranges (a (lo, hi) range here is drawn again for every event).
# `cfg` may override any of them.
SETTINGS = {
    # stoppage rebuild
    "dead_min_s": 1.0,                 # longer stoppages get the roll / carry / wait path
    "roll_decel": 4.0,                 # m/s^2, rolling deceleration after the stop
    "roll_s": (0.3, 2.0),              # s, how long the ball rolls on (less if it comes to rest)
    "walk_mps": (1.0, 2.0),            # m/s, ball carried to the restart spot
    "ready_s": 0.5,                    # s, the ball is on the restart spot at least this long
    "out_margin": 3.0,                 # m, a rolling ball stops this far beyond the lines at most
    "freeze_max_s": 1.0,               # s, DFL's freeze may begin up to this long before the stop
    "jump_mps": 45.0,                  # m/s, a faster ball step right after a restart is DFL's jump
    # stoppage gaps
    "dead_gap_start_s": (-0.5, 2.5),   # s after the stop the gap begins
    "dead_gap_end_s": (-0.5, 1.5),     # s before the restart the gap ends
    # live gaps
    "gap_cap_s": 15.0,                 # s, longest target gap reused for live play
    "gap_median_s": 5.0,               # lognormal live-gap length: median, s
    "gap_sigma": 0.5,                  # ... log-sd
    "gap_clip_s": (3.0, 30.0),         # ... clipped to this range, s
    "crowd_r": 3.0,                    # m, players this close to the ball occlude it
    "crowd_floor": 0.2,                # gap-start weight = n_close ** 2 + crowd_floor
    "fast_mps": 12.0,                  # m/s, a faster ball ...
    "fast_factor": 2.0,                # ... multiplies the gap-start weight by this
    "min_gap_s": 0.3,                  # shorter ball gaps are closed again
    # false ball detections inside gaps
    "blip_per_min": 0.5,               # per graded minute
    "blip_frames": (6, 10),            # length, frames (inclusive), ~0.3 s
    "blip_offset_m": 1.0,              # sd of the offset from the player the blip sticks to
    # jitter correlation times, frames
    "player_tau": 12.0,
    "ball_tau": 4.0,
    # player dropouts
    "dropout_median_s": 3.0,
    "dropout_sigma": 0.7,
    "dropout_clip_s": (1.0, 30.0),
    "blackout_p": 0.3,                 # share of ball gaps (live or dead) that also lose players
    "blackout_frac": (0.2, 0.6),       # share of the tracked players lost in a blackout
    # occlusion bursts at random times, independent of the ball. Off: degraded DFL already tracks
    # fewer players than the target (tracked count p10 16 vs 18); bursts at 0.5-1.5 / min took it
    # to 13 and widened the gap
    "burst_per_min": 0.0,              # per graded minute
    "burst_s": (1.0, 6.0),             # s, length of an occlusion burst
    "burst_frac": (0.25, 0.45),        # share of the tracked players lost in a burst
}


def runs_of(mask):
    """(starts, lengths) of True runs."""
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    s = np.flatnonzero(edges == 1)
    e = np.flatnonzero(edges == -1)
    return s, e - s


def gap_profile(target):
    """Lengths (frames) of every ball gap inside the target's graded halves."""
    graded = target["period"] > 0
    miss = np.isnan(target["ball"][:, 0]) & graded
    _, lengths = runs_of(miss)
    return np.sort(lengths)


def relabel_min_spell(label, period, min_frames):
    """Coarsen possession labels: a team spell shorter than min_frames inside a live
    stretch is given to the team that had the ball before it (to the next team if it opens
    the stretch). Mirrors the rule that loose balls and uncontrolled touches do not change
    possession. Dead ball is untouched."""
    out = label.copy()
    if min_frames <= 1:
        return out
    for a, b in _segments(period):
        lab = out[a:b]
        alive = lab < 2
        s, n = runs_of(alive)
        for start, length in zip(s, n):
            seg = lab[start:start + length]
            changed = True
            while changed:
                changed = False
                edges = np.flatnonzero(np.diff(seg) != 0) + 1
                bounds = np.concatenate(([0], edges, [len(seg)]))
                if len(bounds) <= 2:
                    break
                lens = np.diff(bounds)
                k = int(np.argmin(lens))
                if lens[k] >= min_frames:
                    break
                lo, hi = bounds[k], bounds[k + 1]
                fill = seg[lo - 1] if lo > 0 else seg[hi]
                seg[lo:hi] = fill
                changed = True
            lab[start:start + length] = seg
        out[a:b] = lab
    return out


# -- helpers ------------------------------------------------------------------------------------

def draw_params(rng, cfg=None):
    """Parameters for one degraded copy: SETTINGS plus one draw of every MATCH_RANGES entry,
    with `cfg` applied on top. Exactly one uniform number is consumed per MATCH_RANGES entry
    whatever `cfg` says, so overriding one parameter never shifts the draws of the others.
    A per-event range setting (a tuple in SETTINGS) may be given a single number, which fixes
    it; a single-number setting does not accept a range."""
    cfg = dict(cfg or {})
    unknown = set(cfg) - set(MATCH_RANGES) - set(SETTINGS)
    if unknown:
        raise KeyError("unknown degrade parameters: {}".format(sorted(unknown)))
    p = dict(SETTINGS)
    for key, default in MATCH_RANGES.items():
        u = rng.random()
        spec = cfg.get(key, default)
        if isinstance(spec, (tuple, list)):
            p[key] = float(spec[0] + u * (spec[1] - spec[0]))
        else:
            p[key] = float(spec)
    for key, spec in cfg.items():
        if key not in SETTINGS:
            continue
        is_range = isinstance(spec, (tuple, list))
        if isinstance(SETTINGS[key], tuple):
            p[key] = tuple(spec) if is_range else (spec, spec)
        elif is_range:
            raise TypeError("degrade parameter {} takes a single number".format(key))
        else:
            p[key] = spec
    return p


def runs_in(mask, segs):
    """(starts, lengths) of True runs of mask, split at segment (half) edges."""
    starts, lengths = [], []
    for a, b in segs:
        s, n = runs_of(mask[a:b])
        starts.append(s + a)
        lengths.append(n)
    if not starts:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(starts), np.concatenate(lengths)


def _run_frames(starts, lengths):
    """Frame indices covered by the runs (starts, lengths), run after run."""
    lengths = np.asarray(lengths, np.int64)
    within = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return np.repeat(np.asarray(starts, np.int64), lengths) + within


def _cover(shape, starts, stops, cols=None):
    """Boolean mask of `shape` ([T] or [T, N]) that is True on every [start, stop) interval
    (of column `cols` for 2-D masks). Intervals may overlap; empty ones are ignored."""
    starts = np.asarray(starts, np.int64)
    stops = np.asarray(stops, np.int64)
    keep = stops > starts
    T = shape[0]
    d = np.zeros((T + 1,) + tuple(shape[1:]), np.int32)
    lo, hi = np.clip(starts[keep], 0, T), np.clip(stops[keep], 0, T)
    if cols is None:
        np.add.at(d, lo, 1)
        np.add.at(d, hi, -1)
    else:
        c = np.asarray(cols, np.int64)[keep]
        np.add.at(d, (lo, c), 1)
        np.add.at(d, (hi, c), -1)
    return np.cumsum(d[:-1], axis=0) > 0


def _smooth(xy, sigma, segs):
    """NaN-aware Gaussian smoother along time inside each segment. xy is [T, ..., 2].

    NaN frames give nothing to their neighbours and stay NaN. Each frame gets a
    Gaussian-weighted local-linear fit of its neighbours: on a fully observed stretch that is
    exactly a Gaussian filter, and next to a hole or a segment edge it does not lag (a plain
    renormalised Gaussian would slow a moving ball down towards the edge). Isolated frames
    fall back to the weighted mean."""
    out = xy.copy()
    if sigma <= 0:
        return out
    ok = ~np.isnan(xy[..., 0])
    k = np.arange(-int(4 * sigma + 0.5), int(4 * sigma + 0.5) + 1, dtype=np.float64)
    g = np.exp(-0.5 * (k / sigma) ** 2)

    def corr(x, kernel):                       # sum_j kernel(j - i) * x_j
        return correlate1d(x, kernel, axis=0, mode="constant")

    for a, b in segs:
        w = ok[a:b].astype(np.float64)
        v = np.where(ok[a:b, ..., None], xy[a:b], 0.0)
        s0, s1, s2 = (corr(w, g)[..., None], corr(w, k * g)[..., None],
                      corr(w, k * k * g)[..., None])
        x0, x1 = corr(v, g), corr(v, k * g)
        det = s0 * s2 - s1 ** 2                # >= 0; 0 when one frame carries all weight
        with np.errstate(invalid="ignore", divide="ignore"):
            fit = np.where(det > 0.1 * s0 * s2, (s2 * x0 - s1 * x1) / det, x0 / s0)
        out[a:b] = np.where(ok[a:b, ..., None], fit, xy[a:b])
    return out


def _jitter(rng, shape, sd, tau):
    """Smooth zero-mean position error of standard deviation sd (m), shape [T, ..., 2].

    An AR(1) process with correlation time tau frames, Gaussian-filtered with sigma tau/2 so
    it is differentiable: a tracker's error drifts, it does not flicker from frame to frame.
    (A raw AR(1) error of 0.2 m would add ~80 m/s^2 of frame-to-frame player acceleration.)"""
    if sd <= 0 or shape[0] < 2:
        return np.zeros(shape, np.float32)
    phi = np.exp(-1.0 / tau)
    z = rng.standard_normal(shape, dtype=np.float32)
    z[0] /= np.sqrt(1.0 - phi ** 2)            # start in the stationary state
    x = lfilter(np.array([1.0], np.float32), np.array([1.0, -phi], np.float32), z, axis=0)
    x = gaussian_filter1d(x, tau / 2.0, axis=0, mode="nearest")
    return (sd / max(float(x.std()), 1e-12) * x).astype(np.float32)


# -- stoppages ----------------------------------------------------------------------------------

def stoppage_runs(ball, label, segs, freeze_max, jump_step):
    """Dead-ball runs per half, int array of rows (h0, h1, a, e, r, re).

    [a, e) is the labelled stoppage inside half [h0, h1). [r, re) is the stretch whose ball
    position is DFL's artefact and is rebuilt: DFL may freeze the ball up to `freeze_max`
    frames before the stop (r <= a), and its jump to the restart spot may land up to three
    frames after the restart (re >= e). Consecutive stretches never overlap: r is not before
    the previous stretch's re, and re stays before the next stoppage."""
    step = np.full(len(ball), np.inf)
    step[1:] = np.sqrt(((ball[1:] - ball[:-1]) ** 2).sum(1))
    still = step < 1e-4                        # NaN steps count as moving
    rows = []
    for h0, h1 in segs:
        s, n = runs_of(label[h0:h1] == 2)
        starts, stops = s + h0, s + n + h0
        prev = h0                              # end of the previous rebuilt stretch
        for i, (a, e) in enumerate(zip(starts, stops)):
            nxt = starts[i + 1] if i + 1 < len(starts) else h1
            r = a
            if a > h0:
                lo = max(h0 + 1, prev, a - freeze_max)
                moving = np.flatnonzero(~still[lo:a + 1])
                r = min(a, lo + moving[-1] + 1) if len(moving) else lo
            re = e
            jumps = np.flatnonzero(step[e + 1:min(e + 4, h1, nxt)] > jump_step)
            if e < h1 and len(jumps):
                re = e + 1 + int(jumps[-1])
            rows.append((h0, h1, a, e, r, re))
            prev = re
    return np.array(rows, np.int64).reshape(-1, 6)


def rebuild_stoppages(ball, rows, pitch, P, rng):
    """Replace DFL's frozen stoppage ball (rows from `stoppage_runs`) by a physical path.

    From the last tracked frame the ball keeps its incoming velocity (mean over 5 frames) and
    decelerates at roll_decel for roll_s seconds or until it is at rest, then is carried in a
    straight line at walking pace to the restart spot (the ball on the restart frame) and
    waits there. It arrives at least ready_s before the restart; when walking is too slow it
    moves linearly over the time left. Stoppages up to dead_min_s are a straight line, a
    stoppage that opens a half is the ball waiting on the spot. Returns (ball, rebuilt)."""
    out = ball.copy()
    rebuilt = np.zeros(len(ball), bool)
    lim0 = np.asarray(pitch, np.float64) / 2 + P["out_margin"]
    decel = P["roll_decel"]
    for h0, h1, a, e, r, re in rows:
        t_roll = rng.uniform(*P["roll_s"])     # drawn for every run: a fixed random stream
        v_walk = rng.uniform(*P["walk_mps"])
        n = re - r
        p0 = out[r - 1].astype(np.float64) if r > h0 else np.full(2, np.nan)
        p_rs = out[re].astype(np.float64) if re < h1 else np.full(2, np.nan)
        has0, has_rs = np.isfinite(p0).all(), np.isfinite(p_rs).all()
        if n <= 0 or not (has0 or has_rs):
            continue
        rebuilt[r:re] = True
        if not has0:                           # kick-off: the ball waits on the spot
            out[r:re] = p_rs
            continue
        t = np.arange(1, n + 1) / FPS          # time since the last tracked frame r - 1
        if has_rs and e - a <= P["dead_min_s"] * FPS:
            frac = np.arange(1, n + 1) / (n + 1.0)
            out[r:re] = p0 + frac[:, None] * (p_rs - p0)
            continue
        # roll on with the incoming velocity
        lo = max(h0, r - 6)
        v0 = (p0 - out[lo]) * FPS / max(r - 1 - lo, 1)
        if not np.isfinite(v0).all() or r - 1 == lo:
            v0 = np.zeros(2)
        s0 = float(np.hypot(*v0))
        u = v0 / s0 if s0 > 0 else np.zeros(2)
        t_ready = (n + 1) / FPS - P["ready_s"]  # the restart frame is at t = (n + 1) / FPS
        t_stop = min(t_roll, s0 / decel)
        if has_rs:
            t_stop = min(t_stop, 0.5 * t_ready)  # leave time to bring the ball back
        lim = np.maximum(lim0, np.abs(p0))     # stop at the advertising boards
        tr = np.minimum(t, t_stop)
        path = np.clip(p0 + u * (s0 * tr - 0.5 * decel * tr ** 2)[:, None], -lim, lim)
        if has_rs:
            p1 = np.clip(p0 + u * (s0 * t_stop - 0.5 * decel * t_stop ** 2), -lim, lim)
            t_arr = min(t_stop + float(np.hypot(*(p_rs - p1))) / v_walk, t_ready)
            if t_arr > t_stop:
                frac = np.clip((t - t_stop) / (t_arr - t_stop), 0.0, 1.0)
            else:
                frac = (t > t_stop).astype(np.float64)
            carry = p1 + frac[:, None] * (p_rs - p1)
            path = np.where((t <= t_stop)[:, None], path, carry)
        out[r:re] = path
    return out, rebuilt


# -- the degrader -------------------------------------------------------------------------------

def degrade(m, gaps, rng, cfg=None):
    """Return (ball, home, away) degraded copies of canonical DFL match m (m is not modified).

    gaps: target ball-gap lengths in frames (`gap_profile`), or None for synthetic lengths only.
    rng: numpy Generator, the only source of randomness. cfg: overrides for MATCH_RANGES /
    SETTINGS entries (unknown keys raise KeyError)."""
    P = draw_params(rng, cfg)
    period = np.asarray(m["period"])
    label = np.asarray(m["label"])
    segs = _segments(period)
    T = len(period)
    graded = period > 0
    live = graded & (label < 2)
    ball = np.array(m["ball"], np.float32)
    home = np.array(m["home"], np.float32)
    away = np.array(m["away"], np.float32)

    # 1+2) the target tracker filters the path of a real ball. DFL's stoppage ball is not one
    #    (frozen, then a jump of up to tens of metres on the restart frame), so that stretch
    #    is rebuilt first; filtering the jump instead would smear it into the live frames
    #    after every restart. The filter then runs once over the whole path, before any gap
    #    or noise is added.
    rows = stoppage_runs(ball, label, segs, int(P["freeze_max_s"] * FPS), P["jump_mps"] / FPS)
    ball, _ = rebuild_stoppages(ball, rows, m["pitch"], P, rng)
    ball = _smooth(ball, P["ball_sigma"], segs)
    home = _smooth(home, P["player_sigma"], segs)
    away = _smooth(away, P["player_sigma"], segs)
    clean_ball = ball.copy()

    # 3a) most stoppages lose the ball from shortly after the stop to shortly before the restart
    seg_end = np.zeros(T, np.int64)            # for each frame: where its half ends
    for a, b in segs:
        seg_end[a:b] = b
    k = len(rows)
    hide = rng.random(k) < P["p_dead"]
    g0 = rows[:, 2] + np.round(rng.uniform(*P["dead_gap_start_s"], k) * FPS).astype(np.int64)
    g1 = rows[:, 3] - np.round(rng.uniform(*P["dead_gap_end_s"], k) * FPS).astype(np.int64)
    g0, g1 = np.clip(g0, rows[:, 0], rows[:, 1]), np.clip(g1, rows[:, 0], rows[:, 1])
    miss = _cover((T,), g0[hide], g1[hide])

    # 3b) live play loses the ball in crowds and on fast balls, until q_alive is reached
    xy = np.concatenate([np.asarray(m["home"], np.float32), np.asarray(m["away"], np.float32)], 1)
    with np.errstate(invalid="ignore"):
        dist = np.sqrt(((xy - clean_ball[:, None, :]) ** 2).sum(-1))
        n_close = (dist < P["crowd_r"]).sum(1).astype(np.float64)
        fast = speed(clean_ball, segs) > P["fast_mps"]
    weight = (n_close ** 2 + P["crowd_floor"]) * np.where(fast, P["fast_factor"], 1.0)
    weight[~live] = 0.0
    cap = int(P["gap_cap_s"] * FPS)
    pool = np.asarray(gaps if gaps is not None else [], np.int64)
    pool = pool[(pool > 0) & (pool <= cap)]
    lg0, lg1 = np.log(P["gap_median_s"] * FPS), np.array(P["gap_clip_s"]) * FPS
    need = int(round(P["q_alive"] * live.sum())) - int((miss & live).sum())
    for _ in range(100):                       # batches of candidate gaps; ~5 are ever needed
        if need <= 0 or weight.sum() <= 0:
            break
        starts = rng.choice(T, size=64, p=weight / weight.sum())
        lengths = np.clip(np.exp(lg0 + P["gap_sigma"] * rng.standard_normal(64)), *lg1)
        if len(pool):                          # a mixture with the target's own gap lengths
            from_pool = rng.random(64) < P["pool_share"]
            lengths = np.where(from_pool, pool[rng.integers(len(pool), size=64)], lengths)
        for st, ln in zip(starts, np.maximum(np.round(lengths), 1).astype(np.int64)):
            en = min(st + ln, seg_end[st])
            new = int((live[st:en] & ~miss[st:en]).sum())
            if new == 0:
                continue
            miss[st:en] = True
            need -= new
            if need <= 0:
                break

    # 3c) close very short gaps, then add false detections next to a player inside gaps. A
    #     blip keeps at least one missing frame on either side, so it never joins the real
    #     ball path with a one-frame jump of tens of metres.
    s, n = runs_in(miss, segs)
    short = n < P["min_gap_s"] * FPS
    miss &= ~_cover((T,), s[short], s[short] + n[short])
    ball[miss] = np.nan
    gap_s, gap_n = runs_in(miss, segs)         # the final ball gaps, also used for blackouts
    gap_end = np.zeros(T, np.int64)            # for a missing frame: where its gap ends
    gap_end[_run_frames(gap_s, gap_n)] = np.repeat(gap_s + gap_n, gap_n)
    inner = miss.copy()
    inner[gap_s] = False                       # not on the first frame of a gap
    n_blips = rng.poisson(P["blip_per_min"] * graded.sum() / (60.0 * FPS))
    blip_at = rng.choice(np.flatnonzero(inner), size=n_blips) if inner.any() else []
    for f in blip_at:
        ln = int(rng.integers(P["blip_frames"][0], P["blip_frames"][1] + 1))
        offset = rng.normal(0.0, P["blip_offset_m"], 2)
        players = np.flatnonzero(~np.isnan(xy[f, :, 0]))
        if not len(players):
            continue
        j = players[rng.integers(len(players))]
        en = min(f + ln, gap_end[f] - 1)       # not on the last frame of the gap either
        ball[f:en] = xy[f:en, j] + offset       # NaN wherever that player is not tracked

    # 4) smooth jitter: players, then ball
    home += _jitter(rng, home.shape, P["player_sd"], P["player_tau"])
    away += _jitter(rng, away.shape, P["player_sd"], P["player_tau"])
    ball += _jitter(rng, ball.shape, P["ball_sd"], P["ball_tau"])

    # 5) player dropouts and blackouts, on the two teams side by side. A blackout can hit any
    #    ball gap with the same probability, live or dead: tying it to live gaps only (or to
    #    stoppages only) would turn "players vanish with the ball" into a label cue that the
    #    target tracking does not provide.
    both = np.concatenate([home, away], 1)
    on = ~np.isnan(both[:, :, 0])
    rate = P["dropout_per_min"] / (60.0 * FPS)
    d_lo, d_hi = np.array(P["dropout_clip_s"]) * FPS
    st_all, en_all, col_all = [], [], []
    for j in range(both.shape[1]):
        frames = np.flatnonzero(on[:, j])
        cnt = rng.poisson(rate * len(frames)) if len(frames) else 0
        st = rng.choice(frames, size=cnt) if cnt else np.zeros(0, np.int64)
        ln = np.clip(np.exp(np.log(P["dropout_median_s"] * FPS)
                            + P["dropout_sigma"] * rng.standard_normal(cnt)), d_lo, d_hi)
        st_all.append(st)
        en_all.append(st + np.round(ln).astype(np.int64))
        col_all.append(np.full(cnt, j))
    # occlusion bursts (see SETTINGS; off by default, and then no random draws at all)
    n_burst = rng.poisson(P["burst_per_min"] * graded.sum() / (60.0 * FPS)) if P["burst_per_min"] else 0
    for f in (rng.choice(np.flatnonzero(graded), size=n_burst) if n_burst else []):
        players = np.flatnonzero(on[f])
        if not len(players):
            continue
        q = rng.uniform(*P["burst_frac"])
        chosen = rng.choice(players, size=int(round(q * len(players))), replace=False)
        st_all.append(np.full(len(chosen), f))
        en_all.append(np.full(len(chosen), f + int(rng.uniform(*P["burst_s"]) * FPS)))
        col_all.append(chosen)
    lost = rng.random(len(gap_s)) < P["blackout_p"]
    share = rng.uniform(*P["blackout_frac"], len(gap_s))
    for st, en, is_lost, q in zip(gap_s, gap_s + gap_n, lost, share):
        players = np.flatnonzero(on[st])
        if not is_lost or not len(players):
            continue
        chosen = rng.choice(players, size=int(round(q * len(players))), replace=False)
        st_all.append(np.full(len(chosen), st))
        en_all.append(np.full(len(chosen), en))
        col_all.append(chosen)
    if st_all:
        gone = _cover(both.shape[:2], np.concatenate(st_all), np.concatenate(en_all),
                      np.concatenate(col_all))
        both[gone] = np.nan
    nh = home.shape[1]
    return ball.astype(np.float32), both[:, :nh].astype(np.float32), both[:, nh:].astype(np.float32)
