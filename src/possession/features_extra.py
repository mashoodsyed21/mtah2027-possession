"""Additional per-frame features, computed on top of features.build (same canonical input).

Families (each one a helper below):

* movement    - which share of players walks / sprints, whether everybody accelerates or brakes
                at the same time (whistles and restarts synchronise the players), speed contrast;
* control     - "clear control" (a player on a slow ball with no opponent near it) and a soft
                touch kernel, both per team, with time since / until and past / future shares;
* ball gaps   - ball-out shares, how close each team is to the ball where it was last seen and
                where it is seen again (visible runs shorter than MIN_SIGHTING frames do not
                count as sightings), gap length and implied ball speed across the gap;
* set pieces  - geometric templates for throw-in, corner, goal kick (per team) and kick-off;
* duels       - a "long-gap anchor" (where the closest opponent pairs are, a proxy for the ball
                when it is not tracked) and goal-side marking per team;
* shape       - forward position, width and spread of each team: next 2 s minus past 2 s.

Naming rules:

* (relied on by features.swap_teams) a feature about one specific team starts with "h_" (home)
  or "a_" (away), and every "h_x" has an "a_x" computed by exactly the same code with the teams
  exchanged. Features without that prefix do not depend on which team is home. Hence
  build_extra(match with teams exchanged) == swap_teams(build_extra(match)) up to float rounding;
* (relied on by the ball-free dead-ball model in scripts/train_eval.py, which drops features
  by name) every feature that depends on the ball track has "ball" in its name.

Window suffixes are in frames at 25 fps: "_c<w>" centred mean over w frames (+-w//2),
"_past<w>" / "_fut<w>" mean over the w frames strictly before / after the frame.
All features are NaN outside the graded halves (period 0).
"""

import numpy as np
from scipy.ndimage import maximum_filter1d

from .features import (CAP_S, FPS, _segments, attack_sign, bfill_index, ffill_index, roll_mean,
                       side_mean, speed)

# movement
WALK_V = 2.0               # m/s, below this a player walks / stands
FAST_V = 4.5               # m/s, above this a player runs fast
# control
CTRL_R = 1.0               # m, controlling player to ball
CTRL_OPP_R = 2.5           # m, no opponent this close to the ball
CTRL_BALL_V = 6.0          # m/s, the ball must be slower than this
KERNEL_SIGMA = 1.5         # m, width of the touch kernel exp(-0.5 (d / sigma)^2)
# ball gaps
MIN_SIGHTING = 12          # frames; shorter visible runs do not count as "ball seen again"
GAP_EDGE = 5               # frames of a sighting next to a gap that give the ball speed there
# set pieces (pitch geometry in metres)
SP_WINDOW = 151            # frames, rolling max over +-3 s
THROW_LINE = 0.5           # m, player within this of the touchline (or beyond it)
THROW_BALL = 3.0           # m, ball last / next seen within this of the same touchline
CORNER_R = 2.0             # m, player to corner flag
CORNER_CROWD = 6           # players inside the 16.5 m zone of that end
BOX_DEPTH, BOX_HALF_W = 16.5, 20.16        # penalty area
SIX_DEPTH, SIX_HALF_W = 5.5, 9.16          # goal area (six-yard box)
GK_BALL_LINE = 1.0         # m, goal kick: ball last seen this close to (or over) the goal line
GK_BALL_MARGIN = 1.0       # m, goal kick: ball next seen inside the six-yard box + this margin
KICKOFF_R = 1.5            # m, a player this close to the centre spot
KICKOFF_TOL = 1.5          # m, players may stand this far into the opponent half
KICKOFF_MIN_N = 8          # tracked players needed before "everybody in own half" is believed
# duels
ANCHOR_K = 6               # players with the closest opponent that define the anchor
ANCHOR_EPS = 0.5           # m, weight = 1 / (d_opp + eps)
ANCHOR_R = 5.0             # m, "near the anchor"
MARK_R = 5.0               # m, a goal-side opponent this close marks the player
PAIR_BUDGET = 1 << 22      # home x away player pairs per block of frames (bounds the memory)


# -- helpers ------------------------------------------------------------------------------------

def _row_mean(x, ok):
    """Mean over axis 1 of x where ok; NaN for rows without a valid entry (no warnings)."""
    n = ok.sum(1)
    s = np.where(ok, x, 0.0).sum(1, dtype=np.float64)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan).astype(np.float32)


def _row_std(x, ok):
    """Population std over axis 1 where ok (like np.nanstd), NaN for empty rows."""
    mu = _row_mean(x, ok)
    return np.sqrt(_row_mean((x - mu[:, None]) ** 2, ok))


def _row_min(x):
    """NaN-aware min over axis 1, NaN when the whole row is NaN."""
    v = np.where(np.isnan(x), np.inf, x).min(1)
    return np.where(np.isinf(v), np.nan, v).astype(np.float32)


def _velocity(xy, segs):
    """Per-object velocity [T, ..., 2] in m/s, central difference inside segments (as speed())."""
    v = np.full(xy.shape, np.nan, np.float32)
    for a, b in segs:
        if b - a >= 3:
            v[a + 1:b - 1] = (xy[a + 2:b] - xy[a:b - 2]) * (FPS / 2.0)
    return v


def _diff_time(x, segs):
    """Central time derivative along axis 0 inside segments, NaN elsewhere."""
    out = np.full(x.shape, np.nan, np.float32)
    for a, b in segs:
        if b - a >= 3:
            out[a + 1:b - 1] = (x[a + 2:b] - x[a:b - 2]) * (FPS / 2.0)
    return out


def roll_max(x, w, segs):
    """Centred moving max over w (odd) frames inside each segment; NaN counts as 0."""
    out = np.full(len(x), np.nan, np.float32)
    for a, b in segs:
        v = np.nan_to_num(np.asarray(x[a:b], np.float32), nan=0.0)
        out[a:b] = maximum_filter1d(v, size=w, mode="constant", cval=0.0)
    return out


def _drop_short_runs(mask, min_len, segs):
    """mask without its True runs shorter than min_len frames (runs are cut at segment edges)."""
    out = np.zeros(len(mask), bool)
    for a, b in segs:
        edges = np.diff(np.concatenate(([0], mask[a:b].astype(np.int8), [0])))
        s, e = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        keep = (e - s) >= min_len
        step = np.zeros(b - a + 1, np.int64)
        np.add.at(step, s[keep], 1)
        np.add.at(step, e[keep], -1)
        out[a:b] = np.cumsum(step)[:-1] > 0
    return out


def _since_until(mask, segs):
    """Seconds since the last / until the next True frame (0 on it), capped at CAP_S."""
    idx = np.arange(len(mask))
    last = ffill_index(mask, segs)
    nxt = bfill_index(mask, segs)
    since = np.where(last >= 0, np.minimum((idx - last) / FPS, CAP_S), CAP_S)
    until = np.where(nxt >= 0, np.minimum((nxt - idx) / FPS, CAP_S), CAP_S)
    return since.astype(np.float32), until.astype(np.float32)


def _take(x, idx):
    """x[idx] along axis 0 with idx == -1 giving NaN."""
    out = x[np.maximum(idx, 0)].astype(np.float32)
    out[idx < 0] = np.nan
    return out


def _trend(x, w, segs):
    """mean over the next w frames minus mean over the previous w frames."""
    return side_mean(x, w, segs, future=True) - side_mean(x, w, segs, future=False)


# -- shared per-match state ---------------------------------------------------------------------

class _Ctx:
    """Arrays shared by all feature families (computed once per match)."""

    def __init__(self, m, ball):
        self.period = m["period"].astype(np.int8)
        self.segs = _segments(self.period)
        self.T = len(self.period)
        self.ball = np.asarray(m["ball"] if ball is None else ball, np.float32)
        self.L, self.W = float(m["pitch"][0]), float(m["pitch"][1])
        self.vis = ~np.isnan(self.ball[:, 0])
        self.inseg = self.period > 0
        self.xy = {"h_": np.asarray(m["home"], np.float32), "a_": np.asarray(m["away"], np.float32)}
        self.opp = {"h_": "a_", "a_": "h_"}
        self.ok, self.sign, self.vel, self.sp, self.dball, self.dmin = {}, {}, {}, {}, {}, {}
        self.ctrl = {}                     # filled by _control, read by _ball_gaps
        for p, xy in self.xy.items():
            self.ok[p] = ~np.isnan(xy[:, :, 0])
            self.sign[p] = attack_sign(xy, self.period, self.segs)
            self.vel[p] = _velocity(xy, self.segs)
            self.sp[p] = np.sqrt((self.vel[p] ** 2).sum(-1))
            d = np.sqrt(((xy - self.ball[:, None, :]) ** 2).sum(-1))   # NaN if either missing
            self.dball[p] = d
            self.dmin[p] = _row_min(d)


# -- feature families ---------------------------------------------------------------------------

def _movement(c):
    f = {}
    sp_all = np.concatenate([c.sp["h_"], c.sp["a_"]], 1)
    for p, sp in (("", sp_all), ("h_", c.sp["h_"]), ("a_", c.sp["a_"])):
        ok = ~np.isnan(sp)
        # shares of the tracked players, not counts: the target tracks fewer players per frame
        # (p5 14) than degraded DFL (p5 ~19), and a count would read low coverage as low tempo
        walk = _row_mean(sp < WALK_V, ok)
        fast = _row_mean(sp > FAST_V, ok)
        for w in (51, 151):                                  # +-1 s, +-3 s
            f["%swalk_share_c%d" % (p, w)] = roll_mean(walk, w, c.segs)
            f["%sfast_share_c%d" % (p, w)] = roll_mean(fast, w, c.segs)

    # signed (tangential) acceleration per player, averaged over everybody tracked: strongly
    # negative when the whole pitch brakes at a whistle, positive when play restarts
    acc = _diff_time(sp_all, c.segs)
    acc_mean = _row_mean(acc, ~np.isnan(acc))
    for w in (25, 75):                                       # +-0.5 s, +-1.5 s
        f["acc_sync_c%d" % w] = roll_mean(acc_mean, w, c.segs)

    all_speed = _row_mean(sp_all, ~np.isnan(sp_all))
    f["speed_contrast_75"] = _trend(all_speed, 75, c.segs)  # next 3 s minus past 3 s
    return f


def _control(c):
    f = {}
    bsp = speed(c.ball, c.segs)
    slow = ~(bsp >= CTRL_BALL_V)          # NaN speed (first / last frame around a gap) does not veto
    for p in ("h_", "a_"):
        near = c.dmin[p] <= CTRL_R
        free = ~(c.dmin[c.opp[p]] <= CTRL_OPP_R)
        ctrl = c.vis & slow & near & free
        c.ctrl[p] = np.where(c.vis, ctrl, np.nan).astype(np.float32)
        since, until = _since_until(ctrl, c.segs)
        f[p + "t_since_ball_ctrl"] = since
        f[p + "t_until_ball_ctrl"] = until
        ctrl_f = np.where(c.vis, ctrl, np.nan).astype(np.float32)   # share among visible frames
        for w in (25, 75):
            f["%sball_ctrl_past%d" % (p, w)] = side_mean(ctrl_f, w, c.segs, future=False)
            f["%sball_ctrl_fut%d" % (p, w)] = side_mean(ctrl_f, w, c.segs, future=True)

        # soft touch evidence: sum of Gaussian kernels of the team's distances to the ball
        k = np.exp(-0.5 * (c.dball[p] / KERNEL_SIGMA) ** 2)
        kern = np.where(c.vis, np.where(np.isnan(k), 0.0, k).sum(1), np.nan).astype(np.float32)
        for w in (6, 25, 75):
            f["%sball_tk_past%d" % (p, w)] = side_mean(kern, w, c.segs, future=False)
            f["%sball_tk_fut%d" % (p, w)] = side_mean(kern, w, c.segs, future=True)
    return f


def _ball_gaps(c):
    f = {}
    b = c.ball
    out = np.where(c.vis, (np.abs(b[:, 0]) > c.L / 2) | (np.abs(b[:, 1]) > c.W / 2), np.nan)
    out = out.astype(np.float32)
    f["ball_out_past50"] = side_mean(out, 50, c.segs, future=False)
    f["ball_out_c51"] = roll_mean(out, 51, c.segs)

    # Sightings for the last / next lookups below: visible runs shorter than MIN_SIGHTING are
    # treated as part of the gap (short false detections next to a player inside a gap would
    # otherwise pass for the restart).
    seen = _drop_short_runs(c.vis, MIN_SIGHTING, c.segs)
    last = ffill_index(seen, c.segs)
    nxt = bfill_index(seen, c.segs)

    # restart cues: how close each team is to the ball where it is seen next / was seen last
    # (measured at that frame). On a sighted frame both equal the current distance.
    for p in ("h_", "a_"):
        f[p + "d_ball_next"] = _take(c.dmin[p], nxt)
        f[p + "d_ball_last"] = _take(c.dmin[p], last)

    # the current ball gap: total length (0 on sighted frames; open gaps run to the half edge)
    # and implied ball speed = displacement between last and next sighting / elapsed time
    seg_lo = np.zeros(c.T, np.int64)
    seg_hi = np.zeros(c.T, np.int64)
    for a, bnd in c.segs:
        seg_lo[a:bnd], seg_hi[a:bnd] = a, bnd
    start = np.where(last >= 0, last + 1, seg_lo)
    stop = np.where(nxt >= 0, nxt, seg_hi)
    gap_len = np.where(seen, 0.0, (stop - start) / FPS)
    f["ball_gap_len_s"] = np.where(c.inseg, gap_len, np.nan).astype(np.float32)
    both = (~seen) & (last >= 0) & (nxt >= 0) & c.inseg
    disp = np.sqrt(((_take(b, nxt) - _take(b, last)) ** 2).sum(-1))
    dt = np.maximum(nxt - last, 1) / FPS
    f["ball_gap_speed"] = np.where(both, disp / dt, np.nan).astype(np.float32)

    # gap context: where and how the ball left and came back. A restart reappears slowly on a
    # line or a spot; a pass in flight reappears fast in open play. Position at the sighting is
    # relative to the half-length / half-width, the forward version per team.
    gap = ~seen & c.inseg
    f["ball_gap_pos"] = np.where(gap & (last >= 0) & (nxt >= 0),
                                 (np.arange(c.T) - start) / np.maximum(stop - start, 1),
                                 np.nan).astype(np.float32)
    bsp = speed(b, c.segs)
    sp_in = side_mean(bsp, GAP_EDGE, c.segs, future=False)   # the frames just before a sighting end
    sp_out = side_mean(bsp, GAP_EDGE, c.segs, future=True)   # ... just after a sighting starts
    f["ball_last_speed"] = _take(sp_in, last)
    f["ball_next_speed"] = _take(sp_out, nxt)
    for tag, at in (("last", last), ("next", nxt)):
        pos = _take(b, at)
        ax, ay = np.abs(pos[:, 0]) / (c.L / 2), np.abs(pos[:, 1]) / (c.W / 2)
        f["ball_%s_xabs" % tag] = ax
        f["ball_%s_yabs" % tag] = ay
        f["ball_%s_line_d" % tag] = np.minimum(c.L / 2 - np.abs(pos[:, 0]),
                                               c.W / 2 - np.abs(pos[:, 1]))
        for p in ("h_", "a_"):
            f["%sball_%s_fwd" % (p, tag)] = pos[:, 0] * c.sign[p] / (c.L / 2)
    # control share over the sighting next to the gap (the frame itself included)
    after_last = np.where(last >= 0, np.minimum(last + 1, c.T - 1), -1)
    before_next = np.where(nxt >= 0, nxt - 1, -1)
    for p in ("h_", "a_"):
        f[p + "ball_ctrl_last"] = _take(side_mean(c.ctrl[p], MIN_SIGHTING, c.segs, False), after_last)
        f[p + "ball_ctrl_next"] = _take(side_mean(c.ctrl[p], MIN_SIGHTING, c.segs, True), before_next)
    return f, last, nxt


def _set_pieces(c, last, nxt):
    f = {}
    L, W = c.L, c.W
    held = _take(c.ball, last)                  # ball where it was seen last / will be seen next
    nextb = _take(c.ball, nxt)
    held_y, next_y = held[:, 1], nextb[:, 1]
    all_xy = np.concatenate([c.xy["h_"], c.xy["a_"]], 1)
    for p in ("h_", "a_"):
        xy, s = c.xy[p], c.sign[p]
        x, y = xy[:, :, 0], xy[:, :, 1]
        fx = x * s[:, None]                     # forward x in the team's attacking direction

        # throw-in: a team player on (or past) a touchline, ball last or next seen near that line
        throw = np.zeros(c.T, bool)
        for side in (1.0, -1.0):
            on_line = (y * side > W / 2 - THROW_LINE).any(1)
            ball_near = (held_y * side > W / 2 - THROW_BALL) | (next_y * side > W / 2 - THROW_BALL)
            throw |= on_line & ball_near
        f[p + "sp_throw_ball"] = roll_max(throw, SP_WINDOW, c.segs)

        # corner: a team player at a corner flag of the attacked end, >= 6 players in that end's
        # 16.5 m zone (both teams counted)
        at_flag = np.zeros(c.T, bool)
        for side in (1.0, -1.0):
            at_flag |= (np.hypot(fx - L / 2, y - side * W / 2) <= CORNER_R).any(1)
        crowd = (all_xy[:, :, 0] * s[:, None] > L / 2 - BOX_DEPTH).sum(1) >= CORNER_CROWD
        f[p + "sp_corner"] = roll_max(at_flag & crowd, SP_WINDOW, c.segs)

        # goal kick: the team's deepest player stands in its own six-yard box, no opponent is
        # inside that penalty area, and the ball was last seen at / over that goal line or is
        # next seen in that six-yard box. (The keeper alone is in its six-yard box on ~25% of
        # DFL frames, at the base dead-ball rate; the ball condition lifts P(dead) to ~0.8.)
        fxm = np.where(c.ok[p], fx, np.inf)
        k = fxm.argmin(1)
        rows = np.arange(c.T)
        deep_fx, deep_y = fxm[rows, k], y[rows, k]
        in_six = np.isfinite(deep_fx) & (deep_fx < -L / 2 + SIX_DEPTH) & (np.abs(deep_y) < SIX_HALF_W)
        o = c.xy[c.opp[p]]
        opp_in_box = ((o[:, :, 0] * s[:, None] < -L / 2 + BOX_DEPTH)
                      & (np.abs(o[:, :, 1]) < BOX_HALF_W)).any(1)
        ball_over = held[:, 0] * s < -L / 2 + GK_BALL_LINE
        ball_six = ((nextb[:, 0] * s < -L / 2 + SIX_DEPTH + GK_BALL_MARGIN)
                    & (np.abs(nextb[:, 1]) < SIX_HALF_W + GK_BALL_MARGIN))
        goalkick = in_six & ~opp_in_box & (ball_over | ball_six)
        f[p + "sp_goalkick_ball"] = roll_max(goalkick, SP_WINDOW, c.segs)

    # kick-off: every tracked player in its own half (small tolerance for the kicker), enough
    # players tracked to believe it, and somebody on the centre spot
    own = np.ones(c.T, bool)
    n = np.zeros(c.T, np.int64)
    for p in ("h_", "a_"):
        fx = c.xy[p][:, :, 0] * c.sign[p][:, None]
        own &= ~(fx > KICKOFF_TOL).any(1)
        n += c.ok[p].sum(1)
    centre = (np.hypot(all_xy[:, :, 0], all_xy[:, :, 1]) <= KICKOFF_R).any(1)
    f["sp_kickoff"] = roll_max(own & centre & (n >= KICKOFF_MIN_N), SP_WINDOW, c.segs)
    return f


def _duels(c):
    """Long-gap anchor and goal-side marking; both need home x away player distances."""
    f = {}
    L = c.L
    hxy, axy = c.xy["h_"], c.xy["a_"]
    sh, sa = c.sign["h_"], c.sign["a_"]
    goal_h = np.stack([sh * L / 2, np.zeros(c.T, np.float32)], 1)     # goal attacked by home
    goal_a = np.stack([sa * L / 2, np.zeros(c.T, np.float32)], 1)

    def dist_to(xy, pt):
        return np.sqrt(((xy - pt[:, None, :]) ** 2).sum(-1))

    h_to_gh, a_to_gh = dist_to(hxy, goal_h), dist_to(axy, goal_h)
    h_to_ga, a_to_ga = dist_to(hxy, goal_a), dist_to(axy, goal_a)

    Nh, Na = hxy.shape[1], axy.shape[1]
    dopp_h = np.full((c.T, Nh), np.inf, np.float32)
    dopp_a = np.full((c.T, Na), np.inf, np.float32)
    marked_h = np.zeros((c.T, Nh), bool)
    marked_a = np.zeros((c.T, Na), bool)
    chunk = max(1, PAIR_BUDGET // max(Nh * Na, 1))      # frames per block
    for a in range(0, c.T, chunk):
        b = min(a + chunk, c.T)
        d = np.sqrt(((hxy[a:b, :, None, :] - axy[a:b, None, :, :]) ** 2).sum(-1))  # [n, Nh, Na]
        d = np.where(np.isnan(d), np.inf, d)
        dopp_h[a:b] = d.min(2)
        dopp_a[a:b] = d.min(1)
        close = d <= MARK_R
        # an opponent within MARK_R that is closer to the goal the player attacks
        marked_h[a:b] = (close & (a_to_gh[a:b, None, :] < h_to_gh[a:b, :, None])).any(2)
        marked_a[a:b] = (close & (h_to_ga[a:b, :, None] < a_to_ga[a:b, None, :])).any(1)

    # goal-side marking share over outfield players (the deepest player is taken as the keeper
    # when it stands in its own penalty area)
    for p, marked in (("h_", marked_h), ("a_", marked_a)):
        fx = c.xy[p][:, :, 0] * c.sign[p][:, None]
        y = c.xy[p][:, :, 1]
        fxm = np.where(c.ok[p], fx, np.inf)
        k = fxm.argmin(1)
        rows = np.arange(c.T)
        gk = np.zeros_like(c.ok[p])
        is_gk = (fxm[rows, k] < -L / 2 + BOX_DEPTH) & (np.abs(y[rows, k]) < BOX_HALF_W)
        gk[rows, k] = is_gk
        f[p + "goalside"] = _row_mean(marked, c.ok[p] & ~gk)

    # anchor: weighted centroid of the ANCHOR_K players closest to an opponent. Ties at the
    # K-th distance are all included, and the sums run in float64 before rounding to float32,
    # so the result does not depend on which team is listed first.
    dopp = np.concatenate([dopp_h, dopp_a], 1).astype(np.float64)
    pos = np.concatenate([hxy, axy], 1).astype(np.float64)
    kth = np.sort(dopp, 1)[:, min(ANCHOR_K, dopp.shape[1]) - 1]
    sel = np.isfinite(dopp) & (dopp <= kth[:, None])
    w = np.where(sel, 1.0 / (dopp + ANCHOR_EPS), 0.0)
    wsum = w.sum(1)
    pos0 = np.where(sel[:, :, None], pos, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        anchor = ((w[:, :, None] * pos0).sum(1) / wsum[:, None]).astype(np.float32)
    anchor[wsum <= 0] = np.nan
    has = ~np.isnan(anchor[:, 0])

    for p in ("h_", "a_"):
        rel = anchor[:, None, :] - c.xy[p]                       # player -> anchor
        da = np.sqrt((rel ** 2).sum(-1))
        f[p + "anchor_dmin"] = _row_min(da)
        f[p + "anchor_n5"] = np.where(has, (da <= ANCHOR_R).sum(1), np.nan).astype(np.float32)
        f[p + "anchor_fwd"] = (anchor[:, 0] * c.sign[p]).astype(np.float32)
        # closing speed: velocity component towards the anchor, mean over the 3 nearest players
        with np.errstate(invalid="ignore", divide="ignore"):
            u = rel / np.where(da > 1e-6, da, np.nan)[:, :, None]
        closing = (c.vel[p] * u).sum(-1)
        order = np.argsort(np.where(np.isnan(da), np.inf, da), 1, kind="stable")[:, :3]
        cl3 = np.take_along_axis(closing, order, 1)
        f[p + "anchor_close3"] = _row_mean(cl3, ~np.isnan(cl3))
    return f


def _shape_trends(c):
    f, lines = {}, {}
    for p in ("h_", "a_"):
        xy, ok = c.xy[p], c.ok[p]
        cx_fwd = _row_mean(xy[:, :, 0], ok) * c.sign[p]
        width = _row_std(xy[:, :, 1], ok)
        spread = width * _row_std(xy[:, :, 0], ok)
        for name, v in (("cx_fwd", cx_fwd), ("width", width), ("spread", spread)):
            f["%s%s_trend" % (p, name)] = _trend(v, 50, c.segs)    # next 2 s minus past 2 s

        # the two most advanced players, and the last outfield line (second deepest player,
        # the goalkeeper being the deepest) in the team's forward coordinate
        fwd = np.sort(np.where(ok, xy[:, :, 0] * c.sign[p][:, None], np.nan), axis=1)
        n = ok.sum(1)
        top = lambda k: np.take_along_axis(fwd, np.maximum(n - k, 0)[:, None], 1)[:, 0]
        f[p + "front2_fwd"] = np.where(n >= 3, 0.5 * (top(1) + top(2)), np.nan)
        lines[p] = np.where(n >= 3, fwd[:, 1], np.nan)
    for p in ("h_", "a_"):     # > 0: the front two stand beyond the opponent's last line
        f[p + "front_vs_opp_back"] = f[p + "front2_fwd"] + lines[c.opp[p]]
    return f


# -- entry point --------------------------------------------------------------------------------

def build_extra(m, ball=None):
    """Extra feature dict {name: float32[T]} for canonical match dict m.

    `ball` overrides m["ball"] (used to pass a degraded ball track), as in features.build.
    Deterministic and free of randomness; temporal context never crosses a half boundary,
    and every feature is NaN outside the graded halves (where the attacking direction, and so
    the goal-relative features, are undefined).
    """
    with np.errstate(invalid="ignore"):
        c = _Ctx(m, ball)
        f = {}
        f.update(_movement(c))
        f.update(_control(c))
        fb, last, nxt = _ball_gaps(c)
        f.update(fb)
        f.update(_set_pieces(c, last, nxt))
        f.update(_duels(c))
        f.update(_shape_trends(c))
    return {k: np.where(c.inseg, v, np.nan).astype(np.float32) for k, v in f.items()}
