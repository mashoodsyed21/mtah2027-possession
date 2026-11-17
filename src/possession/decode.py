"""Sequence decoding: turn noisy per-frame H/A/D probabilities into clean possession segments.

Pipeline (each step optional):

    logp = smooth_logits(p, temperature=..., window=..., runs=runs)      # calibrate / denoise
    labels = viterbi_hsmm_lite(logp, runs, params)                       # global decode
    labels = prune_short_segments(labels, runs, min_len, probs=p)        # safety net
    labels = refine_boundaries(labels, evidence, radius, runs=runs)      # snap to touches

Restarter-aware variant (4 states: H, A, dead-home-restarts DH, dead-away-restarts DA):

    logp4 = restarter_logp(p_dead, p_home_alive, p_restart_home)       # [T, 4]
    labels = viterbi_restarter(logp4, runs, params, cost_mod=...)       # D after DH/DA
    labels = shift_dead_edges(labels, runs, start_shift, end_shift)     # optional bias fix
    # or both in one call: decode_restarter(logp4, runs, params, cost_mod)

Label codes follow the official scorer: 0=H, 1=A, 2=D, 3=X. Everything temporal is done
independently inside each graded run [start, stop) (normally the two halves) and never across one;
frames outside the runs come back as X.
"""

import itertools
import math

import numpy as np
from numba import njit

H, A, D, X = 0, 1, 2, 3
N_CLASSES = 3

# Default decoder parameters. Costs are in nats (subtracted from the summed log-probabilities);
# lengths are in frames at 25 fps. The costs are a starting point found on the synthetic test data
# (tests/synthetic.py); they depend on how calibrated the frame model is, so tune them with
# tune_decoder on real validation sequences.
DEFAULT_PARAMS = {
    "cost_switch_team": 40.0,  # H <-> A directly (turnover with the ball alive)
    "cost_to_dead": 25.0,      # H/A -> D (ball goes out / whistle)
    "cost_from_dead": 25.0,    # D -> H/A (restart)
    "min_len_H": 50,           # minimum possession length, frames (2 s)
    "min_len_A": 50,
    "min_len_D": 125,          # minimum stoppage length, frames (5 s)
    # Optional geometric self-loop beyond the minimum: expected number of extra frames. None means
    # no per-frame duration cost at all (pure minimum-duration + switch-cost model).
    "mean_extra_H": None,
    "mean_extra_A": None,
    "mean_extra_D": None,
    # True: the first / last segment of a run may be shorter than its minimum (it is truncated by
    # the half boundary). False: the minimum is enforced there as well.
    "free_edges": True,
    "logp_floor": -30.0,       # clip log-probabilities so log(0) cannot dominate a path
}


# The restarter decoder takes every key above plus the cost of changing which team will restart
# during a stoppage. Kept below the other costs because restart evidence may legitimately change
# within a stoppage (the other team collects the ball, a decision is reversed). It also sets how
# much the restart evidence of a whole stoppage counts: a near-free change lets the path ignore
# everything but the stoppage's last frames, while +inf fixes one restarter per stoppage. On the
# synthetic tests (restart evidence constant within a stoppage) 30-100 scored best; tune it on
# real sequences.
RESTARTER_PARAMS = dict(DEFAULT_PARAMS, cost_restarter=10.0)

# Per-frame transition-cost modifiers understood by viterbi_restarter (see there).
COST_MOD_KEYS = ("switch_team", "to_dead", "from_dead")


def _resolve_params(params, defaults=None):
    """Merge user params over the defaults; accept min_len_HA / mean_extra_HA as shorthands."""
    out = dict(DEFAULT_PARAMS if defaults is None else defaults)
    params = dict(params or {})
    for short, (k1, k2) in (("min_len_HA", ("min_len_H", "min_len_A")),
                            ("mean_extra_HA", ("mean_extra_H", "mean_extra_A"))):
        if short in params:
            value = params.pop(short)
            out[k1] = out[k2] = value
    unknown = set(params) - set(out)
    if unknown:
        raise KeyError("unknown decoder params: {}".format(sorted(unknown)))
    out.update(params)
    return out


def _full_runs(n, runs):
    return [(0, n)] if runs is None else [(int(a), int(b)) for a, b in runs]


# -- smoothing helpers ---------------------------------------------------------------------------

def moving_average(p, window, runs=None):
    """Centred moving average of per-frame probabilities along time, inside each run only.

    window is in frames (rounded up to odd). Near a run edge the window shrinks to the frames that
    exist, so the output is still a proper average and never mixes the two halves.
    """
    p = np.asarray(p, dtype=np.float64)
    window = int(window)
    if window <= 1:
        return p.copy()
    half = window // 2
    out = p.copy()
    for start, stop in _full_runs(len(p), runs):
        seg = p[start:stop]
        n = len(seg)
        if n == 0:
            continue
        cum = np.concatenate([np.zeros((1,) + seg.shape[1:]), np.cumsum(seg, axis=0)], axis=0)
        idx = np.arange(n)
        lo = np.maximum(idx - half, 0)
        hi = np.minimum(idx + half + 1, n)
        count = (hi - lo).reshape((-1,) + (1,) * (seg.ndim - 1))
        out[start:stop] = (cum[hi] - cum[lo]) / count
    return out


def temperature_scale(p, temperature=1.0, eps=1e-6):
    """Sharpen (T < 1) or flatten (T > 1) probabilities: softmax(log p / T), row-normalised."""
    logp = np.log(np.clip(np.asarray(p, dtype=np.float64), eps, None)) / float(temperature)
    logp -= logp.max(axis=1, keepdims=True)
    q = np.exp(logp)
    return q / q.sum(axis=1, keepdims=True)


def smooth_logits(p, temperature=1.0, window=1, runs=None, eps=1e-6):
    """Probabilities [T, 3] -> log-probabilities [T, 3] ready for viterbi_hsmm_lite.

    Applies a centred moving average (``window`` frames, run-aware) and then temperature scaling.
    The temperature also sets the balance between evidence and the decoder's transition costs:
    a higher temperature makes every switch relatively more expensive.
    """
    q = moving_average(p, window, runs)
    q = temperature_scale(q, temperature, eps)
    return np.log(np.clip(q, eps, None))


# -- Viterbi with minimum durations --------------------------------------------------------------

@njit(cache=True)
def _viterbi_kernel(logp, k, trans, stay, exitc, free_edges):
    """Viterbi over the expanded state space of a 3-class explicit-minimum-duration HMM.

    Class c is a left-to-right chain of k[c] sub-states. Sub-state i < k-1 must advance to i+1;
    the last sub-state either stays (cost stay[c]) or leaves to sub-state 0 of another class d
    (cost exitc[c] + trans[c, d]). So every interior segment of class c lasts >= k[c] frames.
    All sub-states of class c emit logp[t, c].

    Backpointers are compact: only the entry decision (which class was left) and the last
    sub-state's stay-or-advance bit need storing; every other predecessor is implied by the chain.
    Memory is O(T * 3), time O(T * sum(k)).
    """
    T = logp.shape[0]
    C = 3
    off = np.zeros(C + 1, np.int64)
    for c in range(C):
        off[c + 1] = off[c] + k[c]
    S = off[C]
    neg = -np.inf

    prev = np.full(S, neg)
    cur = np.full(S, neg)
    entry_from = np.zeros((T, C), np.int8)     # class whose last sub-state preceded sub-state 0
    last_self = np.zeros((T, C), np.bool_)     # last sub-state came from itself (stay)

    # Initial frame. With free edges the half may begin mid-segment, i.e. in any sub-state.
    for c in range(C):
        for i in range(k[c]):
            if free_edges or i == 0:
                prev[off[c] + i] = logp[0, c]

    for t in range(1, T):
        for c in range(C):
            kc = k[c]
            o = off[c]
            e = logp[t, c]

            # Entry into sub-state 0 of class c.
            best = neg
            arg = (c + 1) % C   # placeholder; only used on unreachable states
            for d in range(C):
                last_d = prev[off[d] + k[d] - 1]
                if d == c:
                    if kc != 1:
                        continue
                    v = last_d - stay[c]          # k == 1: sub-state 0 is also the last one
                else:
                    v = last_d - exitc[d] - trans[d, c]
                if v > best:
                    best = v
                    arg = d
            cur[o] = best + e
            entry_from[t, c] = arg

            # Forced chain advance i-1 -> i for the middle sub-states.
            for i in range(1, kc - 1):
                cur[o + i] = prev[o + i - 1] + e

            # Last sub-state: advance from k-2 or stay.
            if kc >= 2:
                adv = prev[o + kc - 2]
                sty = prev[o + kc - 1] - stay[c]
                if sty > adv:
                    cur[o + kc - 1] = sty + e
                    last_self[t, c] = True
                else:
                    cur[o + kc - 1] = adv + e
        tmp = prev
        prev = cur
        cur = tmp

    # Final state: anywhere with free edges, otherwise the segment must have completed its minimum.
    best = neg
    bc = 0
    bi = 0
    for c in range(C):
        for i in range(k[c]):
            if not free_edges and i != k[c] - 1:
                continue
            v = prev[off[c] + i]
            if v > best:
                best = v
                bc = c
                bi = i

    # Backtrack.
    labels = np.empty(T, np.int8)
    c = bc
    i = bi
    for t in range(T - 1, 0, -1):
        labels[t] = c
        if i == 0:
            d = entry_from[t, c]
            c = d
            i = k[d] - 1
        elif i < k[c] - 1:
            i -= 1
        elif not last_self[t, c]:
            i -= 1
    labels[0] = c
    return labels


def _model_arrays(p):
    """Translate a resolved params dict into the kernel's arrays."""
    k = np.array([max(1, int(round(p["min_len_H"]))),
                  max(1, int(round(p["min_len_A"]))),
                  max(1, int(round(p["min_len_D"])))], dtype=np.int64)

    trans = np.zeros((3, 3), dtype=np.float64)
    trans[H, A] = trans[A, H] = p["cost_switch_team"]
    trans[H, D] = trans[A, D] = p["cost_to_dead"]
    trans[D, H] = trans[D, A] = p["cost_from_dead"]

    # Geometric tail: with mean m extra frames, P(stay) = m / (m + 1), P(leave) = 1 / (m + 1).
    stay = np.zeros(3, dtype=np.float64)
    exitc = np.zeros(3, dtype=np.float64)
    for c, key in enumerate(("mean_extra_H", "mean_extra_A", "mean_extra_D")):
        m = p[key]
        if m is not None and m > 0:
            stay[c] = -math.log(m / (m + 1.0))
            exitc[c] = -math.log(1.0 / (m + 1.0))
    return k, trans, stay, exitc


def viterbi_hsmm_lite(logp, runs=None, params=None):
    """Decode per-frame log-probabilities logp [T, 3] (columns H, A, D) into labels.

    runs: list of half-open [start, stop) graded spans, decoded independently (default: whole
    array). params: dict overriding DEFAULT_PARAMS (see there). Returns int8 labels of length T
    with codes 0/1/2 inside the runs and X (3) outside. Deterministic: ties resolve to the lowest
    class / sub-state index.
    """
    p = _resolve_params(params)
    logp = np.asarray(logp, dtype=np.float64)
    if logp.ndim != 2 or logp.shape[1] != N_CLASSES:
        raise ValueError("logp must be [T, 3], got {}".format(logp.shape))
    floor = float(p["logp_floor"])
    logp = np.nan_to_num(logp, nan=floor, neginf=floor, posinf=0.0)
    logp = np.ascontiguousarray(np.maximum(logp, floor))

    k, trans, stay, exitc = _model_arrays(p)
    out = np.full(len(logp), X, dtype=np.int8)
    for start, stop in _full_runs(len(logp), runs):
        if stop > start:
            out[start:stop] = _viterbi_kernel(logp[start:stop], k, trans, stay, exitc,
                                              bool(p["free_edges"]))
    return out


# -- restarter-aware Viterbi ---------------------------------------------------------------------
#
# Four states: H, A, DH (dead ball, home will restart) and DA (dead ball, away will restart).
#
#     H <-> A           cost_switch_team (+ cost_mod['switch_team'][t])
#     H/A -> DH or DA   cost_to_dead     (+ cost_mod['to_dead'][t])
#     DH -> H only      cost_from_dead   (+ cost_mod['from_dead'][t])
#     DA -> A only      cost_from_dead   (+ cost_mod['from_dead'][t])
#     DH <-> DA         cost_restarter   (inside a stoppage; does not restart its minimum length)
#
# So the team after a stoppage is always the team the stoppage said would restart, and evidence
# about the restarter collected during the whole stoppage counts toward that next possession. That
# matters most right after restarts, where the frame model's team evidence is weakest.

def restarter_logp(p_dead, p_home_alive, p_restart_home=None, eps=1e-6, confidence_bonus=0.0):
    """Per-frame log-probabilities [T, 4] (columns H, A, DH, DA) for viterbi_restarter.

    p_dead: P(ball dead) per frame. p_home_alive: P(home has the ball | ball alive), e.g.
    p_H / (p_H + p_A) from a 3-class frame model. p_restart_home: P(home takes the restart | ball
    dead); None means no information. NaN anywhere also means no information (0.5).

    H = log((1 - p_dead) p_home_alive), A = log((1 - p_dead)(1 - p_home_alive)). The restart
    probability p only decides between the two dead columns; it does not change how dead a frame
    looks. With m = max(p, 1 - p):

        DH = log p_dead + log(p / m),   DA = log p_dead + log((1 - p) / m)

    so the better restarter of every frame carries exactly the 3-class dead evidence and the other
    one pays log(min / max). Why not the calibrated ratio log(p / 0.5)? The decoder maximises over
    the restarter, so that ratio adds log(2 m) >= 0 to every frame's dead evidence: wherever the
    restart model is confident the ball looks *more* dead. A restart model applied to live frames
    is typically confident too (it points at the team on the ball), which then inflates stoppages
    -- on synthetic data with equally confident live-frame restart evidence the calibrated ratio
    cost 0.05-0.08 spellF1 against viterbi_hsmm_lite while this form gained 0.01-0.05.
    confidence_bonus = 1 restores the calibrated ratio (b * log(2 m) is added to both dead columns);
    use it only if restart-model confidence is genuinely evidence of a dead ball.

    An uninformative p = 0.5 leaves the dead evidence exactly as in the 3-class model for any
    confidence_bonus, and viterbi_restarter then decodes like viterbi_hsmm_lite with the same
    parameters. Returns float32.
    """
    def prob(x):
        x = np.asarray(x, dtype=np.float64)
        return np.clip(np.nan_to_num(x, nan=0.5), eps, 1.0 - eps)

    pd_ = prob(p_dead)
    ph = prob(p_home_alive)
    pr = np.full(pd_.shape, 0.5) if p_restart_home is None else prob(p_restart_home)
    if not (pd_.shape == ph.shape == pr.shape) or pd_.ndim != 1:
        raise ValueError("inputs must be 1-D arrays of equal length")
    out = np.empty((len(pd_), 4), dtype=np.float64)
    log_alive = np.log1p(-pd_)
    log_dead = np.log(pd_)
    out[:, H] = log_alive + np.log(ph)
    out[:, A] = log_alive + np.log1p(-ph)
    # log(1 - p) rather than log1p(-p): at p = 0.5 both logs are then bit-identical, so an
    # uninformative frame gives exactly equal DH and DA columns.
    log_rh, log_ra = np.log(pr), np.log(1.0 - pr)
    log_m = np.maximum(log_rh, log_ra)
    bonus = float(confidence_bonus) * (math.log(2.0) + log_m)
    out[:, 2] = log_dead + (log_rh - log_m) + bonus
    out[:, 3] = log_dead + (log_ra - log_m) + bonus
    return out.astype(np.float32)


@njit(cache=True)
def _fill_restarter(logp, a, b, c_end, cost_r, labels, states):
    """Recover the restarter path of dead frames [a, b] (inclusive) given states[b] == c_end.

    The forward pass does not store a backpointer for every dead sub-state (that would be
    O(T * min_len_D) memory); the restarter sequence inside the forced part of a stoppage is
    instead re-derived here with a 2-state Viterbi (DH / DA, switch cost cost_r). Inside one
    stoppage the chain position is fixed by the start frame and the entry costs of DH and DA are
    equal, so this reproduces the forward pass's choice, including its tie-breaking (prefer
    staying with the same restarter). The forward values carry the common entry score as an offset,
    so in an exact tie floating-point rounding may pick the other, equally good, restarter path;
    the labels and the state at frame b are unaffected.
    """
    n = b - a + 1
    came_other = np.zeros((n, 2), np.bool_)
    v0 = logp[a, 2]
    v1 = logp[a, 3]
    for j in range(1, n):
        t = a + j
        x0 = v1 - cost_r
        if x0 > v0:
            n0 = x0
            came_other[j, 0] = True
        else:
            n0 = v0
        x1 = v0 - cost_r
        if x1 > v1:
            n1 = x1
            came_other[j, 1] = True
        else:
            n1 = v1
        v0 = n0 + logp[t, 2]
        v1 = n1 + logp[t, 3]
    r = c_end - 2
    for j in range(n - 1, -1, -1):
        labels[a + j] = 2
        states[a + j] = 2 + r
        if j > 0 and came_other[j, r]:
            r = 1 - r


@njit(cache=True)
def _restarter_kernel(logp, k, base, stay, exitc, mod_sw, mod_td, mod_fd, free_edges):
    """Viterbi over the expanded state space of the 4-state restarter model.

    logp: [n, 4] (H, A, DH, DA). k: minimum lengths (H, A, D). base: (cost_switch_team,
    cost_to_dead, cost_from_dead, cost_restarter). stay / exitc: geometric-tail costs per class
    (H, A, D). mod_*: per-frame additions to the switch / to-dead / from-dead costs of a transition
    whose new segment starts at that frame.

    H and A are minimum-duration chains exactly as in _viterbi_kernel. The dead class is two
    parallel chains of k_D sub-states, DH and DA: entering from H/A starts sub-state 0 of either;
    sub-state i may advance to i + 1 of the same chain for free or of the other chain for
    cost_restarter, so changing the restarter never resets the stoppage's minimum length. The last
    dead sub-state stays (cost stay_D, plus cost_restarter when it changes chain) or exits DH -> H,
    DA -> A.

    Compact backpointers, O(n) memory: per frame the entry decision of each team chain, its last
    sub-state's stay bit, the (restarter-independent) entry decision of the dead chains, and the
    decision of each dead chain's last sub-state. The restarter path through the forced part of a
    stoppage is recovered afterwards by _fill_restarter.

    Returns (labels int8[n] with codes 0/1/2, states int8[n] with codes 0..3).
    """
    T = logp.shape[0]
    kk = np.empty(4, np.int64)
    kk[0] = k[0]
    kk[1] = k[1]
    kk[2] = k[2]
    kk[3] = k[2]
    kD = k[2]
    off = np.zeros(5, np.int64)
    for s in range(4):
        off[s + 1] = off[s] + kk[s]
    S = off[4]
    neg = -np.inf
    cost_sw = base[0]
    cost_td = base[1]
    cost_fd = base[2]
    cost_r = base[3]

    prev = np.full(S, neg)
    cur = np.full(S, neg)
    ha_entry = np.zeros((T, 2), np.int8)   # 0: stay (k == 1), 1: from the other team, 2: from dead
    ha_self = np.zeros((T, 2), np.bool_)   # team chain's last sub-state stayed
    d_entry = np.zeros(T, np.int8)         # dead sub-state 0 entered from 0: H, 1: A
    d_last = np.zeros((T, 2), np.int8)     # dead last sub-state: 0/1 entry from H/A (k_D == 1),
    #                                        2 stay, 3 switch restarter, 4 advance

    for s in range(4):
        for i in range(kk[s]):
            if free_edges or i == 0:
                prev[off[s] + i] = logp[0, s]

    for t in range(1, T):
        last_h = prev[off[1] - 1]
        last_a = prev[off[2] - 1]
        last_dh = prev[off[3] - 1]
        last_da = prev[off[4] - 1]

        # Team chains H (c = 0) and A (c = 1).
        for c in range(2):
            kc = kk[c]
            o = off[c]
            e = logp[t, c]
            other_last = last_a if c == 0 else last_h
            dead_last = last_dh if c == 0 else last_da
            best = neg
            arg = 1
            if kc == 1:
                v = prev[o] - stay[c]
                if v > best:
                    best = v
                    arg = 0
            v = other_last - exitc[1 - c] - cost_sw - mod_sw[t]
            if v > best:
                best = v
                arg = 1
            v = dead_last - exitc[2] - cost_fd - mod_fd[t]
            if v > best:
                best = v
                arg = 2
            cur[o] = best + e
            ha_entry[t, c] = arg
            for i in range(1, kc - 1):
                cur[o + i] = prev[o + i - 1] + e
            if kc >= 2:
                adv = prev[o + kc - 2]
                sty = prev[o + kc - 1] - stay[c]
                if sty > adv:
                    cur[o + kc - 1] = sty + e
                    ha_self[t, c] = True
                else:
                    cur[o + kc - 1] = adv + e

        # Entry into a stoppage: identical for DH and DA, so one decision serves both.
        v_h = last_h - exitc[0] - cost_td - mod_td[t]
        v_a = last_a - exitc[1] - cost_td - mod_td[t]
        if v_a > v_h:
            dent = v_a
            d_entry[t] = 1
        else:
            dent = v_h
            d_entry[t] = 0

        # Dead chains DH (r = 0) and DA (r = 1).
        for r in range(2):
            o = off[2 + r]
            q = off[3 - r]          # the other restarter's chain
            e = logp[t, 2 + r]
            if kD == 1:
                best = dent
                arg = d_entry[t]
                v = prev[o] - stay[2]
                if v > best:
                    best = v
                    arg = 2
                v = prev[q] - stay[2] - cost_r
                if v > best:
                    best = v
                    arg = 3
                cur[o] = best + e
                d_last[t, r] = arg
                continue
            cur[o] = dent + e
            for i in range(1, kD - 1):
                same = prev[o + i - 1]
                swap = prev[q + i - 1] - cost_r
                cur[o + i] = (same if same >= swap else swap) + e
            same = prev[o + kD - 2]
            swap = prev[q + kD - 2] - cost_r
            best = same if same >= swap else swap
            arg = 4
            v = prev[o + kD - 1] - stay[2]
            if v > best:
                best = v
                arg = 2
            v = prev[q + kD - 1] - stay[2] - cost_r
            if v > best:
                best = v
                arg = 3
            cur[o + kD - 1] = best + e
            d_last[t, r] = arg
        tmp = prev
        prev = cur
        cur = tmp

    # Final state: anywhere with free edges, otherwise a segment that completed its minimum.
    best = neg
    bc = 0
    bi = 0
    for s in range(4):
        for i in range(kk[s]):
            if not free_edges and i != kk[s] - 1:
                continue
            v = prev[off[s] + i]
            if v > best:
                best = v
                bc = s
                bi = i

    # Backtrack. Team frames step back one at a time; a stoppage's forced part is filled in one go.
    labels = np.empty(T, np.int8)
    states = np.empty(T, np.int8)
    c = bc
    i = bi
    t = T - 1
    while t >= 0:
        if c < 2:
            labels[t] = c
            states[t] = c
            if t == 0:
                break
            if i == 0:
                code = ha_entry[t, c]
                if code == 1:
                    c = 1 - c
                elif code == 2:
                    c = 2 + c           # H is entered only from DH, A only from DA
                i = kk[c] - 1
            elif i < kk[c] - 1 or not ha_self[t, c]:
                i -= 1
            t -= 1
            continue

        r = c - 2
        if i == kD - 1:
            labels[t] = 2
            states[t] = c
            if t == 0:
                break
            code = d_last[t, r]
            if code == 2:
                t -= 1
                continue
            if code == 3:
                c = 3 - r
                t -= 1
                continue
            if code < 2:                # k_D == 1: entered from H / A at this frame
                c = code
                i = kk[c] - 1
                t -= 1
                continue
            start = t - (kD - 1)        # advanced: the forced part began at frame `start`
        else:
            start = t - i               # a free-edge run ending inside the forced part
        _fill_restarter(logp, max(start, 0), t, c, cost_r, labels, states)
        if start <= 0:
            break
        c = d_entry[start]
        i = kk[c] - 1
        t = start - 1
    return labels, states


def _cost_mod_arrays(cost_mod, n):
    """cost_mod dict -> three float64[n] arrays (switch_team, to_dead, from_dead), zeros if absent.

    Values may be scalars or length-n arrays. NaN counts as 0; +inf forbids the transition at that
    frame; -inf is rejected (it would make a path infinitely good).
    """
    cost_mod = dict(cost_mod or {})
    unknown = set(cost_mod) - set(COST_MOD_KEYS)
    if unknown:
        raise KeyError("unknown cost_mod keys: {}".format(sorted(unknown)))
    out = []
    for key in COST_MOD_KEYS:
        value = cost_mod.get(key)
        if value is None:
            out.append(np.zeros(n))
            continue
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            arr = np.full(n, float(arr))
        if arr.shape != (n,):
            raise ValueError("cost_mod['{}'] must have length {}, got {}".format(
                key, n, arr.shape))
        if np.any(arr == -np.inf):
            raise ValueError("cost_mod['{}'] contains -inf".format(key))
        out.append(np.ascontiguousarray(np.nan_to_num(arr, nan=0.0, posinf=np.inf)))
    return out


def viterbi_restarter(logp, runs=None, params=None, cost_mod=None, return_states=False):
    """Decode [T, 4] log-probabilities (H, A, DH, DA) with the restarter-aware model.

    A stoppage carries which team will restart it, and the possession after it must belong to
    that team (DH -> H, DA -> A). Minimum durations apply per class: min_len_H / min_len_A (or the
    shorthand min_len_HA) and min_len_D, shared by DH and DA -- a stoppage is one segment however
    often its restarter changes. Costs and lengths come from RESTARTER_PARAMS (all of
    DEFAULT_PARAMS plus cost_restarter for a DH <-> DA change).

    cost_mod: optional {'switch_team' | 'to_dead' | 'from_dead': float[T] or scalar} added to the
    corresponding transition cost when the new segment starts at frame t, e.g. a higher
    switch_team cost while the ball is missing. +inf forbids that transition at that frame.

    runs: half-open graded spans decoded independently (default: the whole array). Returns int8
    labels (0 H, 1 A, 2 D, 3 X outside the runs); with return_states=True also the int8 state path
    (0 H, 1 A, 2 DH, 3 DA, -1 outside the runs). Deterministic.
    """
    p = _resolve_params(params, RESTARTER_PARAMS)
    logp = np.asarray(logp, dtype=np.float64)
    if logp.ndim != 2 or logp.shape[1] != 4:
        raise ValueError("logp must be [T, 4], got {}".format(logp.shape))
    floor = float(p["logp_floor"])
    logp = np.nan_to_num(logp, nan=floor, neginf=floor, posinf=0.0)
    logp = np.ascontiguousarray(np.maximum(logp, floor))
    n = len(logp)
    mod_sw, mod_td, mod_fd = _cost_mod_arrays(cost_mod, n)

    k, _, stay, exitc = _model_arrays(p)
    base = np.array([p["cost_switch_team"], p["cost_to_dead"], p["cost_from_dead"],
                     p["cost_restarter"]], dtype=np.float64)
    labels = np.full(n, X, dtype=np.int8)
    states = np.full(n, -1, dtype=np.int8)
    for start, stop in _full_runs(n, runs):
        if stop > start:
            lab, st = _restarter_kernel(logp[start:stop], k, base, stay, exitc,
                                        mod_sw[start:stop], mod_td[start:stop],
                                        mod_fd[start:stop], bool(p["free_edges"]))
            labels[start:stop] = lab
            states[start:stop] = st
    return (labels, states) if return_states else labels


# Post-decoding keys understood by decode_restarter / tune_restarter (frames, see shift_dead_edges).
SHIFT_KEYS = ("start_shift", "end_shift")


def decode_restarter(logp, runs=None, params=None, cost_mod=None):
    """viterbi_restarter followed by shift_dead_edges, from one params dict.

    params may hold start_shift / end_shift (frames) next to the RESTARTER_PARAMS keys, so the
    best params of tune_restarter can be used here unchanged.
    """
    params = dict(params or {})
    shifts = [params.pop(key, 0) or 0 for key in SHIFT_KEYS]
    labels = viterbi_restarter(logp, runs, params, cost_mod)
    if any(shifts):
        labels = shift_dead_edges(labels, runs, *shifts)
    return labels


def shift_dead_edges(labels, runs=None, start_shift=0, end_shift=0):
    """Move stoppage boundaries by a fixed number of frames to correct a systematic timing bias.

    Every possession -> D boundary moves earlier by start_shift frames, every D -> possession
    boundary earlier by end_shift frames (negative values move them later). H <-> A boundaries,
    run edges and X frames never move. Each move is capped so no segment -- possession or
    stoppage -- loses more than half of its length (when both of its ends would cut into it, the
    allowance is split in proportion to the two requests), so the sequence of segments and their
    labels never changes.
    """
    labels = np.asarray(labels).astype(np.int8)
    out = labels.copy()
    start_shift, end_shift = int(round(start_shift)), int(round(end_shift))
    if start_shift == 0 and end_shift == 0:
        return out
    for start, stop in _full_runs(len(labels), runs):
        lab = labels[start:stop]
        n = len(lab)
        bounds = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        if not len(bounds):
            continue
        seg_start = np.concatenate(([0], bounds))
        seg_lab = lab[seg_start]
        seg_len = np.diff(np.concatenate((seg_start, [n])))
        left, right = seg_lab[:-1], seg_lab[1:]
        alive_l, alive_r = (left == H) | (left == A), (right == H) | (right == A)

        # Signed request per boundary: > 0 moves it earlier (eats the end of the left segment),
        # < 0 moves it later (eats the start of the right segment).
        want = np.zeros(len(bounds), dtype=np.int64)
        want[alive_l & (right == D)] = start_shift
        want[(left == D) & alive_r] = end_shift

        # Frames each segment is asked to give up at its front and at its back, then the cap.
        front = np.concatenate(([0], np.maximum(-want, 0)))
        back = np.concatenate((np.maximum(want, 0), [0]))
        allow = seg_len // 2
        total = front + back
        over = total > allow
        denom = np.maximum(total, 1)
        front = np.where(over, front * allow // denom, front)
        back = np.where(over, back * allow // denom, back)

        moved = bounds - np.where(want > 0, back[:-1], 0) + np.where(want < 0, front[1:], 0)
        lengths = np.diff(np.concatenate(([0], moved, [n])))
        out[start:stop] = np.repeat(seg_lab, lengths)
    return out


# -- post-processing: prune short segments -------------------------------------------------------

def _min_len_array(min_len):
    """int | (HA, D) | (H, A, D) | {'H':..,'A':..,'D':..} -> int64[4] (X never pruned)."""
    if isinstance(min_len, dict):
        vals = [min_len.get("H", 0), min_len.get("A", 0), min_len.get("D", 0)]
    elif np.ndim(min_len) == 0:
        vals = [min_len] * 3
    elif len(min_len) == 2:
        vals = [min_len[0], min_len[0], min_len[1]]
    else:
        vals = list(min_len)[:3]
    return np.array([int(v) for v in vals] + [0], dtype=np.int64)


@njit(cache=True)
def _prune_kernel(lab, cum, min_len, use_probs):
    """Absorb short segments of one run, shortest first, using a doubly linked segment list.

    lab: int8 labels (modified copy returned). cum: [n+1, 4] cumulative class probabilities.
    Thresholds grow L = 1, 2, ... so flicker is merged away before longer short segments are
    judged; at each L passes repeat until stable. Every absorption merges at least two segments,
    so the loop terminates.
    """
    n = lab.shape[0]
    nseg = 1
    for t in range(1, n):
        if lab[t] != lab[t - 1]:
            nseg += 1
    st = np.empty(nseg, np.int64)
    en = np.empty(nseg, np.int64)
    lb = np.empty(nseg, np.int64)
    prv = np.empty(nseg, np.int64)
    nxt = np.empty(nseg, np.int64)
    j = 0
    st[0] = 0
    for t in range(1, n):
        if lab[t] != lab[t - 1]:
            en[j] = t
            j += 1
            st[j] = t
    en[j] = n
    for j in range(nseg):
        lb[j] = lab[st[j]]
        prv[j] = j - 1
        nxt[j] = j + 1 if j + 1 < nseg else -1

    maxmin = 0
    for c in range(min_len.shape[0]):
        if min_len[c] > maxmin:
            maxmin = min_len[c]

    for L in range(1, maxmin):
        while True:
            changed = False
            s = 0  # the head segment is never removed: merges always keep the left index
            while s != -1:
                length = en[s] - st[s]
                p = prv[s]
                q = nxt[s]
                if length <= L and length < min_len[lb[s]] and (p != -1 or q != -1):
                    # Pick the neighbour to join.
                    if p == -1:
                        target = lb[q]
                    elif q == -1:
                        target = lb[p]
                    elif lb[p] == lb[q]:
                        target = lb[p]
                    else:
                        left_score = 0.0
                        right_score = 0.0
                        if use_probs:
                            left_score = cum[en[s], lb[p]] - cum[st[s], lb[p]]
                            right_score = cum[en[s], lb[q]] - cum[st[s], lb[q]]
                        if left_score > right_score:
                            target = lb[p]
                        elif right_score > left_score:
                            target = lb[q]
                        elif en[q] - st[q] > en[p] - st[p]:   # tie: longer neighbour, then left
                            target = lb[q]
                        else:
                            target = lb[p]
                    lb[s] = target
                    keep = s
                    if p != -1 and lb[p] == lb[s]:          # merge into the left neighbour
                        en[p] = en[s]
                        nxt[p] = q
                        if q != -1:
                            prv[q] = p
                        keep = p
                    q = nxt[keep]
                    if q != -1 and lb[q] == lb[keep]:       # swallow the right neighbour
                        en[keep] = en[q]
                        nxt[keep] = nxt[q]
                        if nxt[q] != -1:
                            prv[nxt[q]] = keep
                    changed = True
                    s = nxt[keep]
                else:
                    s = q
            if not changed:
                break

    out = lab.copy()
    s = 0
    while s != -1:
        for t in range(st[s], en[s]):
            out[t] = lb[s]
        s = nxt[s]
    return out


def prune_short_segments(labels, runs=None, min_len=25, probs=None):
    """Absorb every segment shorter than its class minimum into a neighbour, until stable.

    min_len: int (all classes), (HA, D), (H, A, D) or {'H':..,'A':..,'D':..}, in frames.
    probs: optional [T, 3] per-frame probabilities; a short segment joins the neighbour whose class
    has the higher mean probability over the short segment's frames. Without probs (or on a tie)
    it joins the longer neighbour. A segment spanning a whole run is left alone.
    """
    labels = np.asarray(labels).astype(np.int8)
    out = labels.copy()
    mins = _min_len_array(min_len)
    use_probs = probs is not None
    for start, stop in _full_runs(len(labels), runs):
        if stop <= start:
            continue
        cum = np.zeros((stop - start + 1, 4), dtype=np.float64)
        if use_probs:
            cum[1:, :3] = np.cumsum(np.asarray(probs, dtype=np.float64)[start:stop, :3], axis=0)
        out[start:stop] = _prune_kernel(np.ascontiguousarray(labels[start:stop]), cum, mins,
                                        use_probs)
    return out


# -- post-processing: boundary refinement --------------------------------------------------------

def refine_boundaries(labels, evidence, radius, runs=None, min_seg=1, kinds="all"):
    """Move each boundary within +/- radius frames to the argmax of a boundary-evidence signal.

    evidence[t] should be high where a new segment plausibly starts at frame t (e.g. touch
    strength). Boundaries are processed left to right; each one is confined between the already
    moved previous boundary and the original next boundary (with ``min_seg`` frames kept for every
    segment), so segment order and labels never change -- only where they meet. Ties go to the
    position closest to the original boundary.

    kinds: "all", "team" (H<->A only) or "dead" (any boundary involving D).
    """
    labels = np.asarray(labels).astype(np.int8)
    evidence = np.nan_to_num(np.asarray(evidence, dtype=np.float64), nan=-np.inf)
    radius = int(radius)
    min_seg = max(1, int(min_seg))
    out = labels.copy()
    for start, stop in _full_runs(len(labels), runs):
        lab = labels[start:stop]
        ev = evidence[start:stop]
        n = len(lab)
        if n < 2:
            continue
        bounds = np.flatnonzero(lab[1:] != lab[:-1]) + 1
        if not len(bounds):
            continue
        seg_labels = lab[np.concatenate(([0], bounds))]
        new = bounds.copy()
        for i, b in enumerate(bounds):
            left, right = int(lab[b - 1]), int(lab[b])
            is_team = {left, right} == {H, A}
            if (kinds == "team" and not is_team) or (kinds == "dead" and is_team):
                continue
            lo = max(b - radius, (new[i - 1] if i > 0 else 0) + min_seg)
            hi = min(b + radius, (bounds[i + 1] if i + 1 < len(bounds) else n) - min_seg)
            if lo > hi:
                continue
            window = ev[lo:hi + 1]
            peak = window.max()
            if not np.isfinite(peak):
                continue
            cands = np.flatnonzero(window == peak) + lo
            new[i] = cands[np.argmin(np.abs(cands - b))]  # argmin returns the earliest on ties
        lengths = np.diff(np.concatenate(([0], new, [n])))
        out[start:stop] = np.repeat(seg_labels, lengths)
    return out


# -- grid search ---------------------------------------------------------------------------------

def tune_decoder(sequences, grid, base_params=None, postprocess=None, verbose=False):
    """Grid-search decoder params to maximise the mean official spellF1 over sequences.

    sequences: iterable of (gt_codes, logp, runs); runs may be None (then taken from gt via the
    official valid_runs). grid: {param_name: [values, ...]} over DEFAULT_PARAMS keys (shorthands
    min_len_HA / mean_extra_HA allowed). postprocess: optional callable
    f(labels, runs, logp, params) -> labels applied after Viterbi.

    Returns (best_params, table) where table is a pandas DataFrame, one row per combination,
    sorted by mean spell_f1_mean, then mean macro_f1 (the official tie-break).
    """
    seqs = [(gt, logp, runs, None) for gt, logp, runs in sequences]

    def decode(logp, runs, cost_mod, params):
        pred = viterbi_hsmm_lite(logp, runs, params)
        if postprocess is not None:
            pred = postprocess(pred, runs, logp, params)
        return pred

    return _grid_search(seqs, grid, base_params, decode, verbose)


def tune_restarter(sequences, grid, base_params=None, postprocess=None, verbose=False):
    """Grid-search the restarter decoder (decode_restarter) to maximise mean official spellF1.

    sequences: iterable of (gt_codes, logp4, runs) or (gt_codes, logp4, runs, cost_mod); runs may
    be None (taken from gt via the official valid_runs). grid: {param_name: [values, ...]} over
    RESTARTER_PARAMS keys (shorthands allowed) plus start_shift / end_shift (frames, applied with
    shift_dead_edges after decoding). postprocess: optional f(labels, runs, logp4, params) -> labels
    applied last.

    Returns (best_params, table) exactly like tune_decoder; best_params plugs straight into
    decode_restarter.
    """
    seqs = []
    for seq in sequences:
        gt, logp, runs = seq[:3]
        seqs.append((gt, logp, runs, seq[3] if len(seq) > 3 else None))

    def decode(logp, runs, cost_mod, params):
        pred = decode_restarter(logp, runs, params, cost_mod)
        if postprocess is not None:
            pred = postprocess(pred, runs, logp, params)
        return pred

    return _grid_search(seqs, grid, base_params, decode, verbose)


def _grid_search(sequences, grid, base_params, decode, verbose):
    """Shared loop of tune_decoder / tune_restarter: score every grid point with the official
    scorer. decode(logp, runs, cost_mod, params) -> labels."""
    import pandas as pd
    from . import metrics

    seqs = []
    for gt, logp, runs, cost_mod in sequences:
        gt = np.asarray(gt).astype(np.int8)
        seqs.append((gt, np.asarray(logp, dtype=np.float64),
                     metrics.valid_runs(gt) if runs is None else runs, cost_mod))

    names = list(grid)
    rows = []
    for values in itertools.product(*(grid[name] for name in names)):
        params = dict(base_params or {})
        params.update(zip(names, values))
        spell, macro, ratio = [], [], []
        for gt, logp, runs, cost_mod in seqs:
            pred = decode(logp, runs, cost_mod, params)
            s = metrics.score(gt, pred)
            spell.append(s["spell_f1_mean"])
            macro.append(s["macro_f1"])
            ratio.append(s["n_pred_spells"] / max(1, s["n_gt_spells"]))
        row = dict(zip(names, values))
        row.update(spell_f1_mean=float(np.mean(spell)), macro_f1=float(np.mean(macro)),
                   spell_count_ratio=float(np.mean(ratio)))
        rows.append(row)
        if verbose:
            print(row)

    table = pd.DataFrame(rows).sort_values(["spell_f1_mean", "macro_f1"], ascending=False,
                                           kind="mergesort").reset_index(drop=True)
    best = dict(base_params or {})
    best.update({name: table.loc[0, name] for name in names})
    # pandas turns ints into numpy scalars and None into NaN; hand back plain Python values.
    for name in names:
        value = best[name]
        if isinstance(value, float) and math.isnan(value):
            best[name] = None
        elif hasattr(value, "item"):
            best[name] = value.item()
    return best, table
