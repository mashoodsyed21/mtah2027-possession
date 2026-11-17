"""Metrica Sports open sample games -> canonical format, frame labels derived from events.

Source: https://github.com/metrica-sports/sample-data (three anonymised matches, 25 fps,
synchronised tracking and event data). Games 1 and 2 ship as one tracking CSV per team plus an
event CSV; game 3 ships as FIFA EPTS (metadata XML + text tracking) plus a JSON event file. These
are the same URLs ``kloppy.metrica.load_open_data`` reads; we cache the raw files and parse them
directly, because kloppy has no loader for the CSV events of games 1-2 and its event model drops
the subtypes the rules below need (goal kicks, fouls, advantage, ...). ``scripts/build_metrica.py
--verify N`` cross-checks the tracking parse against kloppy's loaders.

The tracking is the automatic Metrica product, i.e. the same provider family as the competition
target, so these games serve as an extra validation domain. Output per game (see dfl.py):

    home   float32 [T, Nh, 2]  player x, y in metres, pitch-centred, NaN = not on the pitch
    away   float32 [T, Na, 2]
    ball   float32 [T, 2]      raw tracked ball, NaN where Metrica has no ball (kept as is)
    period int8    [T]         1 / 2
    label  int8    [T]         0 home possession, 1 away, 2 dead ball, 3 unknown
    pitch  float32 [2]         (105, 68)

Row i is tracking frame ``frame0 + i`` (frame0 = 1 for all three games). Event frames refer to the
same numbering.

The raw ball is NOT a realistic tracked ball and must be degraded before these games are used to
train or validate anything that sees the ball:

* it is NaN on almost exactly the dead-ball frames (visible on ~99% of live and <15% of dead
  frames, against 92% / 66% in the competition tracking), and its gaps start on the frame after
  the stopping event's end frame and end on the restart frame;
* it is piecewise linear with its knots on the event frames: between two tagged touches it moves
  in a straight line at constant speed (no deceleration, no noise), and ~98% of PASS start frames
  carry a direction change within one frame. Touches - and with them the possession switches the
  labels are derived from - are therefore timed exactly by the ball path;
* inside a few stoppages it is a constant-velocity straight-line drift, often ending in a jump
  on the restart frame (game 1 rows ~38480-42475: 160 s at 0.09 m/s, then 14 m in one frame;
  game 2 rows ~7890-8146: 10 s at 2.9 m/s, then 28 m; game 2 rows ~21200-21636: 17 s at
  0.24 m/s), a dead-ball tell that no real tracker produces.

Coordinates
-----------
Metrica's normalised frame has the origin at the top-left corner, x along the touchline (0..1),
y downwards (0..1). We map x_m = x * 105 - 52.5 and y_m = y * 68 - 34, i.e. exactly the conversion
``target.py`` applies to the competition tracking (no y flip), so both Metrica sources share one
convention. Teams keep their raw ends (they swap at half time), as in the DFL data. Home / away come
from the files: the per-team CSVs of games 1-2, ``Score/@idLocalTeam`` of the game 3 metadata.

Event vocabulary (types and subtypes as Metrica writes them; JSON subtype lists are joined with
"-" so both formats read alike, e.g. ["GROUND", "LOST"] -> "GROUND-LOST")
---------------------------------------------------------------------------------------------
* restart   : type SET PIECE (kick off, throw in, corner kick, free kick, penalty, incl. "-RETAKEN"),
              or any event whose subtype starts with "GOAL KICK" (Metrica tags goal kicks as
              PASS / BALL LOST / BALL OUT with subtype GOAL KICK..., never as SET PIECE).
* on-ball   : PASS, CARRY, SHOT, RECOVERY, SET PIECE, every restart, and CHALLENGE whose subtype
              contains WON (aerial / ground / tackle / dribble won, also advantage-won) unless
              it is a foul challenge (FAULT without ADVANTAGE): being fouled is not control of
              the ball, the fouled team gets it with the free kick. These are the only events
              that can give a team possession.
* ball event: every event except CARD (cards are booked during the stoppage). RECOVERY with
              subtype GOAL (keeper fetching the ball from the net) is also excluded. Ball events
              are the evidence that the ball was still being played.
* end of half: BALL LOST whose subtype contains END HALF.
* out of play: BALL OUT, SHOT whose subtype contains OUT or GOAL, PASS / BALL LOST with OFFSIDE,
              BALL LOST with HAND BALL.
* stopping  : out of play, a foul (FAULT RECEIVED, or CHALLENGE with FAULT but not ADVANTAGE),
              BALL LOST with REFEREE HIT, end of half.

Label rules (applied per period, in this order)
-----------------------------------------------
1. Default: unknown (3).
2. Possession: every on-ball event sets the owner to its team from its start frame on; the owner
   is carried forward until an on-ball event of the other team starts. So possession switches at
   the start frame of the first on-ball event of the other team. BALL LOST, BALL OUT, lost /
   neutral challenges and FAULT RECEIVED never switch possession: a team that loses the ball
   keeps it until the opponent shows control (an uncontrolled touch / loose ball does not change
   possession). Several on-ball events on the same frame: the last one in the file wins. Frames
   of a period before its first on-ball event have no owner (3).

An event's end frame is its last live frame: Metrica's own tracking still shows the ball on the
end frame of the event that stops play and drops it from the next frame on (for ~90% of the
stoppages exactly so), so every stoppage in rules 3-5 starts on the frame AFTER an end frame.

3. Dead ball before every restart R (start frame f_R): dead from s_R to f_R - 1. Candidates are
   the ball events of the same period from the previous restart on that start more than
   ``restart_tol_s`` (0.5 s) before f_R; events closer to f_R are tags of the restart itself
   (the keeper's pseudo RECOVERY that goes with every goal kick, touches logged a frame out of
   order). s_R follows the end frame of the last stopping event among them (frame the ball
   leaves the pitch, the shot's end, the foul, the offside pass, ...). If the tagger logged
   further touches after it - play ran on after a foul until the whistle - s_R moves past the
   end of the last of them, so no tagged touch is ever labelled dead; RECOVERY events do not
   count here, because players picking the ball up during a stoppage are tagged as recoveries.
   Without any stopping event s_R follows the end of the last non-RECOVERY candidate, and with
   no candidate at all it is the start of the period. Consequences: the time before the kick-off
   of each half is dead, and after a goal the ball is dead until the kick-off.
4. Stoppage without a tagged restart: if an out-of-play event is followed by at least
   ``MISSING_RESTART_S`` seconds without any non-RECOVERY ball event and that next event is not a
   restart, the frames after its end frame up to that next event are dead as well (a restart the
   tagger did not log; recoveries in between are players picking the ball up). Fouls are left
   out here because advantage is not always tagged.
5. End of half: after the end frame of a BALL LOST ... END HALF event the ball is dead to the end
   of the period. If a period has no such event, frames after the end frame of its last ball event
   are unknown (3) - the whistle time is not observable from the events.
6. Long silences: alive frames more than ``SILENCE_S`` seconds after the end of the last ball event
   and more than ``SILENCE_S`` before the next one are unknown (3): with no tagged touch for that
   long the ball state cannot be determined from the events (typically an untagged stoppage).

Everything is vectorised over frames; only the event list (a few thousand rows) is iterated.
"""

from __future__ import annotations

import io
import json
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FPS = 25
PITCH = (105.0, 68.0)
GAMES = (1, 2, 3)
HOME, AWAY, DEAD, UNKNOWN = 0, 1, 2, 3

BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
RAW_FILES = {
    1: ["Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
        "Sample_Game_1/Sample_Game_1_RawEventsData.csv"],
    2: ["Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
        "Sample_Game_2/Sample_Game_2_RawEventsData.csv"],
    3: ["Sample_Game_3/Sample_Game_3_metadata.xml",
        "Sample_Game_3/Sample_Game_3_tracking.txt",
        "Sample_Game_3/Sample_Game_3_events.json"],
}

MISSING_RESTART_S = 5.0   # rule 4: silence after a stopping event that implies an untagged restart
SILENCE_S = 20.0          # rule 6: half-width of an event-free stretch that becomes unknown

NON_BALL_TYPES = {"CARD"}
ON_BALL_TYPES = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def download_raw(raw_dir: Path, games=GAMES) -> None:
    """Fetch the raw sample files of `games` into raw_dir (files already present are kept)."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    for g in games:
        for rel in RAW_FILES[g]:
            target = raw_dir / Path(rel).name
            if target.exists() and target.stat().st_size > 0:
                continue
            print(f"downloading {rel}", flush=True)
            tmp = target.with_suffix(target.suffix + ".part")
            with urllib.request.urlopen(BASE_URL + rel, timeout=120) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            tmp.replace(target)


def raw_path(raw_dir: Path, game: int, kind: str) -> Path:
    """Path of one raw file: kind in {'home', 'away', 'events'} (games 1-2) or
    {'metadata', 'tracking', 'events'} (game 3)."""
    order = {1: ("home", "away", "events"), 2: ("home", "away", "events"),
             3: ("metadata", "tracking", "events")}[game]
    return Path(raw_dir) / Path(RAW_FILES[game][order.index(kind)]).name


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

def to_metres(xy: np.ndarray, pitch=PITCH) -> np.ndarray:
    """Metrica normalised coordinates (origin top-left, 0..1) -> pitch-centred metres.

    Same mapping as target.py (x * L - L/2, y * W - W/2, no axis flip). NaN stays NaN.
    """
    xy = np.asarray(xy, np.float64)
    out = np.empty(xy.shape, np.float32)
    out[..., 0] = xy[..., 0] * pitch[0] - pitch[0] / 2
    out[..., 1] = xy[..., 1] * pitch[1] - pitch[1] / 2
    return out


def _read_team_csv(path: Path) -> tuple[pd.DataFrame, list[str]]:
    """One Metrica tracking CSV -> (numeric table, player names in column order).

    Layout: two banner rows (team, jersey), then a header 'Period,Frame,Time [s],PlayerA,,PlayerB,,
    ...,Ball,' where every object owns an x column and an unnamed y column.
    """
    with open(path, encoding="utf-8") as fh:
        for _ in range(2):
            next(fh)
        header = next(fh).rstrip("\n").split(",")
    names = [h for h in header[3:] if h]
    assert names[-1] == "Ball", f"unexpected header in {path}"
    cols = ["period", "frame", "time"] + [f"{n}_{c}" for n in names for c in "xy"]
    df = pd.read_csv(path, skiprows=3, header=None, names=cols, usecols=range(len(cols)),
                     dtype=np.float64, na_values=["NaN"])
    return df, names[:-1]


def load_tracking_csv(home_path: Path, away_path: Path) -> dict:
    """Games 1-2: per-team tracking CSVs -> dict(home, away, ball, period, frame, player ids)."""
    hd, hn = _read_team_csv(home_path)
    ad, an = _read_team_csv(away_path)
    assert len(hd) == len(ad) and np.array_equal(hd["frame"].to_numpy(), ad["frame"].to_numpy()), \
        "home and away tracking files cover different frames"
    assert np.array_equal(hd["period"].to_numpy(), ad["period"].to_numpy(), equal_nan=True)
    # Both files carry the ball; they must agree wherever both have it.
    hb = hd[["Ball_x", "Ball_y"]].to_numpy()
    ab = ad[["Ball_x", "Ball_y"]].to_numpy()
    both = ~np.isnan(hb[:, 0]) & ~np.isnan(ab[:, 0])
    assert np.allclose(hb[both], ab[both], atol=1e-6), "ball differs between home and away files"
    ball = np.where(np.isnan(hb), ab, hb)

    def players(df, names):
        a = np.stack([df[[f"{n}_x", f"{n}_y"]].to_numpy() for n in names], axis=1)
        return to_metres(a)

    return {
        "frame": hd["frame"].to_numpy().astype(np.int64),
        "period": np.nan_to_num(hd["period"].to_numpy(), nan=0).astype(np.int8),
        "home": players(hd, hn), "away": players(ad, an), "ball": to_metres(ball),
        "home_ids": hn, "away_ids": an,
    }


@dataclass
class EptsMeta:
    home_team_id: str
    away_team_id: str
    team_of: dict           # player id -> team id
    fps: float
    pitch: tuple
    periods: dict           # period -> (first frame, last frame), inclusive
    specs: list             # [(start frame, end frame, [player id per slot])]


def parse_epts_metadata(path: Path) -> EptsMeta:
    """FIFA EPTS metadata (game 3): teams, players, periods and the per-range slot layout.

    The data line layout changes with every substitution, so the file has one
    DataFormatSpecification per frame range; each lists the player channel of every slot.
    """
    root = ET.parse(path).getroot()
    score = root.find(".//Score")
    home_id, away_id = score.get("idLocalTeam"), score.get("idVisitingTeam")
    team_of = {p.get("id"): p.get("teamId") for p in root.iter("Player")}
    channel_player = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    params = {p.findtext("Name"): p.findtext("Value") for p in root.find(".//GlobalConfig").iter("ProviderParameter")}
    periods = {}
    for k, name in ((1, "first_half"), (2, "second_half")):
        if params.get(f"{name}_start"):
            periods[k] = (int(params[f"{name}_start"]), int(params[f"{name}_end"]))
    specs = []
    for ds in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in ds.iter("PlayerChannelRef")]
        xs, ys = refs[0::2], refs[1::2]
        slots = []
        for cx, cy in zip(xs, ys):
            px, py = channel_player[cx], channel_player[cy]
            assert px == py and cx.endswith("_x") and cy.endswith("_y"), f"bad channel pair {cx}/{cy}"
            slots.append(px)
        specs.append((int(ds.get("startFrame")), int(ds.get("endFrame")), slots))
    field = root.find(".//FieldSize")
    return EptsMeta(home_id, away_id, team_of, float(root.findtext(".//FrameRate")),
                    (float(field.findtext("Width")), float(field.findtext("Height"))), periods, specs)


def load_tracking_epts(meta_path: Path, data_path: Path) -> dict:
    """Game 3: EPTS text tracking -> dict(home, away, ball, period, frame, player ids).

    Each line is 'frame:x,y;x,y;...;x,y:bx,by' with a fixed number of player slots, so the file
    is read as one flat CSV after turning ':' and ';' into ','. Every player gets one column per
    team (ordered by first slot appearance); a slot's values go to the player that the active
    DataFormatSpecification assigns to it.
    """
    meta = parse_epts_metadata(meta_path)
    assert meta.fps == FPS
    text = Path(data_path).read_text(encoding="ascii").replace(":", ",").replace(";", ",")
    raw = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64, na_values=["NaN"]).to_numpy()
    n_slots = (raw.shape[1] - 3) // 2
    frame = raw[:, 0].astype(np.int64)
    assert np.all(np.diff(frame) == 1), "EPTS tracking is not frame-contiguous"
    slot_xy = raw[:, 1:1 + 2 * n_slots].reshape(len(raw), n_slots, 2)
    ball = to_metres(raw[:, -2:])

    order = []                                  # players by first appearance
    for _, _, slots in meta.specs:
        assert len(slots) == n_slots
        order += [p for p in slots if p not in order]
    ids = {t: [p for p in order if meta.team_of[p] == tid]
           for t, tid in (("home", meta.home_team_id), ("away", meta.away_team_id))}
    out = {t: np.full((len(raw), len(ids[t]), 2), np.nan, np.float32) for t in ("home", "away")}
    for a, b, slots in meta.specs:
        rows = (frame >= a) & (frame <= b)
        for s, pid in enumerate(slots):
            t = "home" if meta.team_of[pid] == meta.home_team_id else "away"
            out[t][rows, ids[t].index(pid)] = to_metres(slot_xy[rows, s])
    period = np.zeros(len(raw), np.int8)
    for k, (a, b) in meta.periods.items():
        period[(frame >= a) & (frame <= b)] = k
    return {"frame": frame, "period": period, "home": out["home"], "away": out["away"], "ball": ball,
            "home_ids": ids["home"], "away_ids": ids["away"],
            "home_team_id": meta.home_team_id, "away_team_id": meta.away_team_id}


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

EVENT_COLUMNS = ["team", "type", "subtype", "period", "start", "end"]


def load_events_csv(path: Path) -> pd.DataFrame:
    """Games 1-2 event CSV -> standard table (team 0 home / 1 away, type, subtype, period, start, end)."""
    e = pd.read_csv(path)
    team = e["Team"].str.strip().str.lower().map({"home": HOME, "away": AWAY})
    assert team.notna().all(), "event without a team"
    out = pd.DataFrame({
        "team": team.astype(np.int8),
        "type": e["Type"].str.strip().str.upper(),
        "subtype": e["Subtype"].fillna("").str.strip().str.upper(),
        "period": e["Period"].astype(np.int8),
        "start": e["Start Frame"].astype(np.int64),
        "end": e["End Frame"].fillna(0).astype(np.int64),
    })
    return _finish_events(out)


def load_events_json(path: Path, home_team_id: str) -> pd.DataFrame:
    """Game 3 event JSON -> standard table (see load_events_csv)."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)["data"]
    rows = []
    for ev in data:
        sub = ev.get("subtypes")
        if isinstance(sub, dict):
            sub = [sub]
        team_id = (ev.get("team") or {}).get("id")
        assert team_id, f"event without a team: {ev.get('index', ev.get('type'))}"
        rows.append((HOME if team_id == home_team_id else AWAY,
                     ev["type"]["name"].strip().upper(),
                     "-".join(s["name"].strip().upper() for s in sub) if sub else "",
                     int(ev["period"]),
                     int(ev["start"]["frame"]),
                     int((ev.get("end") or {}).get("frame") or 0)))
    out = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    out["team"] = out["team"].astype(np.int8)
    out["period"] = out["period"].astype(np.int8)
    return _finish_events(out)


def _finish_events(ev: pd.DataFrame) -> pd.DataFrame:
    """Repair end frames (0 / before start -> start), stable-sort by (period, start), add flags."""
    ev = ev.copy()
    ev["end"] = np.maximum(ev["end"].to_numpy(), ev["start"].to_numpy())
    ev = ev.sort_values(["period", "start"], kind="stable").reset_index(drop=True)
    sub = ev["subtype"]
    typ = ev["type"]
    tokens = sub.str.split("-")

    def has(tok):            # subtype contains this exact "-"-separated token
        return tokens.map(lambda ts: tok in ts).astype(bool)

    foul_challenge = (typ == "CHALLENGE") & has("FAULT") & ~has("ADVANTAGE")
    ev["restart"] = (typ == "SET PIECE") | sub.str.startswith("GOAL KICK")
    ev["on_ball"] = (typ.isin(ON_BALL_TYPES) | ev["restart"]
                     | ((typ == "CHALLENGE") & has("WON") & ~foul_challenge))
    ev["ball_event"] = ~typ.isin(NON_BALL_TYPES) & ~((typ == "RECOVERY") & has("GOAL"))
    ev["end_half"] = (typ == "BALL LOST") & sub.str.contains("END HALF")
    foul = (typ == "FAULT RECEIVED") | foul_challenge
    shot_stop = (typ == "SHOT") & (has("OUT") | has("GOAL"))
    offside = typ.isin(["PASS", "BALL LOST"]) & sub.str.contains("OFFSIDE")
    hand = (typ == "BALL LOST") & has("HAND BALL")
    ev["out_of_play"] = (typ == "BALL OUT") | shot_stop | offside | hand
    referee = (typ == "BALL LOST") & has("REFEREE HIT")
    ev["stop"] = ev["out_of_play"] | foul | referee | ev["end_half"]
    return ev


# ---------------------------------------------------------------------------
# Frame labels from events
# ---------------------------------------------------------------------------

def _period_bounds(period: np.ndarray, k: int) -> tuple[int, int]:
    """Row range [a, b) of period k (periods are contiguous in all three games)."""
    rows = np.flatnonzero(period == k)
    assert len(rows) and rows[-1] - rows[0] + 1 == len(rows), f"period {k} is not contiguous"
    return int(rows[0]), int(rows[-1]) + 1


def _ffill_owner(n: int, pos: np.ndarray, team: np.ndarray) -> np.ndarray:
    """Owner per row of a period: team of the last on-ball event starting at or before the row.

    pos are row offsets (sorted, may repeat); on a repeated row the last event wins. -1 = none yet.
    """
    owner_at = np.full(n, -1, np.int8)
    if len(pos):
        last = len(pos) - 1 - np.unique(pos[::-1], return_index=True)[1]   # last event per row
        owner_at[pos[last]] = team[last]
    has = owner_at >= 0
    src = np.where(has, np.arange(n), -1)
    np.maximum.accumulate(src, out=src)
    return np.where(src >= 0, owner_at[np.maximum(src, 0)], -1).astype(np.int8)


def derive_labels(ev: pd.DataFrame, frame: np.ndarray, period: np.ndarray, fps: int = FPS,
                  restart_tol_s: float = 0.5, missing_restart_s: float = MISSING_RESTART_S,
                  silence_s: float = SILENCE_S) -> tuple[np.ndarray, dict]:
    """Per-frame labels (0 home, 1 away, 2 dead, 3 unknown) from a standard event table.

    Implements the rules of the module docstring. ``restart_tol_s``: events that start less than
    this before a restart are treated as part of it (tags a frame or two out of order around a
    restart) and do not move the start of the stoppage. Returns (label, diagnostics).
    """
    T = len(frame)
    frame0 = int(frame[0])
    assert np.all(np.diff(frame) == 1), "tracking frames must be contiguous"
    tol = int(round(restart_tol_s * fps))
    label = np.full(T, UNKNOWN, np.int8)
    diag = {"restarts": 0, "stoppages_from_stop_event": 0, "stoppages_from_last_touch": 0,
            "stoppages_extended_by_play_on": 0, "restarts_without_stoppage": 0,
            "untagged_restarts": 0, "end_half_tagged": 0, "silence_unknown_frames": 0,
            "tail_unknown_frames": 0, "events_outside_tracking": 0}

    for k in np.unique(period[period > 0]):
        a, b = _period_bounds(period, k)
        n = b - a
        e = ev[ev["period"] == k].reset_index(drop=True)
        start = e["start"].to_numpy() - frame0 - a           # row offsets inside the period
        end = e["end"].to_numpy() - frame0 - a
        diag["events_outside_tracking"] += int(np.sum((start < 0) | (end >= n)))
        start = np.clip(start, 0, n - 1)
        end = np.clip(end, 0, n - 1)
        team = e["team"].to_numpy()
        typ = e["type"].to_numpy()
        on_ball, ball_ev = e["on_ball"].to_numpy(), e["ball_event"].to_numpy()
        stop, restart, end_half = e["stop"].to_numpy(), e["restart"].to_numpy(), e["end_half"].to_numpy()

        # Rule 2: possession owner, carried forward from on-ball events.
        owner = _ffill_owner(n, start[on_ball], team[on_ball])
        dead = np.zeros(n, bool)
        unknown = owner < 0

        # Rule 3: the stoppage before each restart. An event's end frame is its last live frame
        # (Metrica's own tracking still shows the ball there and drops it on the next frame),
        # so a stoppage starts one row after the end of the event that anchors it.
        prev_r = -1                                            # index of the previous restart
        for i in np.flatnonzero(restart):
            diag["restarts"] += 1
            f_r = start[i]
            lo = prev_r if prev_r >= 0 else 0
            # candidate events: after the previous restart (inclusive), clearly before this one
            cand = np.arange(lo, i)
            cand = cand[ball_ev[cand] & (start[cand] < f_r - tol)]
            stops = cand[stop[cand]]
            if len(stops):
                anchor = stops[-1]
                s_r = end[anchor] + 1
                # play-on evidence after the stopping event (pseudo pick-ups, RECOVERY, excluded)
                later = cand[(cand > anchor) & (typ[cand] != "RECOVERY")]
                if len(later) and end[later].max() + 1 > s_r:
                    s_r = end[later].max() + 1
                    diag["stoppages_extended_by_play_on"] += 1
                diag["stoppages_from_stop_event"] += 1
            elif len(cand):
                touch = cand[typ[cand] != "RECOVERY"]
                s_r = (end[touch] if len(touch) else end[cand]).max() + 1
                diag["stoppages_from_last_touch"] += 1
            else:
                s_r = 0                                        # nothing before: period start
            if s_r < f_r:
                dead[s_r:f_r] = True
            else:
                diag["restarts_without_stoppage"] += 1
            prev_r = i

        # Rule 4: out-of-play event, then a long silence and no tagged restart.
        gap_min = int(round(missing_restart_s * fps))
        touch_idx = np.flatnonzero(ball_ev & (typ != "RECOVERY"))
        for i in np.flatnonzero(e["out_of_play"].to_numpy()):
            after = touch_idx[(touch_idx > i) & (start[touch_idx] > end[i])]
            if not len(after) or restart[after[0]]:
                continue                                       # nothing follows / rule 3 case
            j = after[0]
            if start[j] - end[i] >= gap_min:
                if not dead[end[i] + 1:start[j]].all():
                    diag["untagged_restarts"] += 1
                dead[end[i] + 1:start[j]] = True

        # Rule 5: end of the half (the end frame of the END HALF event is still live).
        eh = np.flatnonzero(end_half)
        if len(eh):
            dead[end[eh[0]] + 1:] = True
            diag["end_half_tagged"] += 1
        elif ball_ev.any():
            tail = end[ball_ev].max() + 1
            unknown[tail:] = True
            diag["tail_unknown_frames"] += int(n - tail)

        # Rule 6: long event-free stretches of live play are undeterminable.
        cover = np.zeros(n + 1, np.int32)                      # rows covered by a ball event
        np.add.at(cover, start[ball_ev], 1)
        np.add.at(cover, end[ball_ev] + 1, -1)
        touched = np.cumsum(cover[:n]) > 0
        sil = int(round(silence_s * fps))
        idx = np.arange(n)
        last = np.where(touched, idx, -10 * n)
        np.maximum.accumulate(last, out=last)
        nxt = np.where(touched, idx, 10 * n)
        nxt = np.minimum.accumulate(nxt[::-1])[::-1]
        quiet = (idx - last > sil) & (nxt - idx > sil) & ~dead
        diag["silence_unknown_frames"] += int((quiet & ~unknown).sum())
        unknown |= quiet

        lab = np.where(dead, DEAD, np.where(unknown, UNKNOWN, owner)).astype(np.int8)
        label[a:b] = lab
    return label, diag


# ---------------------------------------------------------------------------
# One game end to end
# ---------------------------------------------------------------------------

def load_game(raw_dir: Path, game: int) -> tuple[dict, pd.DataFrame]:
    """Raw tracking dict (see load_tracking_*) and standard event table of one sample game."""
    if game in (1, 2):
        tr = load_tracking_csv(raw_path(raw_dir, game, "home"), raw_path(raw_dir, game, "away"))
        ev = load_events_csv(raw_path(raw_dir, game, "events"))
    elif game == 3:
        tr = load_tracking_epts(raw_path(raw_dir, game, "metadata"), raw_path(raw_dir, game, "tracking"))
        ev = load_events_json(raw_path(raw_dir, game, "events"), tr["home_team_id"])
    else:
        raise ValueError(f"unknown Metrica sample game {game}")
    return tr, ev


def convert_game(raw_dir: Path, game: int) -> tuple[dict[str, np.ndarray], pd.DataFrame, dict]:
    """Canonical arrays, standard event table and diagnostics of one sample game."""
    tr, ev = load_game(raw_dir, game)
    label, diag = derive_labels(ev, tr["frame"], tr["period"])
    arrays = {"home": tr["home"], "away": tr["away"], "ball": tr["ball"],
              "period": tr["period"], "label": label, "pitch": np.array(PITCH, np.float32)}
    diag.update(frame0=int(tr["frame"][0]), n_events=int(len(ev)),
                home_ids=list(tr["home_ids"]), away_ids=list(tr["away_ids"]))
    return arrays, ev, diag


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _run_lengths(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(values, lengths) of the maximal constant runs of x."""
    if len(x) == 0:
        return x[:0], np.zeros(0, np.int64)
    bounds = np.concatenate(([0], np.flatnonzero(np.diff(x) != 0) + 1, [len(x)]))
    return x[bounds[:-1]], np.diff(bounds)


def spell_stats(label: np.ndarray, period: np.ndarray, fps: int = FPS) -> dict:
    """Possession spells and stoppages, run lengths of the labels (H / A spells, D stoppages).

    A spell is a maximal run of label 0 or of label 1 inside one period (any other label ends it);
    live minutes count frames with label 0/1. Dead segments are maximal runs of label 2.
    """
    spells, deads = [], []
    for k in np.unique(period[period > 0]):
        vals, lens = _run_lengths(label[period == k])
        spells.append(lens[vals < 2])
        deads.append(lens[vals == DEAD])
    sp = np.concatenate(spells) / fps
    dd = np.concatenate(deads) / fps
    live_min = float(np.sum((label < 2) & (period > 0))) / fps / 60
    return {
        "live_minutes": live_min,
        "n_spells": int(len(sp)),
        "spells_per_live_min": len(sp) / live_min if live_min else None,
        "spell_median_s": float(np.median(sp)) if len(sp) else None,
        "spell_mean_s": float(sp.mean()) if len(sp) else None,
        "spells_under_1s": int(np.sum(sp < 1.0)),
        "n_dead_segments": int(len(dd)),
        "dead_median_s": float(np.median(dd)) if len(dd) else None,
        "dead_mean_s": float(dd.mean()) if len(dd) else None,
    }


def game_summary(m: dict, fps: int = FPS, min_spell_s=(1.0, 2.0, 3.0)) -> dict:
    """Label, ball and player statistics of one canonical game (graded frames, period > 0)."""
    from .degrade import relabel_min_spell   # local import: degrade pulls in features

    label, period = m["label"], m["period"]
    graded = period > 0
    lab = label[graded]
    miss = np.isnan(m["ball"][:, 0])
    live, dead = graded & (label < 2), graded & (label == DEAD)

    out = {"T": int(len(label)), "minutes": float(len(label)) / fps / 60,
           "share": {name: float(np.mean(lab == code)) for name, code in
                     (("home", HOME), ("away", AWAY), ("dead", DEAD), ("unknown", UNKNOWN))}}
    out["dead_share"] = out["share"]["dead"]
    out.update(spell_stats(label, period, fps))
    out["spells_per_live_min_after_min_spell"] = {
        f"{k:g}s": spell_stats(relabel_min_spell(label, period, int(k * fps)), period, fps)["spells_per_live_min"]
        for k in min_spell_s}

    gaps = _run_lengths(miss[graded])
    gap_len = gaps[1][gaps[0]] / fps
    out["ball"] = {
        "missing_share": float(miss[graded].mean()),
        "visible_live": float(1 - miss[live].mean()) if live.any() else None,
        "visible_dead": float(1 - miss[dead].mean()) if dead.any() else None,
        "n_gaps": int(len(gap_len)),
        "gap_median_s": float(np.median(gap_len)) if len(gap_len) else None,
        "gap_p90_s": float(np.percentile(gap_len, 90)) if len(gap_len) else None,
        "gap_max_s": float(gap_len.max()) if len(gap_len) else None,
    }

    n_on = (~np.isnan(m["home"][..., 0])).sum(1) + (~np.isnan(m["away"][..., 0])).sum(1)
    out["players_per_frame"] = {"p5": float(np.percentile(n_on[graded], 5)),
                                "median": float(np.median(n_on[graded]))}

    # Sanity check of the event-derived owner: on live frames with a tracked ball, how often is
    # the player nearest to the ball from the team the label says has the ball?
    ok = live & ~miss
    d = [np.sqrt(np.sum((m[t][ok] - m["ball"][ok][:, None, :]) ** 2, axis=2)) for t in ("home", "away")]
    near = [np.min(np.where(np.isnan(x), np.inf, x), axis=1) for x in d]
    nearest_team = np.where(near[1] < near[0], AWAY, HOME)
    out["nearest_player_agrees"] = float(np.mean(nearest_team == label[ok])) if ok.any() else None
    return out
