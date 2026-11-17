"""Loader for the DFL/Sportec open tracking dataset.

Source: Bassek, M., Rein, R., Weber, H., Memmert, D. (2025). "An integrated
dataset of spatiotemporal and event data in elite soccer". Scientific Data.
Files on figshare, article 28196177, licensed CC BY 4.0. Seven matches
(2 Bundesliga, 5 2. Bundesliga), 25 Hz optical tracking.

This module downloads the raw XML files and converts each match into the
canonical possession-segmentation format used throughout this project:

    home   float32 [T, Nh, 2]  player x, y in metres, pitch-centred, NaN = not on pitch
    away   float32 [T, Na, 2]  same for the away team
    ball   float32 [T, 2]      ball x, y in metres, pitch-centred
    period int8    [T]         1 = first half, 2 = second half
    label  int8    [T]         0 = home in possession, 1 = away in possession,
                               2 = dead ball, 3 = unknown
    pitch  float32 [2]         pitch length, width in metres

Raw DFL positions XML layout (one FrameSet per object and half)::

    <Positions>
      <MetaData ...><PitchSize X="105.00" Y="68.00"/></MetaData>
      <FrameSet GameSection="firstHalf" TeamId="DFL-CLU-..." PersonId="DFL-OBJ-...">
        <Frame N="10000" T="..." X="19.95" Y="-34.45" S="..." .../>
        ...
      </FrameSet>
      <FrameSet GameSection="firstHalf" TeamId="BALL" PersonId="...">
        <Frame N="10000" X=".." Y=".." Z=".." BallPossession="2" BallStatus="1" .../>
      ...

X/Y are already metres with the origin at the centre spot (X along the long
axis). Frame numbers N are a running 25 Hz counter. Coordinates are in a fixed
stadium frame, i.e. teams swap playing direction at half time; we keep that
as is (no direction normalisation).

The labels are read directly from the raw ball FrameSet (BallStatus,
BallPossession) rather than through kloppy, so that missing values stay
visible as label 3 instead of being replaced by defaults.
"""

from __future__ import annotations

import json
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from lxml import etree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIGSHARE_ARTICLE = 28196177
FIGSHARE_API = f"https://api.figshare.com/v2/articles/{FIGSHARE_ARTICLE}"

MATCH_IDS = ["J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9"]

FPS = 25
SECTIONS = {"firstHalf": 1, "secondHalf": 2}

# Label codes of the canonical format.
HOME, AWAY, DEAD, UNKNOWN = 0, 1, 2, 3


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def list_figshare_files() -> list[dict]:
    """Return the figshare file listing (name, size, download_url, ...)."""
    with urllib.request.urlopen(FIGSHARE_API, timeout=60) as resp:
        return json.load(resp)["files"]


def download_raw(raw_dir: Path, match_ids: list[str] | None = None) -> None:
    """Download match information and raw positions XML for the given matches.

    Files that already exist with the expected size are skipped. Event files
    are not needed for the possession labels and are not downloaded.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    match_ids = match_ids or MATCH_IDS
    for f in list_figshare_files():
        name = f["name"]
        if not any(f"DFL-MAT-{m}" in name for m in match_ids):
            continue
        if "matchinformation" not in name and "positions_raw" not in name:
            continue
        target = raw_dir / name
        if target.exists() and target.stat().st_size == f["size"]:
            continue
        print(f"downloading {name} ({f['size'] / 1e6:.0f} MB)")
        tmp = target.with_suffix(".part")
        with urllib.request.urlopen(f["download_url"], timeout=120) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out, length=1 << 20)
        tmp.replace(target)


def find_raw(raw_dir: Path, match_id: str, kind: str) -> Path:
    """Locate the raw file of `kind` ('matchinformation' or 'positions_raw')."""
    hits = sorted(raw_dir.glob(f"DFL_*{kind}*DFL-MAT-{match_id}.xml"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one {kind} file for {match_id} in {raw_dir}, found {hits}")
    return hits[0]


# ---------------------------------------------------------------------------
# Match information
# ---------------------------------------------------------------------------

@dataclass
class MatchInfo:
    match_id: str
    home_team_id: str
    away_team_id: str
    home_name: str
    away_name: str
    pitch_length: float
    pitch_width: float
    # PersonId -> "home" / "away" for every rostered player (incl. unused subs)
    roster: dict[str, str] = field(default_factory=dict)


def parse_match_information(path: Path) -> MatchInfo:
    """Read team roles, rosters and pitch size from the match information XML."""
    root = etree.parse(str(path)).getroot()
    general = root.find(".//General")
    env = root.find(".//Environment")
    roster: dict[str, str] = {}
    team_ids = {}
    for team in root.iter("Team"):
        role = {"home": "home", "guest": "away"}[team.get("Role")]
        team_ids[role] = team.get("TeamId")
        for player in team.iter("Player"):
            roster[player.get("PersonId")] = role
    assert team_ids["home"] == general.get("HomeTeamId"), "home team mismatch"
    assert team_ids["away"] == general.get("GuestTeamId"), "away team mismatch"
    return MatchInfo(
        match_id=general.get("MatchId").replace("DFL-MAT-", ""),
        home_team_id=team_ids["home"],
        away_team_id=team_ids["away"],
        home_name=general.get("HomeTeamName"),
        away_name=general.get("GuestTeamName"),
        pitch_length=float(env.get("PitchX")),
        pitch_width=float(env.get("PitchY")),
        roster=roster,
    )


# ---------------------------------------------------------------------------
# Positions (streaming parse)
# ---------------------------------------------------------------------------

@dataclass
class FrameSetData:
    """All frames of one object (player or ball) in one half."""
    section: int          # 1 or 2
    team_id: str          # club id, "BALL" or "referee"
    person_id: str
    n: np.ndarray         # int64 frame numbers
    xy: np.ndarray        # float32 [len, 2]
    status: np.ndarray | None = None      # ball only: int8, -1 = missing
    possession: np.ndarray | None = None  # ball only: int8, -1 = missing


def _attr_float(v: str | None) -> float:
    return float(v) if v is not None else np.nan


def _attr_int(v: str | None) -> int:
    return int(v) if v is not None else -1


def parse_positions(path: Path, skip_referees: bool = True) -> tuple[list[FrameSetData], tuple[float, float] | None]:
    """Stream-parse a DFL raw positions XML file.

    Returns the list of FrameSets and the pitch size from the file's MetaData
    (or None if absent). Memory stays low because every element is cleared as
    soon as it has been consumed.
    """
    framesets: list[FrameSetData] = []
    pitch = None
    cur = None          # attributes of the FrameSet being read
    ns, xs, ys, st, po = [], [], [], [], []

    for event, elem in etree.iterparse(str(path), events=("start", "end"), huge_tree=True):
        tag = elem.tag
        if event == "start":
            if tag == "FrameSet":
                cur = dict(elem.attrib)
                ns, xs, ys, st, po = [], [], [], [], []
            continue

        # --- end events ---
        if tag == "Frame":
            if cur is not None and not (skip_referees and cur["TeamId"] == "referee"):
                a = elem.attrib
                ns.append(int(a["N"]))
                xs.append(_attr_float(a.get("X")))
                ys.append(_attr_float(a.get("Y")))
                if cur["TeamId"] == "BALL":
                    st.append(_attr_int(a.get("BallStatus")))
                    po.append(_attr_int(a.get("BallPossession")))
            elem.clear()
        elif tag == "FrameSet":
            team = cur["TeamId"]
            if not (skip_referees and team == "referee"):
                is_ball = team == "BALL"
                framesets.append(FrameSetData(
                    section=SECTIONS[cur["GameSection"]],
                    team_id=team,
                    person_id=cur["PersonId"],
                    n=np.asarray(ns, dtype=np.int64),
                    xy=np.stack([np.asarray(xs, np.float32), np.asarray(ys, np.float32)], axis=1),
                    status=np.asarray(st, np.int8) if is_ball else None,
                    possession=np.asarray(po, np.int8) if is_ball else None,
                ))
            cur = None
            elem.clear()
            # drop already processed siblings so the tree does not grow
            while elem.getprevious() is not None:
                del elem.getparent()[0]
        elif tag == "PitchSize":
            pitch = (float(elem.get("X")), float(elem.get("Y")))
    return framesets, pitch


# ---------------------------------------------------------------------------
# Conversion to the canonical format
# ---------------------------------------------------------------------------

def labels_from_ball(status: np.ndarray, possession: np.ndarray) -> np.ndarray:
    """Map raw BallStatus / BallPossession to label codes.

    BallStatus 0 -> dead (2) regardless of possession. BallStatus 1 with
    BallPossession 1/2 -> home (0) / away (1). Anything missing or outside
    those values -> unknown (3).
    """
    lab = np.full(status.shape, UNKNOWN, np.int8)
    lab[status == 0] = DEAD
    alive = status == 1
    lab[alive & (possession == 1)] = HOME
    lab[alive & (possession == 2)] = AWAY
    return lab


def build_match(info: MatchInfo, framesets: list[FrameSetData]) -> tuple[dict[str, np.ndarray], dict]:
    """Assemble canonical arrays for one match.

    Each half spans the frame-number range covered by its framesets
    (min..max N); frames missing from the source become NaN rows with
    label 3. Halves are concatenated without gap frames.

    Returns (arrays, diagnostics).
    """
    # Players that actually appear in the tracking data, per team, ordered by
    # first appearance (starters first, then substitutes) and PersonId.
    first_seen: dict[str, tuple[int, int]] = {}
    for fs in framesets:
        if fs.team_id == "BALL" or len(fs.n) == 0:
            continue
        key = (fs.section, int(fs.n.min()))
        first_seen[fs.person_id] = min(first_seen.get(fs.person_id, key), key)
    team_of = {fs.person_id: fs.team_id for fs in framesets if fs.team_id != "BALL"}
    for pid, tid in team_of.items():
        expected = {"home": info.home_team_id, "away": info.away_team_id}[info.roster[pid]]
        assert tid == expected, f"{pid}: team id {tid} disagrees with match information"
    order = sorted(first_seen, key=lambda p: (first_seen[p], p))
    home_ids = [p for p in order if team_of[p] == info.home_team_id]
    away_ids = [p for p in order if team_of[p] == info.away_team_id]
    col = {p: i for i, p in enumerate(home_ids)} | {p: i for i, p in enumerate(away_ids)}

    parts = {k: [] for k in ("home", "away", "ball", "period", "label")}
    diag: dict = {"halves": {}}
    for sec in (1, 2):
        sets = [fs for fs in framesets if fs.section == sec]
        balls = [fs for fs in sets if fs.team_id == "BALL"]
        assert len(balls) == 1, f"expected one ball frameset in half {sec}, got {len(balls)}"
        ball_fs = balls[0]
        n0 = min(int(fs.n.min()) for fs in sets if len(fs.n))
        n1 = max(int(fs.n.max()) for fs in sets if len(fs.n))
        T = n1 - n0 + 1

        home = np.full((T, len(home_ids), 2), np.nan, np.float32)
        away = np.full((T, len(away_ids), 2), np.nan, np.float32)
        ball = np.full((T, 2), np.nan, np.float32)
        label = np.full(T, UNKNOWN, np.int8)

        bi = ball_fs.n - n0
        ball[bi] = ball_fs.xy
        label[bi] = labels_from_ball(ball_fs.status, ball_fs.possession)
        for fs in sets:
            if fs.team_id == "BALL":
                continue
            target = home if fs.team_id == info.home_team_id else away
            target[fs.n - n0, col[fs.person_id]] = fs.xy

        gaps = np.diff(ball_fs.n)
        diag["halves"][sec] = {
            "first_N": n0,
            "last_N": n1,
            "T": T,
            "ball_frames": int(len(ball_fs.n)),
            "ball_frame_gaps": int((gaps != 1).sum()),
            "ball_missing_frames": int(T - len(np.unique(ball_fs.n))),
            "missing_status": int((ball_fs.status < 0).sum()),
            "missing_possession_while_alive": int(((ball_fs.status == 1) & ~np.isin(ball_fs.possession, (1, 2))).sum()),
            "possession_values": sorted(int(v) for v in np.unique(ball_fs.possession)),
            "status_values": sorted(int(v) for v in np.unique(ball_fs.status)),
            "player_xy_range": [float(np.nanmin(np.concatenate([home[..., 0].ravel(), away[..., 0].ravel()]))),
                                float(np.nanmax(np.concatenate([home[..., 0].ravel(), away[..., 0].ravel()]))),
                                float(np.nanmin(np.concatenate([home[..., 1].ravel(), away[..., 1].ravel()]))),
                                float(np.nanmax(np.concatenate([home[..., 1].ravel(), away[..., 1].ravel()])))],
            "ball_xy_range": [float(np.nanmin(ball[:, 0])), float(np.nanmax(ball[:, 0])),
                              float(np.nanmin(ball[:, 1])), float(np.nanmax(ball[:, 1]))],
            "players_on_pitch_median": [int(np.median((~np.isnan(home[..., 0])).sum(1))),
                                        int(np.median((~np.isnan(away[..., 0])).sum(1)))],
        }
        parts["home"].append(home)
        parts["away"].append(away)
        parts["ball"].append(ball)
        parts["period"].append(np.full(T, sec, np.int8))
        parts["label"].append(label)

    arrays = {k: np.concatenate(v, axis=0) for k, v in parts.items()}
    arrays["pitch"] = np.array([info.pitch_length, info.pitch_width], np.float32)
    diag.update(home=info.home_name, away=info.away_name, home_ids=home_ids, away_ids=away_ids,
                n_rostered_home=sum(r == "home" for r in info.roster.values()),
                n_rostered_away=sum(r == "away" for r in info.roster.values()))
    return arrays, diag


def convert_match(raw_dir: Path, match_id: str) -> tuple[dict[str, np.ndarray], dict]:
    """Parse the raw files of one match and return canonical arrays + diagnostics."""
    info = parse_match_information(find_raw(raw_dir, match_id, "matchinformation"))
    framesets, file_pitch = parse_positions(find_raw(raw_dir, match_id, "positions_raw"))
    if file_pitch is not None:
        assert np.allclose(file_pitch, (info.pitch_length, info.pitch_width)), "pitch size mismatch"
    return build_match(info, framesets)


# ---------------------------------------------------------------------------
# Possession statistics
# ---------------------------------------------------------------------------

def _runs(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run-length encode x -> (values, starts, lengths)."""
    if len(x) == 0:
        return x[:0], np.zeros(0, int), np.zeros(0, int)
    change = np.flatnonzero(np.diff(x)) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [len(x)]]))
    return x[starts], starts, lengths


def forward_fill_unknown(label: np.ndarray) -> np.ndarray:
    """Replace label 3 by the last known label; leading unknowns stay 3."""
    lab = label.copy()
    known = lab != UNKNOWN
    idx = np.where(known, np.arange(len(lab)), 0)
    np.maximum.accumulate(idx, out=idx)
    filled = lab[idx]
    filled[: np.argmax(known) if known.any() else len(lab)] = UNKNOWN
    return filled


def segment_stats(label: np.ndarray, period: np.ndarray, fps: int = FPS) -> dict:
    """Possession spell / dead-ball statistics, computed per half and pooled.

    A possession spell is a maximal run of label 0 or of label 1 within one
    half after forward-filling unknown frames; dead-ball frames (2) split
    spells. A direct switch is a home spell immediately followed by an away
    spell (or vice versa) with no dead-ball frame between them.
    """
    spells, deads = [], []
    switches = 0
    per_half_T = {}
    for sec in np.unique(period):
        lab = forward_fill_unknown(label[period == sec])
        per_half_T[int(sec)] = int(len(lab))
        vals, _, lens = _runs(lab)
        keep = vals != UNKNOWN  # leading unknown run, if any
        vals, lens = vals[keep], lens[keep]
        spells.extend(lens[vals != DEAD].tolist())
        deads.extend(lens[vals == DEAD].tolist())
        switches += int(np.sum((vals[:-1] != DEAD) & (vals[1:] != DEAD)))

    spells_s = np.asarray(spells, float) / fps
    deads_s = np.asarray(deads, float) / fps
    T = len(label)
    share = {name: float(np.mean(label == code)) for name, code in
             (("home", HOME), ("away", AWAY), ("dead", DEAD), ("unknown", UNKNOWN))}
    q = [10, 25, 50, 75, 90]
    return {
        "T": T,
        "T_per_half": per_half_T,
        "minutes": T / fps / 60,
        "share": share,
        "n_spells": int(len(spells_s)),
        "spell_mean_s": float(spells_s.mean()) if len(spells_s) else None,
        "spell_median_s": float(np.median(spells_s)) if len(spells_s) else None,
        "spell_quantiles_s": {f"q{p}": float(v) for p, v in zip(q, np.percentile(spells_s, q))} if len(spells_s) else None,
        "n_dead_segments": int(len(deads_s)),
        "dead_median_s": float(np.median(deads_s)) if len(deads_s) else None,
        "n_direct_switches": switches,
    }
