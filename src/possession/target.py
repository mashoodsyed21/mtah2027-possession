"""Load the competition match (BAR-ATH, tactical camera) into the canonical format.

The canonical format is shared with the DFL training data (see dfl.py)::

    home   float32 [T, Nh, 2]  player x, y in metres, pitch-centred, NaN = not tracked
    away   float32 [T, Na, 2]
    ball   float32 [T, 2]      NaN where the tracking has no ball
    period int8    [T]         1, 2 inside the graded halves, 0 elsewhere
    label  int8    [T]         public ground truth codes (0 H, 1 A, 2 D, 3 X)
    pitch  float32 [2]

Row i is video frame i, i.e. submission row i. The ATD counts frames from 1, so
video_frame = frame_id - 1 (see the official starter/load_atd.py).

The match data is licensed and gated: this module only reads it from the local
data folder (MTAH_DATA_DIR, default ../more-than-a-hack-2027/data) and caches the
parsed arrays next to it, never inside this repository.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OFFICIAL = os.environ.get("MTAH_REPO") or os.path.join(
    os.path.dirname(REPO), "more-than-a-hack-2027")
DATA_DIR = os.environ.get("MTAH_DATA_DIR") or os.path.join(OFFICIAL, "data")

N_FRAMES = 199319
PITCH = (105.3, 68.0)              # games.py, measured from the corrected tracking
HALVES = ((1, 11185, 90651), (2, 110808, 190197))   # DATA.md, inclusive video frames
GT_PUBLIC = os.path.join(OFFICIAL, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def _paths():
    folder = os.path.join(DATA_DIR, "tactical")
    return (os.path.join(folder, "BAR-ATH_tactical_atd.xml"),
            os.path.join(folder, "BAR-ATH_tactical_atd.txt"),
            os.path.join(DATA_DIR, "derived", "BAR-ATH_canonical.npz"))


def _slot_teams(meta_path):
    """0 home / 1 away per track slot. First team in the EPTS metadata is home."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]
    return np.array([0 if p.get("teamId") == team_ids[0] else 1 for p in root.iter("Player")],
                    dtype=np.int8)


def _num(text):
    return np.nan if text.startswith("N") else float(text)


def read_gt(path=GT_PUBLIC):
    """Public ground truth as codes 0 H, 1 A, 2 D, 3 X."""
    codes = {"H": 0, "A": 1, "D": 2, "X": 3}
    with open(path) as fh:
        next(fh)
        return np.array([codes[line.rsplit(",", 1)[1].strip().upper()] for line in fh],
                        dtype=np.int8)


def load(cache=True):
    meta_path, raw_path, cache_path = _paths()
    if cache and os.path.isfile(cache_path):
        z = np.load(cache_path)
        return {k: z[k] for k in z.files}

    team = _slot_teams(meta_path)
    length, width = PITCH
    players = np.full((N_FRAMES, len(team), 2), np.nan, np.float32)
    ball = np.full((N_FRAMES, 2), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            frame_id, groups, ball_group = line.rstrip("\n").split(":")
            i = int(frame_id) - 1
            for k, g in enumerate(groups.split(";")):
                v = g.split(",")
                if not v[0].startswith("N"):
                    players[i, k] = (float(v[0]) * length - length / 2,
                                     float(v[1]) * width - width / 2)
            v = ball_group.split(",")
            if not v[0].startswith("N"):
                ball[i] = (float(v[0]) * length - length / 2, float(v[1]) * width - width / 2)

    period = np.zeros(N_FRAMES, np.int8)
    for half, a, b in HALVES:
        period[a:b + 1] = half
    out = {
        "home": players[:, team == 0], "away": players[:, team == 1], "ball": ball,
        "period": period, "label": read_gt(), "pitch": np.array(PITCH, np.float32),
    }
    if cache:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, **out)
    return out
