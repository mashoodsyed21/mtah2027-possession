"""Build canonical possession files from the Metrica Sports open sample games.

Usage:
    python scripts/build_metrica.py [--out DIR] [--games 1 2 3] [--no-download] [--verify N]

For every game this writes
    <out>/metrica_game<g>.npz         arrays home, away, ball, period, label, pitch
                                      (labels derived from events, see possession.metrica_sample)
    <out>/metrica_game<g>_events.csv  the standardised event table the labels come from
and finally <out>/summary.json with per-game and pooled statistics (dead share, spells per live
minute, spell / stoppage durations, ball visibility) plus derivation diagnostics. Raw files are
cached in <out>/raw.

--verify N cross-checks the first N tracking frames of every game against kloppy's own Metrica
loaders (player positions per team as sets, ball) before writing anything.

Data: Metrica Sports sample data, https://github.com/metrica-sports/sample-data
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import metrica_sample as ms  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "extdata" / "metrica"

# Reference numbers quoted in the report (measured elsewhere, aggregates only).
REFERENCE = {
    "competition_public_window": {"spells_per_live_min": 4.23, "spell_median_s": 7.6, "spell_mean_s": 14.2},
    "dfl_raw": {"spells_per_live_min": 8.4},
}


def verify_against_kloppy(raw_dir: Path, game: int, tr: dict, n_frames: int) -> dict:
    """Max abs difference (in normalised units) between our parse and kloppy's for n_frames.

    kloppy's CSV loader (games 1-2) returns y as 1 - y_raw while its EPTS loader (game 3) keeps
    y_raw; both raw formats share Metrica's top-left origin (substitutes enter at y_raw ~ 1 in all
    three games), so kloppy's CSV y is mapped back before comparing.
    """
    from kloppy import metrica

    if game in (1, 2):
        ds = metrica.load_tracking_csv(home_data=str(ms.raw_path(raw_dir, game, "home")),
                                       away_data=str(ms.raw_path(raw_dir, game, "away")),
                                       limit=n_frames, coordinates="metrica")
        flip = True
    else:
        ds = metrica.load_tracking_epts(meta_data=str(ms.raw_path(raw_dir, game, "metadata")),
                                        raw_data=str(ms.raw_path(raw_dir, game, "tracking")),
                                        limit=n_frames, coordinates="metrica")
        flip = False
    L, W = ms.PITCH

    def ours(xy):                      # metres -> raw normalised
        return np.stack([(xy[..., 0] + L / 2) / L, (xy[..., 1] + W / 2) / W], axis=-1)

    def theirs(p):
        return (p.x, 1 - p.y) if flip else (p.x, p.y)

    grounds = {t.ground.name.lower(): t for t in ds.metadata.teams}
    worst, frame_mismatch = 0.0, 0
    for fr in ds.frames:
        i = int(fr.frame_id) - int(tr["frame"][0])
        if i < 0 or i >= len(tr["frame"]) or int(tr["frame"][i]) != int(fr.frame_id):
            frame_mismatch += 1
            continue
        for side in ("home", "away"):
            k = np.array([theirs(pdat.coordinates) for pl, pdat in fr.players_data.items()
                          if pl.team is grounds[side] and pdat.coordinates is not None
                          and not np.isnan(pdat.coordinates.x)], float).reshape(-1, 2)
            o = ours(tr[side][i])
            o = o[~np.isnan(o[:, 0])]
            if len(k) != len(o):
                worst = np.inf
                continue
            k, o = k[np.lexsort(k.T[::-1])], o[np.lexsort(o.T[::-1])]
            worst = max(worst, float(np.abs(k - o).max()) if len(k) else 0.0)
        b = fr.ball_coordinates
        kb = np.array(theirs(b), float) if b is not None and not np.isnan(b.x) else np.full(2, np.nan)
        ob = ours(tr["ball"][i])
        if np.isnan(kb[0]) != np.isnan(ob[0]):
            worst = np.inf
        elif not np.isnan(kb[0]):
            worst = max(worst, float(np.abs(kb - ob).max()))
    return {"frames": len(ds.frames), "frame_mismatch": frame_mismatch, "max_abs_diff": worst}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--games", nargs="*", type=int, default=list(ms.GAMES))
    ap.add_argument("--no-download", action="store_true", help="use cached raw files only")
    ap.add_argument("--verify", type=int, default=0, metavar="N",
                    help="compare the first N frames per game with kloppy's loaders")
    args = ap.parse_args()

    raw_dir = args.out / "raw"
    args.out.mkdir(parents=True, exist_ok=True)
    if not args.no_download:
        ms.download_raw(raw_dir, args.games)

    summary: dict = {
        "source": "Metrica Sports sample data, https://github.com/metrica-sports/sample-data",
        "fps": ms.FPS, "pitch": list(ms.PITCH),
        "labels": {"0": "home possession, ball alive", "1": "away possession, ball alive",
                   "2": "dead ball", "3": "unknown"},
        "label_rules": "derived from events, see possession.metrica_sample module docstring",
        "parameters": {"missing_restart_s": ms.MISSING_RESTART_S, "silence_s": ms.SILENCE_S},
        "reference": REFERENCE,
        "games": {},
    }
    pooled = {k: [] for k in ("ball", "period", "label")}
    for g in args.games:
        t0 = time.time()
        arrays, ev, diag = ms.convert_game(raw_dir, g)
        if args.verify:
            tr, _ = ms.load_game(raw_dir, g)
            diag["kloppy_check"] = verify_against_kloppy(raw_dir, g, tr, args.verify)
            print(f"game {g}: kloppy check {diag['kloppy_check']}", flush=True)
        np.savez_compressed(args.out / f"metrica_game{g}.npz", **arrays)
        ev.to_csv(args.out / f"metrica_game{g}_events.csv", index=False)

        stats = ms.game_summary(arrays)
        stats["diagnostics"] = diag
        summary["games"][f"game{g}"] = stats
        # offset periods so that halves of different games never merge when pooled
        for k in ("ball", "label"):
            pooled[k].append(arrays[k])
        pooled["period"].append(np.where(arrays["period"] > 0, arrays["period"] + 10 * g, 0).astype(np.int32))
        print(f"game {g}: T={stats['T']} dead={stats['dead_share']:.3f} unknown={stats['share']['unknown']:.4f} "
              f"spells/live-min={stats['spells_per_live_min']:.2f} median={stats['spell_median_s']:.1f}s "
              f"mean={stats['spell_mean_s']:.1f}s dead segs={stats['n_dead_segments']} "
              f"(median {stats['dead_median_s']:.1f}s) ball missing={stats['ball']['missing_share']:.3f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    if len(args.games) > 1:
        lab = np.concatenate(pooled["label"])
        per = np.concatenate(pooled["period"])
        overall = ms.spell_stats(lab, per)
        graded = per > 0
        overall["dead_share"] = float(np.mean(lab[graded] == ms.DEAD))
        overall["unknown_share"] = float(np.mean(lab[graded] == ms.UNKNOWN))
        overall["ball_missing_share"] = float(np.mean(np.isnan(np.concatenate(pooled["ball"])[graded, 0])))
        overall["n_games"] = len(args.games)
        summary["overall"] = overall
        print("pooled:", json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in overall.items()}))

    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
