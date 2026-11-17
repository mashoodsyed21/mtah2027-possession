"""Build canonical possession-segmentation files from the DFL/Sportec open data.

Usage:
    python scripts/build_dfl.py [--out DIR] [--matches J03WMX J03WN1 ...] [--no-download]

For every match this writes <out>/<MATCH_ID>.npz (arrays home, away, ball,
period, label, pitch; see possession.dfl) and finally <out>/summary.json with
per-match and pooled possession statistics plus parsing diagnostics.
Raw XML files are cached in <out>/raw.

Data: Bassek et al. (2025), Scientific Data, figshare article 28196177, CC BY 4.0.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from possession import dfl  # noqa: E402

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "extdata" / "dfl"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--matches", nargs="*", default=dfl.MATCH_IDS)
    ap.add_argument("--no-download", action="store_true", help="use cached raw files only")
    args = ap.parse_args()

    raw_dir = args.out / "raw"
    if not args.no_download:
        dfl.download_raw(raw_dir, args.matches)

    summary: dict = {"source": "DFL/Sportec open data, Bassek et al. 2025, figshare 28196177, CC BY 4.0",
                     "fps": dfl.FPS,
                     "labels": {"0": "home possession, ball alive", "1": "away possession, ball alive",
                                "2": "dead ball", "3": "unknown"},
                     "matches": {}}
    all_label, all_period = [], []
    for mid in args.matches:
        t0 = time.time()
        arrays, diag = dfl.convert_match(raw_dir, mid)
        np.savez_compressed(args.out / f"{mid}.npz", **arrays)

        stats = dfl.segment_stats(arrays["label"], arrays["period"])
        stats["diagnostics"] = diag
        summary["matches"][mid] = stats
        # offset the period so halves of different matches never merge when pooled
        all_label.append(arrays["label"])
        all_period.append(arrays["period"].astype(np.int32) + 10 * len(all_period))
        print(f"{mid}: T={stats['T']} spells={stats['n_spells']} "
              f"median={stats['spell_median_s']:.1f}s dead={stats['share']['dead']:.3f} "
              f"unknown={stats['share']['unknown']:.4f} home={arrays['home'].shape[1]} "
              f"away={arrays['away'].shape[1]} ({time.time() - t0:.0f}s)", flush=True)

    overall = dfl.segment_stats(np.concatenate(all_label), np.concatenate(all_period))
    overall.pop("T_per_half")
    overall["n_matches"] = len(args.matches)
    overall["spells_per_match_mean"] = overall["n_spells"] / len(args.matches)
    summary["overall"] = overall
    with open(args.out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in overall.items()}, indent=2))


if __name__ == "__main__":
    main()
