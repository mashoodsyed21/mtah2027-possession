"""Reproduce the submission end to end and print its SHA-256.

    python scripts/run_all.py [--tag final] [--from STEP] [--skip STEP ...] [--expect SHA256]

Runs every step as its own process with fixed flags (FEAT = possession.FEAT_DIR):

    dfl       build_dfl.py                         DFL/Sportec open data -> ../extdata/dfl/ (~2.6 GB download)
    metrica   build_metrica.py                     Metrica sample games, validation only -> ../extdata/metrica/
    audio     build_audio.py                       whistle cue (needs ffmpeg), lab only -> <MTAH data>/derived/
    features  build_features.py --seeds 0 1 2 --feat-dir FEAT
    train     train_eval.py --tag TAG --feat-dir FEAT --relabel 0.5 --team-relabel 0 --drop ball_acc
    tune      lab/joint_tune.py --tag TAG          decoder chosen on raw DFL + Metrica labels (external only)
    final     final_predict.py --tag TAG --joint   stage 2 (5 seeds), pre-kick-off frames 110808-113741
                                                   forced dead (FAQ), decoding, DFL-estimated edge shift

Every choice is made on external data (DFL, Metrica); nothing is tuned on the competition match,
its public window or its video. metrica is required (the decoder is tuned on it); audio feeds only
lab scripts, `--skip audio` leaves it out. The submission is results/final_<TAG>.csv. Thread counts are pinned before any
child starts (OpenMP / BLAS below, LightGBM via possession.LGB_THREADS), so a rerun gives the same
file. Cached intermediate files (downloads, canonical arrays, feature files) are reused when present.
"""
import os

# Before anything numeric is imported, here or in a child process (which inherits them).
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYTHONHASHSEED", "0")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
from possession import FEAT_DIR  # noqa: E402

STEPS = ("dfl", "metrica", "audio", "features", "train", "tune", "final")


def commands(tag):
    return {
        "dfl": ["build_dfl.py"],
        "metrica": ["build_metrica.py"],
        "audio": ["build_audio.py"],
        "features": ["build_features.py", "--seeds", "0", "1", "2", "--feat-dir", FEAT_DIR],
        "train": ["train_eval.py", "--tag", tag, "--feat-dir", FEAT_DIR, "--relabel", "0.5",
                  "--team-relabel", "0", "--drop", "ball_acc"],
        "tune": [os.path.join("lab", "joint_tune.py"), "--tag", tag, "--trials", "150"],
        "final": ["final_predict.py", "--tag", tag, "--joint"],
    }


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="final", help="run name; the submission is results/final_<tag>.csv")
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0], help="resume at this step")
    ap.add_argument("--skip", nargs="*", choices=STEPS, default=[], help="steps to leave out")
    ap.add_argument("--expect", default=None, help="expected SHA-256; exit 1 on mismatch")
    args = ap.parse_args()

    cmds = commands(args.tag)
    t0 = time.time()
    for step in STEPS[STEPS.index(args.start):]:
        if step in args.skip:
            continue
        cmd = [sys.executable, os.path.join("scripts", cmds[step][0])] + cmds[step][1:]
        print("== %s: %s" % (step, " ".join(cmd[1:])), flush=True)
        subprocess.run(cmd, cwd=REPO, check=True)
        print("== %s done (%.0f s)" % (step, time.time() - t0), flush=True)

    csv = os.path.join(REPO, "results", "final_%s.csv" % args.tag)
    digest = sha256(csv)
    print("SHA-256  %s  %s" % (digest, os.path.relpath(csv, REPO)))
    if args.expect and digest != args.expect.lower():
        print("MISMATCH: expected %s" % args.expect, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
