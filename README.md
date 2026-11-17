# Frame-level ball possession — More than a Hack 2027, Phase 1

For every frame of FC Barcelona – Athletic Club (tactical camera, 25 fps, 199,319 frames)
this repository predicts `H` (home in possession), `A` (away in possession) or `D` (dead ball).

Nothing in this repository contains match data. The competition bundle is read from the
official challenge repository's `data/` folder (or `MTAH_DATA_DIR`), and every derived file is
written next to it or under `../extdata/`, both outside version control. Per-frame outputs
(`results/*.csv`, including the submission) are git-ignored.

## Approach in one paragraph

Frame models are trained on the seven DFL/Sportec open matches (Bassek et al. 2025, CC BY 4.0),
which carry per-frame `BallStatus` and `BallPossession`. Their clean tracking is first degraded
so it looks like the competition's automatic tracking (ball missing in long blocks, frozen
stoppage ball rebuilt as a realistic roll-and-carry path, smoothed ball, jitter, player
dropouts), with ball-gap statistics measured on the competition's automatic tracking (no labels, no video). Three LightGBM
models — dead ball, team in possession, team that restarts — read 220+ identity-free features
(team shape, movement, ball control history, touch kernels, set-piece templates, gap cues)
and are made home/away symmetric. A second LightGBM stage re-predicts the dead and team
probabilities from the stage-1 probabilities in temporal context. A minimum-duration Viterbi
decoder turns per-frame probabilities into possession segments; its parameters are chosen with
the official spellF1 scorer on leave-one-match-out DFL predictions and on the Metrica sample
games, both against their raw labels. Nothing is trained or tuned on the competition match, its
public ground truth or its video. Broadcast evidence and a referee-whistle detector were tried in
the lab scripts and are not used in the submission. See `TECHNICAL.md` (Section 9: rules).

## Reproduce

Needs the official challenge repository next to this one (or `MTAH_REPO` / `MTAH_DATA_DIR`) and
(for the lab whistle step only) `ffmpeg` on `PATH`. One command runs everything and prints the SHA-256 of the submission
(on Linux/macOS use `.venv/bin/` instead of `.venv/Scripts/`):

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
.venv/Scripts/python scripts/run_all.py --tag v5_k05 --skip audio   # -> results/final_v5_k05.csv + SHA-256
```

**Submitted file:** `1337.csv` = `results/final_v5_k05.csv`, SHA-256
`0325d257ec549b97a9132a3087b8317a2c79bf0c0797562b6e349cc724f90dbb`. A fresh run on 17 Nov 2026
reproduced it byte for byte. The submitted code state is tagged `submission-1337`.

`run_all.py` runs exactly these steps (`--from STEP` resumes, `--skip audio` leaves out the
whistle step the submission does not read):

```bash
# 1. external training data (downloads ~2.6 GB from figshare)
.venv/Scripts/python scripts/build_dfl.py
.venv/Scripts/python scripts/build_metrica.py          # second external domain (decoder choice)
# 2. whistle cue (lab only), features for degraded DFL, Metrica and the competition match
.venv/Scripts/python scripts/build_audio.py
.venv/Scripts/python scripts/build_features.py --seeds 0 1 2 --feat-dir features_v4
# 3. train with leave-one-match-out validation
.venv/Scripts/python scripts/train_eval.py --tag v5_k05 --feat-dir features_v4 --relabel 0.5 --team-relabel 0 --drop ball_acc
# 4. decoder chosen on raw DFL + Metrica labels
.venv/Scripts/python scripts/lab/joint_tune.py --tag v5_k05 --trials 150
# 5. stacking (5 seeds), pre-kick-off frames 110808-113741 forced to D (FAQ), decoding,
#    DFL-estimated edge shift -> results/final_v5_k05.csv
.venv/Scripts/python scripts/final_predict.py --tag v5_k05 --joint
# tests
.venv/Scripts/python -m pytest -q
```

Determinism: every LightGBM model uses `deterministic=True`, fixed seeds and
`possession.LGB_THREADS` (8) threads; degradation draws from numpy Generators seeded per match
and seed; the numba kernels are compiled without `fastmath`. `run_all.py` also sets
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` (1) and `PYTHONHASHSEED` (0) unless
they are already set. Run the steps by hand with the same settings to get the same file.

The submission is `results/final_<tag>.csv`; check it with the official
`evaluation/validate_submission.py`.

## Layout

- `src/possession/` — library: data loaders (`dfl.py`, `metrica_sample.py`, `target.py`), tracking
  degradation, features, decoders, stacking, target-only evidence (`broadcast.py`, `audio.py`,
  `fusion.py`) and a thin wrapper around the official scorer (`metrics.py`).
- `scripts/` — the pipeline above.
- `scripts/lab/` — exploratory experiments behind the choices in `TECHNICAL.md` (decoder,
  refinement, stacking and fusion searches, label granularity, domain checks). Not needed to
  reproduce the submission.
- `tests/` — unit tests on synthetic matches.

## Acknowledgements and data

No data is redistributed here; each source is downloaded or read locally under its own terms.

- **DFL/Sportec open data** — training data, licensed CC BY 4.0. Bassek, M., Rein, R., Weber, H.,
  Memmert, D. (2025). An integrated dataset of spatiotemporal and event data in elite soccer.
  *Scientific Data*. Files: figshare article 28196177.
- **Metrica Sports sample data** — validation only,
  [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data).
- **kloppy** (PySport, BSD 3-Clause) — used to cross-check our Metrica tracking parser.
- The competition match data and the official scorer belong to the More than a Hack 2027
  organisers and are read from their repository, never copied here.

## License

Code: MIT, see `LICENSE`. The licence covers this repository's code only, not any of the data above.
