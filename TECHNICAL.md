# Frame-level ball possession: technical document

More than a Hack 2027, Phase 1. Team 1337, repository `https://github.com/mashoodsyed21/mtah2027-possession`,
submission `1337.csv`.

All numbers are
aggregates from `results/experiments.jsonl`, `results/decodes.jsonl` or the saved out-of-fold
(OOF) DFL predictions. No match data is reproduced here.

## 1. Approach and why

For each of the 199,319 frames of FC Barcelona – Athletic Club (tactical camera, 25 fps) we predict
`H`, `A` or `D`, and `X` outside the two graded halves. There are two stages:

1. **Frame models.** Three LightGBM classifiers read 243 identity-free features per frame and give
   P(ball dead), P(home | ball alive) and P(home restarts | ball dead). They are trained on the
   seven DFL/Sportec open matches, degraded to look like the competition's automatic tracking.
2. **Decoder.** A Viterbi decoder with minimum segment durations and transition costs turns the
   per-frame probabilities into segments. Its parameters are chosen by the official spellF1 on
   leave-one-match-out (LOMO) DFL predictions and on the Metrica sample games, both scored against
   their raw labels. Every trained or tuned quantity comes from external data (Section 9).

Why this design:

- **The metric counts possessions, not frames.** On DFL OOF predictions the per-frame argmax
  reaches macroF1 0.91 but spellF1 only 0.39: it reports 5,151 possessions where there are 1,964.
  Decoding gains more than any frame-model change we tried (table B).
- **Labels have to come from elsewhere.** The competition match has five public minutes of ground
  truth: enough for a sanity check, not for training. The DFL open data (Bassek et al. 2025,
  CC BY 4.0) has per-frame `BallStatus` and `BallPossession` for seven full matches (668 minutes,
  about 1.0 M frames). That is the target variable itself, but on operator-corrected tracking much
  cleaner than the target's, so we degrade it first (Section 3).
- **Gradient-boosted trees, not a sequence network.** Seven matches is little data for a deep
  temporal model. LightGBM handles missing values natively, which matters with the ball missing on
  41% of graded frames. A full LOMO run fits on a CPU, and each feature can be checked for domain
  shift separately. Temporal context comes from windowed features (0.2–30 s, past, future and
  centred, since decoding is offline) and from the decoder.
- **Three binary models instead of one three-class model.** Whether the ball is dead is a property
  of the whole pitch (everyone slows down, the ball leaves the pitch, set-piece shapes form). Who has
  the ball is a question about two teams and should be symmetric between them. Splitting the two
  lets the team model train on live frames only and be exactly symmetric. The frame probabilities
  are P(H) = (1 − p_dead)·p_home, P(A) = (1 − p_dead)(1 − p_home), P(D) = p_dead.

## 2. Assets: used, tried, dropped

| Asset | How it is used | In the final file |
|---|---|---|
| Tactical ATD | The only per-frame model input. Positions in metres, pitch-centred. Track slots are used only for team membership. Its ball-gap lengths feed the degradation (3.2). | yes |
| DFL/Sportec open data, 7 matches | Training set and LOMO validation, after degradation. | yes |
| Metrica Sports sample games 1–3 | Second external domain. Labels derived from events (rules in `metrica_sample.py`), degraded like DFL, never trained on; half of the decoder-selection objective (4.3). | decoder choice |
| Broadcast ATD + homography | `valid` flag (a pitch homography exists) as dead-ball evidence, aligned to the tactical timeline by ball-position matching. | no: its weight could only be estimated on the competition match itself |
| Tactical video, audio track | Referee-whistle detector, evidence that play has just stopped. | no (lab only) |
| Tactical video, images | Not used. No own detection or tracking. | no |
| Smart Tagging | Not used (below). | no |
| Public 5-minute ground truth | Checks format, frame alignment and gross failures. Never used to train or choose parameters (an earlier version chose its label granularity from it; corrected, Section 9). | check only |
| kloppy | Cross-checks our Metrica tracking parser (`build_metrica.py --verify`). | tooling |
| Pre-trained models, LLMs | None in the pipeline. The code was written with an AI coding assistant that saw code and aggregate scores only, never match data, video or per-frame predictions. | no |

**Broadcast.** The TV feed cuts to close-ups, replays and crowd shots mostly while the ball is
dead. On the competition match it shows the pitch on about 96–97% of live frames and about 24% of
dead frames. It runs on its own timeline. We find one constant offset per half by matching the two
recordings' ball positions in pitch coordinates (coarse search over ±80,000 frames, then ±15
frames). An offset scores by the share of co-visible frames where the two balls are within 3 m. The
1-s mean of "pitch not shown" enters as a naive-Bayes log-likelihood ratio on logit P(D). The
ratio is estimated on the competition match without labels, by EM (`final_predict.py`; the lab
script `scripts/lab/decode_target.py` can instead use the frames the frame model is confident
about). It is then multiplied by a shrink factor (0.5), because the signal and the frame model
are not independent (`src/possession/fusion.py`). **Not in the final file:** on simulated DFL
signals it did not help (table C), and its ratio and shrink could only be set on the competition
match, while the rules require tuning on external data.

**Whistle.** The audio is decoded locally with ffmpeg. Per frame we whiten the 1.8–5 kHz band
against a ±10 s running background and score how far the strongest bin stands out. Frames above the
match's 99th percentile, merged into events of at least 0.16 s, count as whistles. The events fall
within 0.3–4.4 s of all four published kick-off and end-of-half frames. Restarts usually have no
whistle, so this is evidence for the ball going dead only. It enters P(D) like the broadcast
signal.

**Smart Tagging: not used.** We considered it. Its intervals are phases of play, not possessions,
and DFL has no equivalent, so it could only enter as a hand-set rule. The only labelled check for
such a rule on this match is the five-minute public window, too short to set its weight.

**Dropped after measurement** (Section 6):

- Dead-ball model without ball features: worse by 0.06 spellF1 on DFL and 0.06–0.10 on Metrica
  once the v2 degradation and features were in place.
- Coarsened team-model training labels: v1c (raw team labels) beat v1b.
- `ball_acc`: removing it moves DFL spellF1 by −0.002 and Metrica by −0.008 to +0.004, i.e. no
  measurable effect. The final run drops it (`--drop ball_acc`).
- Restart-aware 4-state decoder: 0.7196 against 0.7198 for the 3-state decoder.
- Whistle fusion: not measurable on DFL, public spellF1 unchanged (table C).

## 3. Missing and noisy tracking

### 3.1 How the target differs from DFL

The competition tracking misses the ball on 41% of graded frames: long blocks during stoppages,
and crowded or fast play. It filters the ball path, positions jitter, players drop out for short
stretches, and the number of tracked players varies differently (5th percentile 14; 12 in
degraded DFL, after its dropouts and blackouts). DFL is operator-corrected: the ball and every player on the pitch are present on
every frame. During a
stoppage DFL freezes the ball where play stopped and jumps it to the restart spot on the restart
frame, which no real tracker does. A model trained on clean DFL would rely on signals that do not
exist at test time.

### 3.2 Degradation of DFL (`src/possession/degrade.py`)

Each DFL match is degraded three times (seeds 0–2). Each copy draws its own noise levels from broad
ranges, so the model does not fit one exact noise level. In order:

1. **Stoppage ball rebuilt.** The ball rolls on with its incoming velocity (deceleration 4 m/s²,
   0.3–2 s, at most 3 m past the lines), is carried at 1–2 m/s to the restart spot and waits there
   for at least 0.5 s. Stoppages of 1 s or less are interpolated linearly.
2. **Smoothing.** A NaN-aware Gaussian local-linear filter on the ball (σ 4–6 frames) and players
   (σ 0.5–1.5 frames), as the target filters its paths.
3. **Ball gaps that depend on context.** 60–90% of stoppages lose the ball from around the stop
   (0.5 s before to 2.5 s after it) until around the restart (1.5 s before to 0.5 s after it). In
   live play, gaps start where players crowd the ball (weight n_within_3m² + 0.2, doubled above
   12 m/s ball speed) until 8–24% of live frames are missing. A share of 30–70% of live gap lengths is drawn from the target match's own gap
   lengths (up to 15 s, measured at runtime from the automatic tracking only, no labels, no
   video; see Section 9); the rest are lognormal (median 5 s, 3–30 s). Gaps
   under 0.3 s are closed. False ball detections (0.5 per minute, 6–10 frames) appear next to a
   player inside gaps.
4. **Jitter.** Slowly drifting errors: players 0.1–0.3 m (correlation time 12 frames), ball
   0.15–0.4 m (4 frames).
5. **Player dropouts.** 0.2–0.8 per player and minute (median 3 s). In 30% of ball gaps, 20–60% of
   the players vanish as well, in live and dead gaps alike. Tying these blackouts to stoppages
   would create a label cue the target does not have.

All randomness comes from one seeded numpy Generator per copy, so the training data is regenerated
exactly from the provided files.

### 3.3 Features that tolerate a missing ball and anonymous slots

- **No identities.** A team is a set of positions per frame, and attacking direction is derived per
  half from where each team stands. Where player numbers matter we use shares, not counts, so low
  tracking coverage does not read as low tempo.
- **Ball features are NaN when the ball is missing,** and LightGBM routes NaN explicitly. Gap
  features describe the gap itself: time since and until the ball was seen, where it was last and
  next seen (visible runs under 12 frames do not count, which filters false detections), gap
  length, and implied ball speed across it.
- **Ball-free proxies:** whole-pitch movement (walking and fast-running shares, synchronised
  braking), team shape and its 2-s trend, an "anchor" where the closest home–away pairs are, as a
  stand-in for the ball, goal-side marking, and geometric set-piece templates (throw-in, corner,
  goal kick, kick-off).

### 3.4 Team symmetry

Every feature about one team has an `h_` and an `a_` version computed by the same code. The team
and restart models are trained on every frame twice, once as recorded and once with the teams
exchanged (features swapped, labels flipped). They predict p_home = ½·(m(x) + 1 − m(swap(x))), so
neither model can prefer a side. The dead model is not symmetrised; its target is team-free.

## 4. Possession rules implemented

### 4.1 Base labels and coarsening

From DFL's ball frames: `BallStatus` 0 is `D`; `BallStatus` 1 with `BallPossession` 1 or 2 is `H`
or `A`. Anything else is unknown, and is neither trained on nor scored. DFL's possession changes
on every controlled touch, which is finer than the competition's definition. We coarsen it: inside
each live stretch, a team spell shorter than *k* goes to the team that had the ball before it (to
the next team if it opens the stretch), shortest first. Dead frames never change.

| Labels | Spells per live minute | Median spell | Mean spell |
|---|---|---|---|
| DFL raw | 8.38 (6.51–10.97) | 3.8 s | 7.3 s |
| DFL, k = 1 s | 6.98 | 5.0 s | 8.8 s |
| DFL, k = 2 s (superseded) | 5.38 (4.55–6.60) | 7.1 s | 11.3 s |
| DFL, k = 3 s | 4.48 | 9.1 s | 13.6 s |
| Metrica event labels, raw / k = 2 s | 6.11 / 4.3–5.0 | 5.4 s / – | 9.8 s / – |
| Competition, public window | 4.23 | 7.6 s | 14.2 s |

**Final choice: raw labels (k = 0) for scoring and decoder selection.** CHALLENGE.md states that
the competition labels are derived mechanically from corrected tracking (dead-ball flag plus the
team the tracking names as owner) and that nothing is smoothed afterwards. DFL `BallStatus` /
`BallPossession` are of the same kind, so the raw DFL labels and the raw Metrica event labels are
the external stand-ins; no coarsening rule is applied to them. The team model trains on raw DFL
possession throughout. The dead-ball labels are not changed by coarsening, so the `--relabel`
value of a run only moves its own decoder search, which the final chain replaces (4.3).

*Superseded:* earlier versions coarsened the labels at k = 2–2.5 s, chosen because that matched the
public window's spell statistics in this table. That is a choice made on the ground truth and is
withdrawn (Section 9); the rows above stay for reference.

### 4.2 Where the line is

- **Loose balls and challenges.** The training labels are DFL's own, uncoarsened. In the output a
  possession lasts at least 1.5 s (`min_len_HA` = 38 frames, except the first and last segment of a
  half), as chosen on external data (4.3). A challenge won for a moment, a deflection or a 50/50 ball that comes back
  stays with the team that had the ball. In the Metrica labels possession changes only on an
  on-ball event of the other team (pass, carry, shot, recovery, set piece, challenge won). Lost or
  neutral challenges and fouls do not change it; a fouled team gains the ball with its free kick.
- **Dead ball.** DFL's `BallStatus`: from the ball leaving play, or play being stopped, to the
  restart. A stoppage splits possession, so the same team before and after counts twice, as in
  the official spell definition. Stoppages last at least 2 s in the output (`min_len_D` = 50).
- **Restarts.** The possession after a stoppage starts on the restart frame. The restart model
  predicts the team of the next live spell. The 4-state decoder uses it (dead-home-restarts may
  exit only to `H`, dead-away-restarts only to `A`); it scored the same as the 3-state decoder,
  which takes the team after a stoppage from the team model. Submitted: the 3-state decoder.
- **Second-half kick-off.** The second half is graded from frame 110,808 but kicks off at 113,742
  (FAQ). Those frames are forced to `D` before decoding (`final_predict.py`).
- **Abstention.** `X` only outside the graded halves; every graded frame gets `H`, `A` or `D`.

### 4.3 Decoder (`src/possession/decode.py`)

The 3-class probabilities are averaged over a centred window inside each half and rescaled by a
temperature: logp = log softmax(log p̄ / T). Viterbi then runs over an explicit-minimum-duration
HMM: class c is a chain of k_c sub-states, so every interior segment lasts at least k_c frames.
Leaving a class costs `cost_switch_team` (H↔A), `cost_to_dead` (H/A→D) or `cost_from_dead` (D→H/A),
in nats. Each half is decoded separately, and ties resolve deterministically. The parameters come
from 150 seeded random draws over window, temperature, the three costs and the two minimum
lengths, scored by mean official spellF1 over the seven DFL OOF matches.

Best for v2b: window 2 s, T = 0.5, switch 5, to-dead 50, from-dead 5, `min_len_HA` 75 frames,
`min_len_D` 1. Entering a stoppage needs strong, sustained dead evidence; leaving one is cheap.

**Final decoder** (`scripts/lab/joint_tune.py`, 150 seeded draws plus the run's own setting):
objective = mean of official spellF1 on the seven DFL OOF matches and on the Metrica games
(3 games × 3 degraded copies), both against raw labels. Result for v5: window 13 frames, T = 0.5,
switch 2, to-dead 25, from-dead 5, `min_len_HA` 38, `min_len_D` 50.

## 5. Validation protocol

1. **LOMO on 7 DFL matches.** Per held-out match, the three models train on the other six (3
   degraded copies each, every 4th graded frame) and predict its seed-0 copy. The decoder is tuned
   on all seven OOF sequences with the official scorer, imported unmodified from the challenge
   repository.
2. **Metrica sample games.** Models retrained on all DFL matches, decoder from step 1 unchanged,
   applied to the degraded copies of each game. Its tracking comes from the same provider family
   as the target. Since the final chain the decoder is chosen on DFL and Metrica together, so
   Metrica is no longer an untouched check; the nested DFL score of the post-model chain is.
3. **Public 5-minute window,** as a sanity check only. It holds about 11 predicted possessions, so
   one possession moves spellF1 by several points. The FAQ also shows the public optimum can lose on
   the full match. No parameter is chosen by it.
4. **Domain shift per feature.** `scripts/lab/domain_check.py` computes, per feature, the AUC of
   "target vs degraded DFL" and the difference in missing share: 1 of 223 features
   have |AUC − 0.5| > 0.15 (`ball_acc`, inflated by simulated ball jitter; dropped from the final
model). Speed and team-spread features differ modestly (AUC 0.41–0.59), consistent with a match
that has more dead time and a possession-heavy home side. `scripts/lab/noise_profile.py` compares tracking statistics of the target,
   clean DFL and degraded DFL. We also watch the predicted class shares on the full match
   (DFL dead share: 45%).

## 6. Results

**A0. Final chain (3 Nov 2026).** All on external data, against raw labels (k = 0).

| Frame model | Decoder | DFL LOMO | Metrica (3 games × 3 copies) | Mean |
|---|---|---|---|---|
| v3 features, decoder of the superseded k = 2.5 run | own | 0.530 | 0.649 | 0.589 |
| v3 features | joint (4.3) | 0.599 | 0.705 | 0.652 |
| **v5 = v3 + 21 gap-context and front-line features** | joint | **0.600** | **0.707** | **0.654** |
| average of v2, v3 and v5 probabilities | joint | 0.604 | 0.710 | 0.657 |

The three-model average is 0.003 better, inside the noise, and one member (v2) cannot be rebuilt
with today's code (it predates a half-time fix of the `vcx` features), so the final file uses v5
alone. Post-model chain on v5, nested LOMO on DFL raw labels (`final_predict.py --joint`):
stage 1 0.6000, + stacking (5 seeds) 0.6112, + H↔A edge shift of 1 frame 0.6118. Public window
(check only, not used for any choice): spellF1 0.667, macroF1 0.902, 13 of 17 possessions.
File `results/final_v5_k05.csv` (submitted as `1337.csv`), SHA-256
`0325d257ec549b97a9132a3087b8317a2c79bf0c0797562b6e349cc724f90dbb`.

**A. Frame-model runs**, each with its own tuned decoder (superseded scoring: each run against
labels coarsened at its own k).

| Run | Change | Feat. | DFL LOMO spellF1 | Metrica g1 / g2 / g3 | Public spellF1 / macroF1 | Full-match D share |
|---|---|---|---|---|---|---|
| v1_k2 | base features, ball-aware dead model, team labels at 2 s | 153 | 0.7139 | – | 0.586 / 0.698 | 0.347 |
| v1b_noballdead | dead model without ball features | 153 | 0.6364 | – | 0.652 / 0.906 | 0.506 |
| v1c_team0 | v1b + raw team labels | 153 | 0.6513 | – | 0.622 / 0.908 | 0.507 |
| v2a_noball | v1c + `features_extra` | 223 | 0.6572 | 0.698 / 0.693 / 0.704 | 0.634 / 0.882 | 0.520 |
| **v2b_ball** | v2a, ball features back in the dead model | 223 | **0.7198** | **0.777 / 0.788 / 0.768** | 0.657 / 0.909 | 0.478 |
| v2c_noacc | v2b without `ball_acc` | 222 | 0.7182 | 0.769 / 0.790 / 0.772 | 0.657 / 0.900 | 0.477 |
| v2d_k1 | v2c, labels coarsened at 1 s | 222 | 0.6676* | 0.741 / 0.759 / 0.717* | 0.606 / 0.896 | 0.475 |
| v2e_k3 | v2c, labels coarsened at 3 s | 222 | 0.7427* | 0.810 / 0.847 / 0.811* | 0.657 / 0.922 | 0.483 |
| v3_k25 (superseded) | features_v3 (half-time fix), labels at 2.5 s, dead model swap-averaged | 222 | 0.7412* | 0.789 / 0.821 / 0.801* | 0.643 / 0.927 | 0.480 |

\* scored against labels coarsened at the run's own k, so not comparable across k; see table A2.

**A2. Superseded: choosing a coarsened label granularity.** This range of k came from the public
window (4.2 possessions per live minute), which is not allowed as a tuning signal; kept for
reference only, see 4.1 and A0. Each run's out-of-fold DFL decode, scored against DFL labels coarsened at several k
(earlier lab script `cross_k.py`, removed; see Section 9):

| Run (trained at k) | k = 1.5 | k = 2.0 | k = 2.5 | k = 3.0 | k = 3.5 | mean |
|---|---|---|---|---|---|---|
| v2d (1 s) | 0.694 | 0.694 | 0.684 | 0.665 | 0.646 | 0.677 |
| v2c (2 s) | 0.687 | 0.718 | 0.728 | 0.715 | 0.699 | 0.709 |
| v2e (3 s) | 0.636 | 0.694 | 0.730 | 0.743 | 0.745 | 0.710 |
| **v3 (2.5 s)** | 0.660 | 0.715 | **0.741** | **0.741** | 0.734 | **0.718** |

This made v3_k25 the earlier submission. A 50/50 blend of v2c and v3 probabilities scored lower
than v3 alone at every k.

**A3. Superseded chain on v3** (`scripts/final_predict.py`, all numbers nested leave-one-match-out on
DFL at k = 2.5): stage 1 0.7412, + stage-2 stacking (5 seeds) 0.7423, + H↔A boundary timing
correction 0.7426 (median shift −1 frame). On v2c the same stacking gave 0.7182 → 0.7296.
Official scorer on the public window: spellF1 0.643, macroF1 0.916, 11 of 17 possessions predicted;
full match: 214 possessions of mean 15.5 s, dead share 0.48. The best public baseline
(`baseline_smooth50`) scores 0.326 / 0.621 there.

v2b per DFL match: 0.741, 0.674, 0.733, 0.716, 0.738, 0.738, 0.699. In v1 the ball-aware dead
model predicted far less dead ball on the competition match (34.7% of graded frames, against 45% in
DFL and 48–52% in every other run) and did poorly on the public window. We read that as domain
shift in the ball features and went ball-free. With the v2 features and three degraded copies the
shift is gone, and the ball-aware dead model is 0.06 better on DFL and 0.06–0.10 on Metrica.

**B. Decoders on the v2b OOF predictions** (DFL, 1,964 true possessions).

| Decoder | spellF1 | macroF1 | Predicted possessions |
|---|---|---|---|
| Per-frame argmax | 0.391 | 0.910 | 5,151 |
| 2-s moving average, then argmax | 0.664 | 0.912 | 2,249 |
| Min-duration Viterbi, `min_len_HA` = 25 | 0.691 | 0.912 | 2,038 |
| **Min-duration Viterbi, best (`min_len_HA` = 75)** | **0.720** | **0.911** | **1,821** |
| Restart-aware 4-state Viterbi (`scripts/lab/decoder_lab.py` search) | 0.7196 | – | – |

**C. Evidence that exists only for the competition match** (v2b frame model).

| Variant | DFL spellF1, simulated signal | Public spellF1 / macroF1 |
|---|---|---|
| none | 0.7198 | 0.657 / 0.909 |
| broadcast, shrink 0.5 | 0.7179 | 0.657 / 0.927 |
| broadcast, shrink 0.7 | 0.7184 | 0.621 / 0.917 |
| broadcast, shrink 1.0 | 0.7182 | 0.621 / 0.915 |
| whistle (tested on v1 runs) | not measurable | spellF1 unchanged |

DFL has no broadcast or audio. `scripts/lab/fusion_sim.py` simulates a "pitch not shown" signal with
the structure measured on the competition match, adds it to the DFL OOF predictions and applies the
same EM fusion. With a decoder tuned without fusion, fusion does not help. The fused public rows
also include the forced `D` before the second-half kick-off, which the "none" row (from
`train_eval.py`) lacks. An earlier submission used broadcast with shrink 0.5, the value that
looked best on the public window; that was a choice on the ground truth and is withdrawn. Final
file: no broadcast, no whistle.

**Submitted file:** `1337.csv`, built by `scripts/run_all.py --tag v5_k05` and checked with the official
`validate_submission.py`.

## 7. Limitations and what we would do with more time

- **Our labels are a proxy for the competition's.** DFL's provider labels may place a turnover
  after a duel, or the start of a stoppage (ball over the line vs whistle), differently from the
  competition's tracking-derived labels. DFL and Metrica scores measure agreement
  with the proxy; the gap to the real definition is unknown.
- **The decoder is tuned on the predictions it is scored on.** The 150-draw search over seven
  parameters uses the same DFL OOF sequences and Metrica games it is scored on, so the A0 numbers
  are optimistic by a small, unknown amount. Nested LOMO for the decoder would fix this.
- **The degradation matches marginal statistics, not joint ones.** Identity switches and false
  ball detections outside gaps are not modelled, and the distribution of tracked players per frame
  differs from the target's (3.1).
- **Turnovers inside long ball gaps.** Without the ball the team model has only shape and movement,
  so a change of possession inside a long gap can be placed late or missed. We have not measured
  how much of the error this accounts for. An own ball detector on the tactical video, to shorten
  the gaps, is what we would build first.
- **Target-only signals cannot be learned.** Their weights are estimated without labels. Replays
  can show the pitch while the ball is dead. The constant per-half offset assumes no time skips in
  the broadcast. Whistles mark stoppages, not restarts. Fusion strength and decoder should be tuned
  jointly on the simulation.
- **Restarts are under-used.** The restart model did not improve the decoder. Smart Tagging set
  pieces could anchor stoppage ends, given a labelled domain to validate them on.
- **Data.** Seven training matches give a per-match spread of 0.674–0.741. More open tracking with
  event-derived labels (built as for Metrica) would make a temporal network over these features
  worth trying.
- **Reproducibility.** LightGBM runs with `deterministic=True`, fixed seeds and pinned versions.
  Verified on 17 Nov 2026: a fresh `scripts/run_all.py --tag repro --from train` (training, decoder
  tuning, final chain) reproduced the submitted file byte for byte (SHA-256 `0325d257…0dbb`).
  Not yet checked on a second machine.

## 8. Reproduction

Needs Python 3.13 (tested 3.13.7) and the official challenge repository with its `data/` bundle
next to this one as `../more-than-a-hack-2027` (or set `MTAH_REPO`, `MTAH_DATA_DIR`,
`POSSESSION_EVAL_DIR`). ffmpeg is only needed for the whistle lab step. Derived files go to
`../extdata/` and `<data>/derived/`, outside version control.

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt
# everything in one go (prints the SHA-256 of results/final_<tag>.csv)
.venv/Scripts/python scripts/run_all.py --tag v5_k05 --skip audio
# or step by step
.venv/Scripts/python scripts/build_dfl.py                # DFL: ~2.6 GB from figshare 28196177
.venv/Scripts/python scripts/build_metrica.py            # Metrica sample games (decoder selection)
.venv/Scripts/python scripts/build_features.py --seeds 0 1 2 --feat-dir features_v4
.venv/Scripts/python scripts/train_eval.py --tag v5_k05 --feat-dir features_v4 --relabel 0.5 --team-relabel 0 --drop ball_acc
.venv/Scripts/python scripts/lab/joint_tune.py --tag v5_k05 --trials 150
.venv/Scripts/python scripts/final_predict.py --tag v5_k05 --joint --out 1337
python ../more-than-a-hack-2027/evaluation/validate_submission.py results/1337.csv
.venv/Scripts/python -m pytest tests
```

## 9. Rules compliance

Checked on 3 Nov 2026 against the organisers' clarification ("tune parameters on external data,
never on the ground-truth possession data or on the provided video"), reviewed again on 17 Nov 2026.

- **Training:** only DFL/Sportec open data (CC BY 4.0). The competition match is only predicted.
- **Every tuned value is chosen on external data:** the decoder (DFL OOF + Metrica, raw labels),
  stacking and the edge shift (nested LOMO on DFL). The public window and the full-match numbers
  published in EVALUATION.md are printed for checks only and enter no choice.
- **Corrected on 3 Nov 2026:** the earlier submission (v3, labels coarsened at 2.5 s, broadcast
  fusion with shrink 0.5) took the coarsening and the shrink from the public window. Both are
  withdrawn: labels are raw, as CHALLENGE.md defines them, and the broadcast fusion is off because
  its weight could only be estimated on the competition match. Lab tools that compared candidates
  with statistics of the ground truth were removed from the repository.
- **Video:** no frame or image of the video is used; no output was checked against the video. The
  audio-whistle detector is a lab experiment and not in the final file.
- **No hand labels** on the competition match, and no data created manually.
- **Provided data and AI services:** no match data, video, labels or per-frame predictions were
  given to an AI service. The code was written with an AI coding assistant that saw code and
  aggregate scores only.
- **Open question for the organisers: input statistics of the competition match.** No labels and
  no video are involved, but these choices look at the competition's automatic tracking:
  - the degradation of DFL reuses ball-gap lengths measured on it (`degrade.gap_profile`,
    30–70% of live gaps), aims at its share of missing ball frames, and sets the ranges of
    smoothing, jitter, dropouts and stoppage gaps so that degraded DFL resembles it
    (`scripts/lab/noise_profile.py` compares the two);
  - `ball_acc` was dropped because its distribution differed most between degraded DFL and the
    competition tracking (`scripts/lab/domain_check.py`, |AUC − 0.5| > 0.15), not because of any
    score; on DFL and Metrica removing it changed nothing measurable (Section 2).
  We read this as using the provided automatic data as input, which the clarification allows,
  rather than tuning on the ground truth or the video. If the organisers see it differently, the
  fallback is the purely parametric gap model already in `degrade.py` (`pool_share` 0) with the
  noise ranges left as they are.
- **Removed lab tools:** scripts that chose or compared label granularities with the public window
  or with numbers about the private ground truth (`cross_k.py`, `label_granularity.py`,
  `anchor_check.py`) are not in the repository; the superseded tables A2/A3 that came from them
  stay in Section 6 for transparency only.
- **Licences:** DFL CC BY 4.0 (cited in README). Metrica sample data: published by Metrica Sports
  on GitHub; used for validation and decoder selection only.
