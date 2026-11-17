"""Whistle evidence for the competition match -> <MTAH data>/derived/BAR-ATH_whistle.npz"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from possession import audio, target  # noqa: E402

derived = os.path.join(target.DATA_DIR, "derived")
pcm = audio.decode_audio(os.path.join(target.DATA_DIR, "tactical", "BAR-ATH_tactical.mp4"),
                         os.path.join(derived, "BAR-ATH_tactical_audio16k.s16"))
graded = np.zeros(target.N_FRAMES, bool)
for _, a, b in target.HALVES:
    graded[a:b + 1] = True
score = audio.whistle_score(pcm, target.N_FRAMES)
ev = audio.whistle_events(score, graded)
recent = audio.whistle_recent(ev, target.N_FRAMES)
np.savez(os.path.join(derived, "BAR-ATH_whistle.npz"), score=score, recent=recent,
         events=np.array(ev, np.int64).reshape(-1, 2))
print("whistle events in graded frames: %d" % sum(graded[a] for a, _ in ev))
