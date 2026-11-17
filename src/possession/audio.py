"""Referee-whistle evidence from the tactical video's audio track.

A whistle is a loud, narrow tone around 3-4 kHz; crowd noise is broadband and stadium
hum is steady. Per video frame we take the band 1.8-5 kHz, subtract each frequency bin's
running background (+/- 10 s), and score how far the strongest bin stands above the band
median ("whitened tonality", dB). Frames above the match's own 99th percentile, merged
into events of >= 0.16 s, are whistles.

Validated locally (aggregates only): events land within 0.3-4.4 s of all four published
kick-off / end-of-half frames and mark the start of most stoppages; restarts usually have
no whistle (IFAB), so this is evidence for *going dead*, not for coming alive.

The audio is decoded from the local video with ffmpeg; nothing leaves the machine.
"""

import os
import subprocess

import numpy as np

SR = 16000
FPS = 25
HOP = SR // FPS
NFFT = 1024
BAND = (1800.0, 5000.0)
BG_HALF_S = 10
CHUNK = 20000


def decode_audio(video, out):
    if not os.path.isfile(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
        subprocess.check_call(["ffmpeg", "-v", "error", "-y", "-i", video, "-vn", "-ac", "1",
                               "-ar", str(SR), "-f", "s16le", out])
    return np.memmap(out, dtype="<i2", mode="r")


def whistle_score(audio, n_frames):
    freqs = np.fft.rfftfreq(NFFT, 1.0 / SR)
    band = (freqs >= BAND[0]) & (freqs <= BAND[1])
    window = np.hanning(NFFT).astype(np.float32)
    offsets = np.arange(NFFT) - NFFT // 2
    logp = np.zeros((n_frames, int(band.sum())), np.float32)
    for start in range(0, n_frames, CHUNK):
        frames = np.arange(start, min(n_frames, start + CHUNK))
        idx = np.clip((frames * HOP + HOP // 2)[:, None] + offsets[None, :], 0, len(audio) - 1)
        power = np.abs(np.fft.rfft(audio[idx].astype(np.float32) * window, axis=1)) ** 2 + 1e-9
        logp[frames] = np.log10(power[:, band])
    coarse = logp[::FPS]
    pad = np.pad(coarse, ((BG_HALF_S, BG_HALF_S), (0, 0)), mode="edge")
    bg = np.median(np.lib.stride_tricks.sliding_window_view(pad, 2 * BG_HALF_S + 1, axis=0), axis=-1)
    nov = logp - np.repeat(bg, FPS, axis=0)[:n_frames]
    s = 10 * (nov.max(1) - np.median(nov, 1))
    return np.median(np.stack([np.roll(s, -1), s, np.roll(s, 1)]), axis=0).astype(np.float32)


def whistle_events(score, graded, pct=99.0, merge_gap=3, min_len=4):
    thr = np.percentile(score[graded], pct)
    mask = score >= thr
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts, stops = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    ev = []
    for a, b in zip(starts, stops):
        if ev and a - ev[-1][1] <= merge_gap:
            ev[-1][1] = b
        else:
            ev.append([a, b])
    return [(a, b) for a, b in ev if b - a >= min_len]


def whistle_recent(events, n_frames, horizon_s=8.0):
    """Per frame: 1.0 right at a whistle onset, decaying linearly to 0 after horizon_s."""
    out = np.zeros(n_frames, np.float32)
    h = int(horizon_s * FPS)
    ramp = np.linspace(1.0, 0.0, h, dtype=np.float32)
    for a, _ in events:
        seg = slice(a, min(n_frames, a + h))
        out[seg] = np.maximum(out[seg], ramp[:seg.stop - a])
    return out
