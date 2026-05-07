"""
feature_extraction/pitch.py

Extracts pitch (F0) and pitch velocity (frame-to-frame derivative) from audio.

    extract_pitch(audio, sr) -> tuple[np.ndarray, np.ndarray]
        Returns (pitch_hz, pitch_velocity_hz_s), both shape (T,).

        - pitch_hz          : F0 in Hz;  0.0 for unvoiced frames.
        - pitch_velocity_hz_s: dF0/dt in Hz/s; 0.0 wherever either the current
                               or the previous frame is unvoiced, and at frame 0.

Uses Praat via parselmouth (autocorrelation method) for accurate F0 tracking,
tuned for Carnatic vocal and instrument recordings.

Dependencies: praat-parselmouth
"""

import numpy as np
import parselmouth
from parselmouth.praat import call

# ── Settings ──────────────────────────────────────────────────────────────────
PITCH_FLOOR   = 75.0   # Hz — lower bound for F0 search
PITCH_CEILING = 600.0  # Hz — upper bound for F0 search
HOP_MS        = 10.0   # ms — frame hop size (must match all other features)


def extract_pitch(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract F0 pitch and pitch velocity at 10 ms resolution.

    Parameters
    ----------
    audio : np.ndarray, shape (N,)
        Mono audio signal (float32 or float64).
    sr : int
        Sample rate in Hz.

    Returns
    -------
    pitch_hz : np.ndarray, shape (T,)
        F0 values in Hz. Unvoiced frames are 0.0.
    pitch_velocity_hz_s : np.ndarray, shape (T,)
        Frame-to-frame pitch derivative in Hz/s.
        0.0 at frame 0 and at any voiced/unvoiced boundary.
    """
    hop_s = HOP_MS / 1000.0

    # Build a parselmouth Sound from the numpy array
    snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=float(sr))

    pitch_obj = call(
        snd,
        "To Pitch (ac)",
        hop_s,       # time step
        PITCH_FLOOR,
        15,          # max candidates
        True,        # very accurate
        0.03,        # silence threshold
        0.45,        # voicing threshold
        0.01,        # octave cost
        0.35,        # octave jump cost
        0.14,        # voiced/unvoiced cost
        PITCH_CEILING,
    )

    n_frames = call(pitch_obj, "Get number of frames")

    # ── Extract F0 per frame ───────────────────────────────────────────────────
    pitch_hz = np.zeros(n_frames, dtype=np.float32)
    for i in range(1, n_frames + 1):
        f0 = call(pitch_obj, "Get value in frame", i, "Hertz")
        if not (f0 != f0):   # NaN check without math.isnan
            pitch_hz[i - 1] = float(f0)

    # ── Compute pitch velocity (Hz/s) ─────────────────────────────────────────
    voiced = pitch_hz != 0.0
    velocity = np.zeros(n_frames, dtype=np.float32)

    for i in range(1, n_frames):
        if voiced[i] and voiced[i - 1]:
            velocity[i] = (pitch_hz[i] - pitch_hz[i - 1]) / hop_s

    return pitch_hz, velocity
