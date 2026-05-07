"""
feature_extraction/energy.py

Extracts frame-level RMS energy from audio.

    extract_energy(audio, sr) -> np.ndarray, shape (T,)

Energy is computed using a short-time window centred on each 10 ms frame.
The window length is set to 25 ms (a standard speech/music analysis window)
so that each frame captures enough signal for a stable RMS estimate while
remaining aligned with the 10 ms hop used by every other feature.

Dependencies: librosa
"""

import numpy as np
import librosa

# ── Settings ──────────────────────────────────────────────────────────────────
HOP_MS    = 10.0   # ms — must match all other features
WINDOW_MS = 25.0   # ms — analysis window length


def extract_energy(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Compute per-frame RMS energy at 10 ms resolution.

    Parameters
    ----------
    audio : np.ndarray, shape (N,)
        Mono audio signal (float32 or float64).
    sr : int
        Sample rate in Hz.

    Returns
    -------
    energy : np.ndarray, shape (T,), dtype float32
        RMS energy per frame, range [0, 1] (unnormalised linear scale).
    """
    hop_length  = int(sr * HOP_MS    / 1000.0)
    frame_length = int(sr * WINDOW_MS / 1000.0)

    # librosa.feature.rms returns shape (1, T)
    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )[0]  # → (T,)

    return rms.astype(np.float32)
