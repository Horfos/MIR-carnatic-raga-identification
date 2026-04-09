"""
feature_extraction/harmonic.py

Extracts the spectral energy at harmonic positions of Sa (tonic) and Pa
(perfect fifth = tonic × 3/2) for each frame.

    extract_harmonic(audio, sr, tonic_hz) -> np.ndarray, shape (T, H)

In Carnatic music Sa and Pa are the two most structurally significant
svara-s. Tracking the energy at their harmonic series reveals how
prominently those notes are sounding at each moment — useful for
raga characterisation and tonic stability analysis.

Column layout of the returned array (H = 2 × N_HARMONICS):
    cols  0 … N_HARMONICS-1  : Sa harmonics  (tonic × 1, 2, 3, …)
    cols  N_HARMONICS … H-1  : Pa harmonics  (tonic × 1.5 × 1, 2, 3, …)

Dependencies: librosa, numpy
"""

import numpy as np
import librosa

# ── Settings ──────────────────────────────────────────────────────────────────
HOP_MS      = 10.0   # ms — must match all other features
N_FFT       = 2048   # FFT size (~46 ms at 44.1 kHz — fine frequency resolution)
N_HARMONICS = 5      # how many harmonics to track for Sa and Pa each → H = 10


def extract_harmonic(
    audio: np.ndarray,
    sr: int,
    tonic_hz: float,
) -> np.ndarray:
    """
    Compute per-frame spectral energy at Sa and Pa harmonic frequencies.

    Parameters
    ----------
    audio : np.ndarray, shape (N,)
        Mono audio signal.
    sr : int
        Sample rate in Hz.
    tonic_hz : float
        Estimated tonic (Sa) frequency in Hz.

    Returns
    -------
    harmonic : np.ndarray, shape (T, H), dtype float32
        H = 2 × N_HARMONICS (Sa harmonics first, then Pa harmonics).
        Values are magnitude spectrum amplitudes (linear, not dB).
    """
    hop_length = int(sr * HOP_MS / 1000.0)

    # Short-time magnitude spectrum → shape (n_fft//2 + 1, T)
    S = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=hop_length, center=True))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)   # (n_fft//2 + 1,)
    T = S.shape[1]

    # Build target frequencies: Sa harmonics then Pa harmonics
    sa_harmonics = [tonic_hz * k       for k in range(1, N_HARMONICS + 1)]
    pa_harmonics = [tonic_hz * 1.5 * k for k in range(1, N_HARMONICS + 1)]
    target_freqs = sa_harmonics + pa_harmonics   # length H = 2 × N_HARMONICS

    H = len(target_freqs)
    harmonic = np.zeros((T, H), dtype=np.float32)

    for j, f in enumerate(target_freqs):
        if f >= freqs[-1]:
            # Harmonic exceeds Nyquist — leave column as zeros
            continue
        # Nearest FFT bin to the target frequency
        bin_idx = int(np.argmin(np.abs(freqs - f)))
        harmonic[:, j] = S[bin_idx, :].astype(np.float32)

    return harmonic   # (T, H)
