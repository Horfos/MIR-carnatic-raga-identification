"""
feature_extraction/tonic.py

Estimates the tonic (Sa) frequency of a Carnatic music recording.

    extract_tonic(audio, sr) -> float

Strategy
--------
1.  Run pYIN F0 tracking over the full clip to collect voiced F0 values.
2.  Fold all F0s into a single octave relative to a reference (C2 = 65.4 Hz)
    using octave equivalence, giving a chroma-like pitch-class histogram.
3.  The bin with the highest accumulated weight is taken as the tonic.

This works well for Carnatic recordings where Sa is the single most
sustained and repeated pitch across a performance.

Dependencies: librosa, numpy
"""

import numpy as np
import librosa

# ── Settings ──────────────────────────────────────────────────────────────────
FMIN_HZ      = 60.0    # Hz — lowest F0 to consider (covers male vocalists)
FMAX_HZ      = 500.0   # Hz — highest F0 to consider
HOP_MS       = 10.0    # ms — pYIN hop (matches other features)
N_CHROMA_BINS = 120    # resolution of the folded histogram (10 × standard 12)


def extract_tonic(audio: np.ndarray, sr: int) -> float:
    """
    Estimate the tonic (Sa) frequency of the recording.

    Parameters
    ----------
    audio : np.ndarray, shape (N,)
        Mono audio signal.
    sr : int
        Sample rate in Hz.

    Returns
    -------
    tonic_hz : float
        Estimated tonic frequency in Hz. Returns 0.0 if estimation fails
        (e.g. no voiced frames found).
    """
    hop_length = int(sr * HOP_MS / 1000.0)

    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=FMIN_HZ,
            fmax=FMAX_HZ,
            sr=sr,
            hop_length=hop_length,
        )
    except Exception:
        return 0.0

    # Keep only confidently voiced frames
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) == 0:
        return 0.0

    # ── Octave-fold into a pitch-class histogram ───────────────────────────────
    # Convert Hz → cents relative to FMIN_HZ, then modulo one octave (1200 cents)
    cents_from_ref = 1200.0 * np.log2(voiced_f0 / FMIN_HZ)
    cents_in_octave = cents_from_ref % 1200.0   # fold into [0, 1200)

    hist, bin_edges = np.histogram(
        cents_in_octave,
        bins=N_CHROMA_BINS,
        range=(0.0, 1200.0),
    )

    # Centre of the winning bin in cents
    peak_bin   = int(np.argmax(hist))
    peak_cents = (bin_edges[peak_bin] + bin_edges[peak_bin + 1]) / 2.0

    # Convert back to Hz — pick the octave that best matches the median voiced F0
    median_f0 = float(np.median(voiced_f0))
    tonic_raw = FMIN_HZ * (2.0 ** (peak_cents / 1200.0))

    # Shift to the octave closest to the median voiced F0
    ratio = median_f0 / tonic_raw
    octave_shift = round(np.log2(ratio))
    tonic_hz = tonic_raw * (2.0 ** octave_shift)

    # Clamp to a sensible range
    tonic_hz = float(np.clip(tonic_hz, FMIN_HZ, FMAX_HZ))
    return round(tonic_hz, 3)
