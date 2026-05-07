"""
pitch_processing.py
Converts F0 (Hz) to tonic-normalized cents.
"""

import numpy as np

def f0_to_cents(f0: np.ndarray, tonic: float) -> np.ndarray:
    """
    Convert an F0 array (Hz) to cents relative to the given tonic.

    Unvoiced frames (f0 <= 0) are set to NaN.

    Parameters
    f0    : np.ndarray, shape (T,) — F0 values in Hz
    tonic : float — reference tonic frequency in Hz (must be > 0)

    Returns
    np.ndarray, shape (T,)
        Cents values; NaN where f0 was <= 0.
    """
    if tonic <= 0:
        raise ValueError(f"Tonic must be > 0 Hz, got: {tonic}")

    f0 = f0.astype(np.float64, copy=True)

    # Mask unvoiced frames
    f0[f0 <= 0] = np.nan

    # cents = 1200 * log2(f0 / tonic)
    with np.errstate(invalid="ignore"):          # silence NaN log warnings
        cents = 1200.0 * np.log2(f0 / tonic)

    return cents