"""
binning.py
Maps tonic-normalised cents into discrete octave bins.
"""

import numpy as np


def bin_pitch(cents: np.ndarray, num_bins: int) -> np.ndarray:
    """
    Wrap cents into one octave [0, 1200) and assign each frame to a bin.

    Parameters
    cents    : np.ndarray, shape (T,) — cents relative to tonic (may contain NaN)
    num_bins : int — number of equal-width bins covering one octave

    Returns
    np.ndarray, shape (T,), dtype float64
        Bin indices (0 … num_bins-1); NaN where input was NaN.
    """

    if num_bins <= 0:
        raise ValueError(f"num_bins must be > 0, got: {num_bins}")

    bin_size = 1200.0 / num_bins

    cents = cents.astype(np.float64, copy=True)

    wrapped = cents % 1200.0     

    bin_indices = np.floor(wrapped / bin_size)

    # Clamp rounding artefacts at the upper boundary
    bin_indices = np.where(
        np.isfinite(bin_indices),
        np.clip(bin_indices, 0, num_bins - 1),
        np.nan,
    )

    return bin_indices