"""
histogram.py
Builds a normalised, RMS-weighted pitch histogram.
"""
import numpy as np


def compute_histogram(
    bins: np.ndarray,
    weights: np.ndarray,
    num_bins: int,
) -> np.ndarray:
    """
    Compute a normalised, weighted histogram over pitch bins.

    Parameters
    bins     : np.ndarray, shape (T,) — bin indices (float); NaN = unvoiced
    weights  : np.ndarray, shape (T,) — per-frame weights (e.g. RMS)
    num_bins : int — total number of bins

    Returns
    np.ndarray, shape (num_bins,), dtype float64
        Normalised histogram (sums to 1.0); all-zero vector if no voiced frames.
    """

    if bins.shape[0] != weights.shape[0]:
        raise ValueError(
            f"bins and weights must have the same length, "
            f"got {bins.shape[0]} vs {weights.shape[0]}"
        )

    #remove Nan
    voiced_mask = np.isfinite(bins)
    voiced_bins = bins[voiced_mask].astype(np.int64)
    voiced_weights = weights[voiced_mask].astype(np.float64)

    if voiced_bins.size == 0:
        return np.zeros(num_bins, dtype=np.float64)

    voiced_weights = np.maximum(voiced_weights, 0.0)

    hist = np.bincount(voiced_bins, weights=voiced_weights, minlength=num_bins)
    hist = hist[:num_bins].astype(np.float64)   # trim if any index == num_bins

    total = hist.sum()
    if total > 0.0:
        hist /= total

    return hist