"""
statistics.py
Summary statistics computed from processed clip features.
All outputs are float32 arrays.
"""

import numpy as np


def compute_pitch_stats(
    f0_cents: np.ndarray,
    voiced_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute pitch summary statistics over voiced frames.

    Parameters
    ----------
    f0_cents : np.ndarray, shape (T,)
        Pitch in cents relative to tonic.
    voiced_mask : np.ndarray, shape (T,), dtype bool
        True for voiced frames.

    Returns
    -------
    np.ndarray, shape (4,), dtype float32
        [mean_pitch, std_pitch, median_pitch, pitch_range]
    """
    voiced_cents = f0_cents[voiced_mask]
    # Drop NaNs just in case
    voiced_cents = voiced_cents[~np.isnan(voiced_cents)]

    if voiced_cents.size == 0:
        return np.zeros(4, dtype=np.float32)

    mean_pitch = float(np.mean(voiced_cents))
    std_pitch = float(np.std(voiced_cents))
    median_pitch = float(np.median(voiced_cents))
    pitch_range = float(np.max(voiced_cents) - np.min(voiced_cents))

    return np.array([mean_pitch, std_pitch, median_pitch, pitch_range], dtype=np.float32)


def compute_velocity_stats(
    velocity: np.ndarray,
    voiced_mask: np.ndarray,
) -> np.ndarray:
    """
    Compute velocity summary statistics over voiced frames.

    Parameters
    ----------
    velocity : np.ndarray, shape (T,)
    voiced_mask : np.ndarray, shape (T,), dtype bool

    Returns
    -------
    np.ndarray, shape (3,), dtype float32
        [mean_abs_velocity, std_velocity, max_abs_velocity]
    """
    voiced_vel = velocity[voiced_mask]

    if voiced_vel.size == 0:
        return np.zeros(3, dtype=np.float32)

    mean_abs = float(np.mean(np.abs(voiced_vel)))
    std_vel = float(np.std(voiced_vel))
    max_abs = float(np.max(np.abs(voiced_vel)))

    return np.array([mean_abs, std_vel, max_abs], dtype=np.float32)


def compute_rms_stats(
    rms: np.ndarray,
) -> np.ndarray:
    """
    Compute RMS summary statistics over all frames.

    Parameters
    ----------
    rms : np.ndarray, shape (T,)

    Returns
    -------
    np.ndarray, shape (3,), dtype float32
        [mean_rms, std_rms, max_rms]
    """
    if rms.size == 0:
        return np.zeros(3, dtype=np.float32)

    mean_rms = float(np.mean(rms))
    std_rms = float(np.std(rms))
    max_rms = float(np.max(rms))

    return np.array([mean_rms, std_rms, max_rms], dtype=np.float32)


def compute_harmonic_stats(
    harmonics: np.ndarray,
) -> np.ndarray:
    """
    Compute harmonic energy summary statistics over all frames.

    Parameters
    ----------
    harmonics : np.ndarray, shape (T, 10)

    Returns
    -------
    np.ndarray, shape (2,), dtype float32
        [mean_harmonic_energy, std_harmonic_energy]
    """
    if harmonics.size == 0:
        return np.zeros(2, dtype=np.float32)

    harmonic_energy = harmonics.sum(axis=1)  # shape (T,)
    mean_he = float(np.mean(harmonic_energy))
    std_he = float(np.std(harmonic_energy))

    return np.array([mean_he, std_he], dtype=np.float32)