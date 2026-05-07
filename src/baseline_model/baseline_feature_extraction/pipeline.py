"""
pipeline.py
Orchestrates per-clip feature extraction and full-dataset processing.
Stage 1: converts raw .npz clips → processed per-clip .npz files.
"""

import sys
from pathlib import Path

# Force-add project root to sys.path
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.baseline_model.baseline_feature_extraction.load_npz import load_clip
from src.baseline_model.baseline_feature_extraction.pitch_processing import f0_to_cents
from src.baseline_model.baseline_feature_extraction.binning import bin_pitch
from src.baseline_model.baseline_feature_extraction.histogram import compute_histogram
from src.baseline_model.baseline_feature_extraction.statistics import (
    compute_pitch_stats,
    compute_velocity_stats,
    compute_rms_stats,
    compute_harmonic_stats,
)
from src.baseline_model.baseline_feature_extraction.save_processed import save_processed_clip


def process_clip(npz_path: Path, num_bins: int = 24) -> dict:
    """
    Load a raw clip .npz and compute all processed features and statistics.

    Parameters
    ----------
    npz_path : Path
        Path to the raw clip .npz file.
    num_bins : int
        Number of pitch histogram bins (default 24).

    Returns
    -------
    dict
        Processed data dictionary ready for saving, containing:
        - raw:       f0, rms, velocity, harmonics, tonic
        - processed: f0_cents, voiced_mask, pitch_histogram_24
        - stats:     pitch_stats, velocity_stats, rms_stats, harmonic_stats

    Raises
    ------
    ValueError
        If tonic is zero or negative.
    """
    clip = load_clip(str(npz_path))

    f0: np.ndarray = clip["f0"]
    rms: np.ndarray = clip["rms"]
    velocity: np.ndarray = clip["velocity"]
    harmonics: np.ndarray = clip["harmonics"]
    tonic: float = float(clip["tonic"])

    if tonic <= 0:
        raise ValueError(f"Invalid tonic={tonic!r} in {npz_path}. Tonic must be > 0.")

    # --- processed features ---
    f0_cents: np.ndarray = f0_to_cents(f0, tonic)          # shape (T,)
    voiced_mask: np.ndarray = f0 > 0                        # shape (T,) bool
    bins = bin_pitch(f0_cents, num_bins)
    pitch_histogram_24: np.ndarray = compute_histogram(bins, rms, num_bins)  # shape (num_bins,)

    # --- summary statistics ---
    pitch_stats = compute_pitch_stats(f0_cents, voiced_mask)        # (4,)
    velocity_stats = compute_velocity_stats(velocity, voiced_mask)  # (3,)
    rms_stats = compute_rms_stats(rms)                              # (3,)
    harmonic_stats = compute_harmonic_stats(harmonics)              # (2,)

    return {
        # raw
        "f0": f0,
        "rms": rms,
        "velocity": velocity,
        "harmonics": harmonics,
        "tonic": np.float32(tonic),
        # processed
        "f0_cents": f0_cents,
        "voiced_mask": voiced_mask,
        "pitch_histogram_24": pitch_histogram_24,
        # statistics
        "pitch_stats": pitch_stats,
        "velocity_stats": velocity_stats,
        "rms_stats": rms_stats,
        "harmonic_stats": harmonic_stats,
    }
# data= process_clip("data\interim/npz_features\Abhayamba.npz")
# for key, value in data.items():
#     print(f"{key}: {value.shape if isinstance(value, np.ndarray) else value}")

def process_dataset(
    input_dir: str,
    output_dir: str,
    num_bins: int = 24,
) -> None:
    """
    Iterate over all .npz clips in input_dir and save one processed .npz per clip.

    Parameters
    ----------
    input_dir : str
        Directory containing raw clip .npz files.
    output_dir : str
        Directory where processed clip .npz files will be written.
    num_bins : int
        Number of pitch histogram bins.
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)

    clip_paths = sorted(in_dir.glob("*.npz"))
    if not clip_paths:
        raise FileNotFoundError(f"No .npz files found in: {in_dir}")

    print(f"[INFO] Processing {len(clip_paths)} clips from {in_dir}")

    ok = 0
    errors = 0
    for clip_path in clip_paths:
        try:
            processed = process_clip(clip_path, num_bins=num_bins)
            out_path = out_dir / clip_path.name
            save_processed_clip(str(out_path), processed)
            print(f"  [OK] {clip_path.name}")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARNING] Skipping {clip_path.name}: {exc}")
            errors += 1

    print(f"[INFO] Done. {ok} saved, {errors} skipped.")


def main() -> None:
    input_dir = "data/interim/npz_features"
    output_dir = "data/processed/baseline_features"
    num_bins = 24

    process_dataset(input_dir, output_dir, num_bins=num_bins)


if __name__ == "__main__":
    main()