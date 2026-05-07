"""
build_training_matrix.py
Stage 2: assemble a training matrix from processed per-clip .npz files.

Usage
-----
    python build_training_matrix.py
"""

from pathlib import Path

import numpy as np


def build_feature_vector(data: dict) -> np.ndarray:
    """
    Concatenate per-clip features into a single flat feature vector.

    Concatenation order
    -------------------
    pitch_histogram_24  (24,)
    pitch_stats          (4,)
    velocity_stats       (3,)
    rms_stats            (3,)
    harmonic_stats       (2,)
    --------------------------------
    total               (36,)

    Parameters
    ----------
    data : dict
        Loaded processed clip dictionary.

    Returns
    -------
    np.ndarray, shape (36,), dtype float32
    """
    parts = [
        data["pitch_histogram_24"].astype(np.float32),  # (24,)
        data["pitch_stats"].astype(np.float32),         # (4,)
        data["velocity_stats"].astype(np.float32),      # (3,)
        data["rms_stats"].astype(np.float32),           # (3,)
        data["harmonic_stats"].astype(np.float32),      # (2,)
    ]
    vector = np.concatenate(parts)
    assert vector.shape == (36,), f"Expected shape (36,), got {vector.shape}"
    return vector


def build_dataset_matrix(
    input_dir: str,
    output_path: str,
) -> None:
    """
    Load all processed clip .npz files and build a dataset matrix.

    Saves
    -----
    X   : np.ndarray, shape (N, 36), dtype float32  — feature matrix
    ids : np.ndarray, shape (N,),    dtype str       — clip stem names

    Parameters
    ----------
    input_dir : str
        Directory containing processed clip .npz files
        (i.e. data/processed/baseline_features/).
    output_path : str
        Destination .npz file for the dataset matrix.
    """
    in_dir = Path(input_dir)
    clip_paths = sorted(in_dir.glob("*.npz"))

    if not clip_paths:
        print(f"[WARNING] No .npz files found in {in_dir}")
        return

    print(f"[INFO] Building matrix from {len(clip_paths)} clips in {in_dir}")

    vectors = []
    ids = []

    for clip_path in clip_paths:
        try:
            data = dict(np.load(clip_path, allow_pickle=False))
            vec = build_feature_vector(data)
            vectors.append(vec)
            ids.append(clip_path.stem)
        except Exception as exc:  # noqa: BLE001
            print(f"  [ERROR] {clip_path.name}: {exc}")

    if not vectors:
        print("[ERROR] No valid clips; matrix not saved.")
        return

    X = np.stack(vectors, axis=0).astype(np.float32)   # (N, 36)
    ids_arr = np.array(ids, dtype=str)                  # (N,)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, ids=ids_arr)

    print(f"[INFO] Saved matrix X={X.shape} to {out}")


def main() -> None:
    input_dir = "data/processed/baseline_features"
    output_path = "data/processed/training_matrix.npz"

    build_dataset_matrix(input_dir, output_path)


if __name__ == "__main__":
    main()