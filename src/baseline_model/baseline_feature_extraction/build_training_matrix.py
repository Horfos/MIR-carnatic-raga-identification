"""
build_training_matrix.py

Purpose:
    Create dataset matrices from processed per-clip feature files.
    Loads per-clip .npz features, concatenates into (36,) vectors,
    encodes raga labels, and saves a single training_matrix.npz.
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
FEATURES_DIR = ROOT / "data" / "processed" / "baseline_features"
METADATA_CSV = ROOT / "data" / "metadata" / "raga_20_dataset_frozen.csv"
OUTPUT_NPZ   = ROOT / "data" / "processed" / "training_matrix.npz"
MODELS_DIR   = ROOT / "models"
ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

EXPECTED_DIM = 36


# ---------------------------------------------------------------------------
# Feature loading
# ---------------------------------------------------------------------------

def load_clip_features(npz_path: Path) -> np.ndarray:
    """
    Load and concatenate baseline features from a single .npz file.

    Feature order (must match pipeline specification):
        1. pitch_histogram_24  (24,)
        2. pitch_stats          (4,)
        3. velocity_stats       (3,)
        4. rms_stats            (3,)
        5. harmonic_stats       (2,)
        ─────────────────────────────
        Total                  (36,)

    Parameters
    ----------
    npz_path : Path
        Path to the clip .npz feature file.

    Returns
    -------
    np.ndarray
        Concatenated feature vector of shape (36,).

    Raises
    ------
    KeyError
        If a required key is missing from the .npz file.
    ValueError
        If the resulting feature vector is not shape (36,).
    """
    data = np.load(npz_path, allow_pickle=True)

    required_keys = [
        "pitch_histogram_24",
        "pitch_stats",
        "velocity_stats",
        "rms_stats",
        "harmonic_stats",
    ]

    for key in required_keys:
        if key not in data:
            raise KeyError(
                f"Missing required key '{key}' in {npz_path}. "
                f"Available keys: {list(data.keys())}"
            )

    pitch_histogram_24 = np.asarray(data["pitch_histogram_24"]).flatten()
    pitch_stats        = np.asarray(data["pitch_stats"]).flatten()
    velocity_stats     = np.asarray(data["velocity_stats"]).flatten()
    rms_stats          = np.asarray(data["rms_stats"]).flatten()
    harmonic_stats     = np.asarray(data["harmonic_stats"]).flatten()

    # Dimension checks
    expected_shapes = {
        "pitch_histogram_24": (24,),
        "pitch_stats":        (4,),
        "velocity_stats":     (3,),
        "rms_stats":          (3,),
        "harmonic_stats":     (2,),
    }
    actual_arrays = {
        "pitch_histogram_24": pitch_histogram_24,
        "pitch_stats":        pitch_stats,
        "velocity_stats":     velocity_stats,
        "rms_stats":          rms_stats,
        "harmonic_stats":     harmonic_stats,
    }
    for key, expected in expected_shapes.items():
        actual = actual_arrays[key].shape
        if actual != expected:
            raise ValueError(
                f"Dimension mismatch for '{key}' in {npz_path}: "
                f"expected {expected}, got {actual}"
            )

    feature_vector = np.concatenate([
        pitch_histogram_24,
        pitch_stats,
        velocity_stats,
        rms_stats,
        harmonic_stats,
    ])

    assert feature_vector.shape == (EXPECTED_DIM,), (
        f"Final feature vector shape mismatch: expected ({EXPECTED_DIM},), "
        f"got {feature_vector.shape} for {npz_path}"
    )

    return feature_vector


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------

def build_training_matrix(
    features_dir: Path = FEATURES_DIR,
    metadata_csv: Path = METADATA_CSV,
    output_npz: Path   = OUTPUT_NPZ,
    encoder_path: Path = ENCODER_PATH,
) -> None:
    """
    Build and save the training matrix from per-clip feature files.

    Steps:
        1. Load metadata CSV.
        2. For each track_id, attempt to load the corresponding .npz.
        3. Skip missing files with a warning.
        4. Concatenate features into (N, 36) matrix.
        5. Encode raga labels with LabelEncoder.
        6. Save training_matrix.npz and label_encoder.joblib.

    Parameters
    ----------
    features_dir : Path
        Directory containing per-clip .npz files.
    metadata_csv : Path
        Path to the frozen dataset metadata CSV.
    output_npz : Path
        Destination path for the training matrix .npz.
    encoder_path : Path
        Destination path for the fitted LabelEncoder.
    """
    print(f"[build_training_matrix] Loading metadata from: {metadata_csv}")
    metadata = pd.read_csv(metadata_csv)

    # Drop unnamed index columns if present
    drop_cols = [c for c in metadata.columns if c.startswith("Unnamed")]
    if drop_cols:
        metadata = metadata.drop(columns=drop_cols)
        print(f"[build_training_matrix] Dropped columns: {drop_cols}")

    required_cols = {"track_id", "raga", "split"}
    missing_cols = required_cols - set(metadata.columns)
    if missing_cols:
        raise ValueError(
            f"Metadata CSV is missing required columns: {missing_cols}. "
            f"Found columns: {list(metadata.columns)}"
        )

    print(f"[build_training_matrix] Total clips in metadata: {len(metadata)}")
    print(f"[build_training_matrix] Split distribution:\n{metadata['split'].value_counts().to_string()}")

    X_list      : list[np.ndarray] = []
    y_list      : list[str]        = []
    ids_list    : list[str]        = []
    splits_list : list[str]        = []

    skipped = 0
    loaded  = 0

    for _, row in metadata.iterrows():
        track_id  : str = str(row["track_id"])
        raga      : str = str(row["raga"])
        split     : str = str(row["split"])

        npz_path = features_dir / f"{track_id}.npz"

        if not npz_path.exists():
            print(f"[WARNING] Feature file not found, skipping: {npz_path}")
            skipped += 1
            continue

        try:
            feature_vector = load_clip_features(npz_path)
        except (KeyError, ValueError, Exception) as exc:
            print(f"[WARNING] Failed to load features for '{track_id}': {exc}. Skipping.")
            skipped += 1
            continue

        X_list.append(feature_vector)
        y_list.append(raga)
        ids_list.append(track_id)
        splits_list.append(split)
        loaded += 1

    print(f"\n[build_training_matrix] Loaded : {loaded} clips")
    print(f"[build_training_matrix] Skipped: {skipped} clips")

    if loaded == 0:
        raise RuntimeError(
            "No clips were successfully loaded. "
            "Check that features_dir contains valid .npz files "
            f"matching track_ids in the metadata CSV.\n"
            f"  features_dir : {features_dir}\n"
            f"  metadata_csv : {metadata_csv}"
        )

    X      = np.stack(X_list,   axis=0)   # (N, 36)
    splits = np.array(splits_list)         # (N,)
    ids    = np.array(ids_list)            # (N,)

    assert X.shape[1] == EXPECTED_DIM, (
        f"Assembled matrix has wrong number of features: "
        f"expected {EXPECTED_DIM}, got {X.shape[1]}"
    )

    # Encode raga labels
    le = LabelEncoder()
    y  = le.fit_transform(y_list)          # (N,)
    class_names = le.classes_              # (20,)

    print(f"\n[build_training_matrix] Feature matrix shape : {X.shape}")
    print(f"[build_training_matrix] Number of classes    : {len(class_names)}")
    print(f"[build_training_matrix] Classes              : {list(class_names)}")

    # Save outputs
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_npz,
        X=X,
        y=y,
        ids=ids,
        splits=splits,
        class_names=class_names,
    )
    print(f"\n[build_training_matrix] Saved training matrix → {output_npz}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(le, encoder_path)
    print(f"[build_training_matrix] Saved label encoder  → {encoder_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    build_training_matrix()


if __name__ == "__main__":
    main()