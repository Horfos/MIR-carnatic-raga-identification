"""
train_random_forest.py

Baseline Random Forest classifier for Carnatic raga identification.

Workflow:
    1. Load dataset (training_matrix.npz)
    2. Split into train / val / test using pre-assigned split column
    3. NO feature scaling (trees are scale-invariant)
    4. Train RandomForestClassifier(n_estimators=200, random_state=42)
    5. Evaluate on validation split
    6. Evaluate on test split
    7. Save trained model
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.baseline_model.training.utils import (
    load_dataset,
    split_dataset,
    save_model,
)
from src.baseline_model.training.evaluate import evaluate_model


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAINING_MATRIX = ROOT / "data" / "processed" / "training_matrix.npz"
MODEL_PATH       = ROOT / "models" / "random_forest.joblib"
RESULTS_DIR      = ROOT / "results"


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_random_forest(
    training_matrix: Path = TRAINING_MATRIX,
    model_path: Path       = MODEL_PATH,
    results_dir: Path      = RESULTS_DIR,
) -> None:
    """
    Full Random Forest training and evaluation pipeline.

    No feature scaling is applied – tree-based models are invariant to
    monotonic feature transformations.

    Parameters
    ----------
    training_matrix : Path
        Path to training_matrix.npz built by build_training_matrix.py.
    model_path : Path
        Destination path for the serialized model.
    results_dir : Path
        Directory where evaluation artefacts (confusion matrices) are saved.
    """
    # ------------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------------
    print("[train_random_forest] Loading dataset …")
    X, y, ids, splits, class_names = load_dataset(training_matrix)
    print(f"[train_random_forest] X shape      : {X.shape}")
    print(f"[train_random_forest] Classes      : {list(class_names)}")

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, splits)

    # ------------------------------------------------------------------
    # 3. No scaling for Random Forest
    # ------------------------------------------------------------------
    print("[train_random_forest] Skipping feature scaling (not required for RF).")

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("[train_random_forest] Training RandomForestClassifier(n_estimators=200) …")
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    print("[train_random_forest] Training complete.")

    # Feature importance summary (top 10)
    importances = model.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("\n[train_random_forest] Top-10 feature importances (by index):")
    for rank, idx in enumerate(top_idx, start=1):
        print(f"  {rank:2d}. feature[{idx:2d}]  importance={importances[idx]:.4f}")

    # ------------------------------------------------------------------
    # 5. Evaluate – validation split
    # ------------------------------------------------------------------
    print("\n[train_random_forest] ── Validation Evaluation ──")
    val_acc = evaluate_model(
        model=model,
        X_test=X_val,
        y_test=y_val,
        class_names=class_names,
        save_dir=results_dir,
        split_name="val",
    )
    print(f"[train_random_forest] Validation accuracy : {val_acc:.4f}")

    # ------------------------------------------------------------------
    # 6. Evaluate – test split
    # ------------------------------------------------------------------
    print("\n[train_random_forest] ── Test Evaluation ──")
    test_acc = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        save_dir=results_dir,
        split_name="test",
    )
    print(f"[train_random_forest] Test accuracy       : {test_acc:.4f}")

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    save_model(model, model_path)
    print(f"[train_random_forest] Model saved → {model_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    train_random_forest()


if __name__ == "__main__":
    main()