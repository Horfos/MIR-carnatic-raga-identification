"""
train_logreg.py

Baseline Logistic Regression classifier for Carnatic raga identification.

Workflow:
    1. Load dataset (training_matrix.npz)
    2. Split into train / val / test using pre-assigned split column
    3. Standardize features  (scaler fitted ONLY on train split)
    4. Train LogisticRegression(max_iter=5000)
    5. Evaluate on validation split
    6. Evaluate on test split
    7. Save trained model
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
MODEL_PATH       = ROOT / "models" / "logreg.joblib"
RESULTS_DIR      = ROOT / "results"


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_logreg(
    training_matrix: Path = TRAINING_MATRIX,
    model_path: Path       = MODEL_PATH,
    results_dir: Path      = RESULTS_DIR,
) -> None:
    """
    Full Logistic Regression training and evaluation pipeline.

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
    print("[train_logreg] Loading dataset …")
    X, y, ids, splits, class_names = load_dataset(training_matrix)
    print(f"[train_logreg] X shape      : {X.shape}")
    print(f"[train_logreg] Classes      : {list(class_names)}")

    # ------------------------------------------------------------------
    # 2. Split
    # ------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, splits)

    # ------------------------------------------------------------------
    # 3. Feature standardization
    #    Fit scaler ONLY on training data.
    # ------------------------------------------------------------------
    print("[train_logreg] Fitting StandardScaler on training split …")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 4. Train
    # ------------------------------------------------------------------
    print("[train_logreg] Training LogisticRegression(max_iter=5000) …")
    model = LogisticRegression(max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    print("[train_logreg] Training complete.")

    # ------------------------------------------------------------------
    # 5. Evaluate – validation split
    # ------------------------------------------------------------------
    print("\n[train_logreg] ── Validation Evaluation ──")
    val_acc = evaluate_model(
        model=model,
        X_test=X_val,
        y_test=y_val,
        class_names=class_names,
        save_dir=results_dir,
        split_name="val",
    )
    print(f"[train_logreg] Validation accuracy : {val_acc:.4f}")

    # ------------------------------------------------------------------
    # 6. Evaluate – test split
    # ------------------------------------------------------------------
    print("\n[train_logreg] ── Test Evaluation ──")
    test_acc = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        class_names=class_names,
        save_dir=results_dir,
        split_name="test",
    )
    print(f"[train_logreg] Test accuracy       : {test_acc:.4f}")

    # ------------------------------------------------------------------
    # 7. Save model
    # ------------------------------------------------------------------
    save_model(model, model_path)
    print(f"[train_logreg] Model saved → {model_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    train_logreg()


if __name__ == "__main__":
    main()