"""
evaluate.py

Shared evaluation routine for all baseline classifiers.

Provides:
    evaluate_model(model, X_test, y_test, class_names, save_dir, split_name)
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from src.baseline_model.training.utils import (
    compute_metrics,
    plot_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(
    model: object,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: np.ndarray,
    save_dir: Path,
    split_name: str = "test",
) -> float:
    """
    Evaluate a trained classifier and persist diagnostics.

    Steps:
        1. Generate predictions from the model.
        2. Compute accuracy and classification report.
        3. Print the report to stdout.
        4. Plot and save a confusion matrix image.

    Parameters
    ----------
    model : object
        Fitted classifier exposing a `.predict(X)` method.
        For PyTorch models, pass a callable wrapper that accepts numpy arrays.
    X_test : np.ndarray, shape (N, F)
        Feature matrix for the evaluation split.
    y_test : np.ndarray, shape (N,)
        Ground-truth integer labels.
    class_names : np.ndarray, shape (n_classes,)
        Human-readable class names indexed by integer label.
    save_dir : Path
        Directory where results (confusion matrix image) will be saved.
    split_name : str, optional
        Identifier for the split being evaluated – used in filenames and
        printed output. Typical values: ``"val"``, ``"test"``.

    Returns
    -------
    accuracy : float
        Overall accuracy on the evaluation split ∈ [0, 1].
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test)
    y_pred = np.asarray(y_pred)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    accuracy, report = compute_metrics(y_test, y_pred, class_names=class_names)

    print(f"\n{'=' * 60}")
    print(f"  Evaluation  |  split = {split_name}")
    print(f"{'=' * 60}")
    print(f"  Accuracy : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"\n{report}")

    # ------------------------------------------------------------------
    # Infer model name for filename prefix
    # ------------------------------------------------------------------
    model_tag = _infer_model_tag(model)
    cm_filename = f"{model_tag}_{split_name}_confusion_matrix.png"
    cm_save_path = save_dir / cm_filename

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        save_path=cm_save_path,
    )

    return accuracy


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_model_tag(model: object) -> str:
    """
    Derive a short snake_case tag from the model's class name.

    Examples
    --------
    LogisticRegression  → logreg
    SVC                 → svm
    RandomForestClassifier → random_forest
    RagaCNN / nn.Module → cnn
    """
    class_name = type(model).__name__.lower()

    tag_map = {
        "logisticregression": "logreg",
        "svc": "svm",
        "randomforestclassifier": "random_forest",
    }

    for key, tag in tag_map.items():
        if key in class_name:
            return tag

    # PyTorch modules
    if "cnn" in class_name or "module" in class_name or "net" in class_name:
        return "cnn"

    # Fallback: use raw class name with spaces replaced
    return class_name.replace(" ", "_")


# ---------------------------------------------------------------------------
# Entry point (standalone smoke-test)
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Smoke test: not intended for production use.
    Run individual train_*.py scripts instead.
    """
    print("[evaluate.py] This module is intended to be imported, not run directly.")
    print("  Use train_logreg.py / train_svm.py / train_random_forest.py / train_CNN.py.")


if __name__ == "__main__":
    main()