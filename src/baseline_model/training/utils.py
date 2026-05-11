"""
utils.py

Reusable helper functions shared across all training scripts.

Provides:
    - load_dataset        : load training_matrix.npz
    - split_dataset       : divide arrays by pre-assigned split labels
    - compute_metrics     : accuracy + classification_report
    - plot_confusion_matrix: matplotlib-based confusion matrix
    - save_model          : joblib serialization
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; must be set before pyplot import
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Tuple


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(
    npz_path: Path,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the pre-built training matrix from disk.

    Parameters
    ----------
    npz_path : Path
        Path to training_matrix.npz produced by build_training_matrix.py.

    Returns
    -------
    X : np.ndarray, shape (N, 36)
        Feature matrix.
    y : np.ndarray, shape (N,)
        Integer-encoded raga labels.
    ids : np.ndarray, shape (N,)
        Track identifiers (strings).
    splits : np.ndarray, shape (N,)
        Split labels – one of {'train', 'val', 'test'}.
    class_names : np.ndarray, shape (n_classes,)
        Human-readable raga names corresponding to label indices.

    Raises
    ------
    FileNotFoundError
        If npz_path does not exist.
    KeyError
        If a required key is missing from the .npz archive.
    """
    if not npz_path.exists():
        raise FileNotFoundError(
            f"Training matrix not found at: {npz_path}\n"
            "Run build_training_matrix.py first."
        )

    data = np.load(npz_path, allow_pickle=True)

    required_keys = {"X", "y", "ids", "splits", "class_names"}
    missing = required_keys - set(data.files)
    if missing:
        raise KeyError(
            f"Training matrix is missing keys: {missing}. "
            f"Found: {list(data.files)}"
        )

    X           = data["X"].astype(np.float32)
    y           = data["y"].astype(np.int64)
    ids         = data["ids"]
    splits      = data["splits"]
    class_names = data["class_names"]

    return X, y, ids, splits, class_names


# ---------------------------------------------------------------------------
# Dataset splitting
# ---------------------------------------------------------------------------

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    splits: np.ndarray,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
]:
    """
    Divide arrays into train / val / test partitions using pre-assigned split labels.

    Parameters
    ----------
    X : np.ndarray, shape (N, F)
        Feature matrix.
    y : np.ndarray, shape (N,)
        Label array.
    splits : np.ndarray, shape (N,)
        Per-sample split labels ∈ {'train', 'val', 'test'}.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray
        Feature sub-matrices.
    y_train, y_val, y_test : np.ndarray
        Label sub-arrays.

    Raises
    ------
    ValueError
        If any split partition is empty.
    """
    train_mask = splits == "train"
    val_mask   = splits == "val"
    test_mask  = splits == "test"

    for name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
        if not mask.any():
            raise ValueError(
                f"Split '{name}' is empty. "
                f"Unique split values found: {np.unique(splits).tolist()}"
            )

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]

    print(
        f"[split_dataset] train={len(y_train)}  "
        f"val={len(y_val)}  "
        f"test={len(y_test)}"
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray | None = None,
) -> Tuple[float, str]:
    """
    Compute accuracy and a full classification report.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    class_names : array-like, optional
        Human-readable class names for the classification report.

    Returns
    -------
    accuracy : float
        Overall accuracy ∈ [0, 1].
    report : str
        Formatted classification report string.
    """
    accuracy = float(accuracy_score(y_true, y_pred))

    target_names = list(class_names) if class_names is not None else None

    labels = np.arange(len(class_names))

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )


    return accuracy, report


# ---------------------------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: np.ndarray,
    save_path: Path,
) -> None:
    """
    Compute, plot, and save a confusion matrix using matplotlib only.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth integer labels.
    y_pred : np.ndarray
        Predicted integer labels.
    class_names : np.ndarray
        Class name strings indexed by label integer.
    save_path : Path
        File path where the figure will be saved (e.g. .png).
    """
    labels = np.arange(len(class_names))

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    n_classes = len(class_names)

    fig_size = max(8, n_classes // 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=6,
            )

    fig.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[plot_confusion_matrix] Saved → {save_path}")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: object, path: Path) -> None:
    """
    Serialize a model to disk with joblib.

    Parameters
    ----------
    model : object
        Any sklearn-compatible model or pipeline.
    path : Path
        Destination file path (e.g. models/logreg.joblib).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"[save_model] Saved model → {path}")