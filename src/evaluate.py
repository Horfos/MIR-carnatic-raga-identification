"""
evaluate.py — Carnatic Raga CNN Classifier
===========================================
Mirrors the dataset behaviour in train.py exactly — log-mel spectrograms are
computed on first access and cached in memory.

Outputs
-------
results/baseline_metrics.json      — accuracy, macro F1, per-class F1
results/classification_report.txt  — sklearn classification report
results/baseline_confusion.png     — confusion matrix heatmap

Usage:
    python src/evaluate.py
"""

import os
import sys
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from env_config import AUDIO_ROOT
from src.features import extract_logmel
from src.models import BaselineCNN

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR    = "results"
CHECKPOINT     = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
METRICS_FILE   = os.path.join(RESULTS_DIR, "baseline_metrics.json")
REPORT_FILE    = os.path.join(RESULTS_DIR, "classification_report.txt")
CONFUSION_FILE = os.path.join(RESULTS_DIR, "baseline_confusion.png")

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset — in-memory cache (mirrors train.py)
# ─────────────────────────────────────────────────────────────────────────────

class RagaDataset(Dataset):
    """
    Loads raw audio and extracts log-mel spectrograms on first access, then
    caches the result in memory.

    Parameters
    ----------
    df        : pd.DataFrame   rows for one split
    label_map : dict           { raga_name: int_index }

    Returns (per sample)
    --------------------
    spectrogram : torch.FloatTensor  shape (1, N_MELS, T)
    label       : int
    """

    def __init__(self, df: pd.DataFrame, label_map: dict):
        self.df        = df.reset_index(drop=True)
        self.label_map = label_map
        self._cache    = {}          # track_id  →  (1, N_MELS, T) FloatTensor

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row      = self.df.iloc[idx]
        track_id = row["track_id"]

        if track_id not in self._cache:
            audio_path = os.path.join(AUDIO_ROOT, row["relative_part"])
            log_mel    = extract_logmel(audio_path)
            self._cache[track_id] = torch.tensor(
                log_mel, dtype=torch.float32
            ).unsqueeze(0)

        tensor = self._cache[track_id]
        label  = self.label_map[row["raga"]]
        return tensor, label


# ─────────────────────────────────────────────────────────────────────────────
#  load_model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """
    Instantiate BaselineCNN and restore weights from a checkpoint saved by
    train.py.

    Returns
    -------
    model     : BaselineCNN   in eval mode
    label_map : dict          { raga_name: int_index }
    meta      : dict          epoch / val_accuracy / train_loss
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run  python src/train.py  first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = BaselineCNN(num_classes=config.NUM_CLASSES).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    label_map = checkpoint["label_map"]
    meta = {
        "epoch"        : checkpoint.get("epoch"),
        "val_accuracy" : checkpoint.get("val_accuracy"),
        "train_loss"   : checkpoint.get("train_loss"),
    }

    print(
        f"[evaluate] Loaded checkpoint — "
        f"epoch {meta['epoch']}, "
        f"val_acc {meta['val_accuracy'] * 100:.2f}%"
    )
    return model, label_map, meta


# ─────────────────────────────────────────────────────────────────────────────
#  evaluate_model
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    model  : torch.nn.Module,
    loader : DataLoader,
    device : torch.device,
):
    """
    Run inference over the full DataLoader.

    Returns
    -------
    true_labels : list[int]
    pred_labels : list[int]
    """
    model.eval()
    true_labels = []
    pred_labels = []

    n_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (spectrograms, labels) in enumerate(loader, start=1):
            spectrograms = spectrograms.to(device)
            preds        = model(spectrograms).argmax(dim=1).cpu().tolist()
            pred_labels.extend(preds)
            true_labels.extend(labels.tolist())

            print(f"  batch {batch_idx}/{n_batches}", flush=True)

    return true_labels, pred_labels


# ─────────────────────────────────────────────────────────────────────────────
#  compute_metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(true_labels: list, pred_labels: list, class_names: list):
    """
    Compute accuracy, macro F1, per-class F1, confusion matrix, and
    sklearn classification report.

    Returns
    -------
    metrics : dict
    report  : str
    cm      : ndarray  (n_classes × n_classes)
    """
    n         = len(class_names)
    idx_range = list(range(n))

    accuracy     = accuracy_score(true_labels, pred_labels)
    macro_f1     = f1_score(true_labels, pred_labels, average="macro",
                            zero_division=0)
    per_class_f1 = f1_score(true_labels, pred_labels, average=None,
                            labels=idx_range, zero_division=0)
    cm     = confusion_matrix(true_labels, pred_labels, labels=idx_range)
    report = classification_report(
        true_labels, pred_labels, target_names=class_names,
        labels=idx_range, zero_division=0
    )

    metrics = {
        "accuracy"     : round(float(accuracy), 6),
        "macro_f1"     : round(float(macro_f1), 6),
        "per_class_f1" : {
            class_names[i]: round(float(per_class_f1[i]), 6)
            for i in range(n)
        },
    }

    return metrics, report, cm


# ─────────────────────────────────────────────────────────────────────────────
#  plot_confusion_matrix
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: str) -> None:
    """Save a labelled confusion-matrix heatmap as PNG."""
    n        = len(class_names)
    fig_size = max(10, n * 0.65)

    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    im      = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks      = np.arange(n),
        yticks      = np.arange(n),
        xticklabels = class_names,
        yticklabels = class_names,
        xlabel      = "Predicted label",
        ylabel      = "True label",
        title       = "Confusion Matrix — Carnatic Raga Classifier",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=7,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[evaluate] Confusion matrix      → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  save_results
# ─────────────────────────────────────────────────────────────────────────────

def save_results(metrics: dict, report: str) -> None:
    """Write metrics JSON and classification report to results/."""
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"[evaluate] Metrics JSON          → {METRICS_FILE}")

    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[evaluate] Classification report → {REPORT_FILE}")


# ─────────────────────────────────────────────────────────────────────────────
#  main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  Carnatic Raga Classifier — Evaluation")
    print("=" * 55)
    print(f"  Device : {device}\n")

    # ── Load model + label map ────────────────────────────────────────────────
    model, label_map, _ = load_model(CHECKPOINT, device)

    # class_names ordered by integer index so sklearn labels align
    class_names = [
        raga for raga, _ in sorted(label_map.items(), key=lambda x: x[1])
    ]

    # ── Build test dataset ────────────────────────────────────────────────────
    df = pd.read_csv(config.CSV_PATH)
    df["split"] = df["split"].str.strip().str.lower()
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"[data]     Test rows : {len(test_df)}\n")

    test_loader = DataLoader(
        RagaDataset(test_df, label_map),
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
    )

    # ── Inference ─────────────────────────────────────────────────────────────
    print("[evaluate] Running inference …")
    true_labels, pred_labels = evaluate_model(model, test_loader, device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics, report, cm = compute_metrics(true_labels, pred_labels, class_names)

    print(f"\n  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    print("\nClassification Report:\n")
    print(report)

    # ── Save ──────────────────────────────────────────────────────────────────
    save_results(metrics, report)
    plot_confusion_matrix(cm, class_names, CONFUSION_FILE)

    print("\n" + "=" * 55)
    print("  Evaluation complete.")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()