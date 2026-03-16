"""
evaluate.py — Carnatic Raga CNN Classifier
===========================================
Mirrors the dataset behaviour in train.py exactly.
Each test audio file is split into 30-second chunks in memory.
Final prediction per file is the majority vote across all its chunks.

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
from collections import Counter

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
from src.features import load_audio, compute_logmel, pad_or_crop_logmel
from src.models import BaselineCNN

# ── paths ─────────────────────────────────────────────────────────────────────
RESULTS_DIR    = "results"
CHECKPOINT     = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
METRICS_FILE   = os.path.join(RESULTS_DIR, "baseline_metrics.json")
REPORT_FILE    = os.path.join(RESULTS_DIR, "classification_report.txt")
CONFUSION_FILE = os.path.join(RESULTS_DIR, "baseline_confusion.png")

os.makedirs(RESULTS_DIR, exist_ok=True)

# ── clip config ───────────────────────────────────────────────────────────────
CLIP_SAMPLES = config.SAMPLE_RATE * config.CLIP_DURATION
FIXED_FRAMES = int(config.CLIP_DURATION * config.SAMPLE_RATE / config.HOP_LENGTH)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset  (mirrors train.py)
# ─────────────────────────────────────────────────────────────────────────────

class RagaDataset(Dataset):
    """
    Loads each audio file once, splits into non-overlapping 30-second chunks,
    and caches all chunks as tensors in memory.

    Parameters
    ----------
    df        : pd.DataFrame   rows for one split
    label_map : dict           { raga_name: int_index }
    """

    def __init__(self, df: pd.DataFrame, label_map: dict):
        self.label_map = label_map
        self._cache    = {}   # track_id → list of (1, N_MELS, T) tensors
        self._index    = []   # list of (track_id, chunk_idx, label)

        print(f"  [dataset] Loading and splitting {len(df)} audio files …")

        for _, row in df.iterrows():
            track_id   = row["track_id"]
            audio_path = os.path.join(AUDIO_ROOT, row["relative_part"])
            label      = label_map[row["raga"]]
            chunks     = self._load_and_split(track_id, audio_path)

            for chunk_idx in range(len(chunks)):
                self._index.append((track_id, chunk_idx, label))

        print(f"  [dataset] Total chunks (samples): {len(self._index)}\n")

    def _load_and_split(self, track_id, audio_path):
        if track_id in self._cache:
            return self._cache[track_id]

        if not os.path.exists(audio_path):
            print(f"  [WARN] File not found: {audio_path}")
            self._cache[track_id] = []
            return []

        try:
            waveform = load_audio(audio_path)
        except Exception as exc:
            print(f"  [WARN] Could not load {audio_path}: {exc}")
            self._cache[track_id] = []
            return []

        n_chunks = len(waveform) // CLIP_SAMPLES
        if n_chunks == 0:
            n_chunks = 1

        tensors = []
        for i in range(n_chunks):
            start = i * CLIP_SAMPLES
            chunk = waveform[start : start + CLIP_SAMPLES]

            if len(chunk) < CLIP_SAMPLES:
                chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))

            log_mel = compute_logmel(chunk)
            log_mel = pad_or_crop_logmel(log_mel, FIXED_FRAMES)
            tensor  = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)
            tensors.append(tensor)

        self._cache[track_id] = tensors
        return tensors

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        track_id, chunk_idx, label = self._index[idx]
        return self._cache[track_id][chunk_idx], label


# ─────────────────────────────────────────────────────────────────────────────
#  load_model
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    """
    Restore BaselineCNN weights from a checkpoint saved by train.py.

    Returns
    -------
    model     : BaselineCNN   in eval mode
    label_map : dict
    meta      : dict
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            "Run  python src/train.py  first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model      = BaselineCNN(num_classes=config.NUM_CLASSES).to(device)
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
    model     : torch.nn.Module,
    loader    : DataLoader,
    dataset   : RagaDataset,
    device    : torch.device,
):
    """
    Run chunk-level inference, then aggregate per-file predictions via
    majority vote.

    Returns
    -------
    true_labels : list[int]   one per audio file
    pred_labels : list[int]   majority-vote prediction per audio file
    """
    model.eval()

    # Collect chunk-level predictions keyed by track_id
    # chunk_preds[track_id] = list of predicted class indices
    # chunk_true[track_id]  = true label (same for all chunks of a file)
    chunk_preds = {}
    chunk_true  = {}

    n_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (spectrograms, labels) in enumerate(loader, start=1):
            spectrograms = spectrograms.to(device)
            preds        = model(spectrograms).argmax(dim=1).cpu().tolist()

            # Map back to track_id using the dataset index
            batch_start = (batch_idx - 1) * loader.batch_size
            for i, (pred, label) in enumerate(zip(preds, labels.tolist())):
                global_idx        = batch_start + i
                if global_idx >= len(dataset._index):
                    break
                track_id, _, _    = dataset._index[global_idx]

                if track_id not in chunk_preds:
                    chunk_preds[track_id] = []
                    chunk_true[track_id]  = label

                chunk_preds[track_id].append(pred)

            print(f"  batch {batch_idx}/{n_batches}", flush=True)

    # Majority vote per file
    true_labels = []
    pred_labels = []

    for track_id, preds in chunk_preds.items():
        majority = Counter(preds).most_common(1)[0][0]
        pred_labels.append(majority)
        true_labels.append(chunk_true[track_id])

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
    cm      : ndarray
    """
    n         = len(class_names)
    idx_range = list(range(n))

    accuracy     = accuracy_score(true_labels, pred_labels)
    macro_f1     = f1_score(true_labels, pred_labels, average="macro",   zero_division=0)
    per_class_f1 = f1_score(true_labels, pred_labels, average=None,
                            labels=idx_range, zero_division=0)
    cm     = confusion_matrix(true_labels, pred_labels, labels=idx_range)
    report = classification_report(
        true_labels, pred_labels,
        target_names=class_names, labels=idx_range, zero_division=0
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
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
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
    print(f"  Device        : {device}")
    print(f"  Clip duration : {config.CLIP_DURATION}s\n")

    # ── Load model ────────────────────────────────────────────────────────────
    model, label_map, _ = load_model(CHECKPOINT, device)

    class_names = [
        raga for raga, _ in sorted(label_map.items(), key=lambda x: x[1])
    ]

    # ── Build test dataset ────────────────────────────────────────────────────
    df = pd.read_csv(config.CSV_PATH)
    df["split"] = df["split"].str.strip().str.lower()
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"[data] Test files : {len(test_df)}\n")
    print("[data] Building test dataset …")
    test_dataset = RagaDataset(test_df, label_map)

    test_loader = DataLoader(
        test_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
    )

    # ── Inference with majority vote ──────────────────────────────────────────
    print("[evaluate] Running chunk-level inference with majority vote …")
    true_labels, pred_labels = evaluate_model(model, test_loader, test_dataset, device)

    print(f"\n[evaluate] Evaluated {len(true_labels)} files "
          f"({len(test_dataset)} chunks total)\n")

    # ── Metrics ───────────────────────────────────────────────────────────────
    metrics, report, cm = compute_metrics(true_labels, pred_labels, class_names)

    print(f"  Accuracy  : {metrics['accuracy'] * 100:.2f}%")
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