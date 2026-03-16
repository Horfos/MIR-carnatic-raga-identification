"""
train.py — Carnatic Raga CNN Classifier
========================================
RagaDataset computes log-mel spectrograms on first access and caches them
in memory.  Epoch 1 pays the librosa cost once; epochs 2-30 are fast.

Usage:
    python src/train.py
"""

import os
import sys
import json
import time

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from env_config import AUDIO_ROOT
from src.features import extract_logmel
from src.models import BaselineCNN

# ── output dirs ───────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
TRAIN_LOG  = os.path.join(config.LOG_DIR, "train_log.json")

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR,        exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset — in-memory cache
# ─────────────────────────────────────────────────────────────────────────────

class RagaDataset(Dataset):
    """
    Loads raw audio and extracts log-mel spectrograms on first access, then
    caches the result in memory.  Subsequent epochs reuse the cache, so
    librosa is only called once per track across the entire training run.

    Parameters
    ----------
    df        : pd.DataFrame   rows for one split (train or val)
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
            log_mel    = extract_logmel(audio_path)          # (N_MELS, T)
            self._cache[track_id] = torch.tensor(
                log_mel, dtype=torch.float32
            ).unsqueeze(0)                                   # (1, N_MELS, T)

        tensor = self._cache[track_id]
        label  = self.label_map[row["raga"]]
        return tensor, label


# ─────────────────────────────────────────────────────────────────────────────
#  Training helpers
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(
    model     : nn.Module,
    loader    : DataLoader,
    criterion : nn.Module,
    optimizer : torch.optim.Optimizer,
    device    : torch.device,
    epoch     : int,
) -> float:
    """
    One full pass over the training DataLoader.
    Prints a progress line after every batch so you know it's alive.

    Returns
    -------
    avg_loss : float
    """
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for batch_idx, (spectrograms, labels) in enumerate(loader, start=1):
        spectrograms = spectrograms.to(device)
        labels       = labels.to(device)

        optimizer.zero_grad()
        logits = model(spectrograms)
        loss   = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        print(
            f"  [epoch {epoch:>3}]  batch {batch_idx}/{n_batches}"
            f"  loss: {loss.item():.4f}",
            flush=True,
        )

    return total_loss / max(n_batches, 1)


def validate(
    model  : nn.Module,
    loader : DataLoader,
    device : torch.device,
) -> float:
    """
    Evaluate on a DataLoader without updating weights.

    Returns
    -------
    accuracy : float
    """
    model.eval()
    correct = 0
    total   = 0

    with torch.no_grad():
        for spectrograms, labels in loader:
            spectrograms = spectrograms.to(device)
            labels       = labels.to(device)

            preds    = model(spectrograms).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    return correct / max(total, 1)


# ─────────────────────────────────────────────────────────────────────────────
#  Main training pipeline
# ─────────────────────────────────────────────────────────────────────────────

def train_model() -> None:
    device = torch.device(config.DEVICE if torch.cuda.is_available() else "cpu")

    print("=" * 55)
    print("  Carnatic Raga Classifier — Training")
    print("=" * 55)
    print(f"  Device        : {device}")
    print(f"  Epochs        : {config.NUM_EPOCHS}")
    print(f"  Batch size    : {config.BATCH_SIZE}")
    print(f"  Learning rate : {config.LEARNING_RATE}")
    print(f"  Num classes   : {config.NUM_CLASSES}")
    print("=" * 55 + "\n")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(config.CSV_PATH)
    df["split"] = df["split"].str.strip().str.lower()

    # ── Label map (sorted → reproducible) ────────────────────────────────────
    all_ragas = sorted(df["raga"].unique())
    assert len(all_ragas) == config.NUM_CLASSES, (
        f"Found {len(all_ragas)} unique ragas but config.NUM_CLASSES={config.NUM_CLASSES}"
    )
    label_map = {raga: idx for idx, raga in enumerate(all_ragas)}

    # ── Splits ────────────────────────────────────────────────────────────────
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)

    print(f"[data] Train rows : {len(train_df)}")
    print(f"[data] Val   rows : {len(val_df)}")
    print(f"\n[cache] Epoch 1 will extract all spectrograms via librosa.")
    print(f"[cache] Epochs 2-{config.NUM_EPOCHS} will use the in-memory cache.\n")

    train_loader = DataLoader(
        RagaDataset(train_df, label_map),
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )
    val_loader = DataLoader(
        RagaDataset(val_df, label_map),
        batch_size  = config.BATCH_SIZE,
        shuffle     = False,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )

    # ── Model / loss / optimiser ──────────────────────────────────────────────
    model     = BaselineCNN(num_classes=config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    history      = []

    for epoch in range(1, config.NUM_EPOCHS + 1):
        t0         = time.time()
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_acc    = validate(model, val_loader, device)
        elapsed    = time.time() - t0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch"            : epoch,
                    "model_state_dict" : model.state_dict(),
                    "val_accuracy"     : val_acc,
                    "train_loss"       : train_loss,
                    "label_map"        : label_map,
                },
                CHECKPOINT,
            )
            marker = "  ◀ best saved"
        else:
            marker = ""

        print(
            f"\nEpoch [{epoch:>3}/{config.NUM_EPOCHS}]  "
            f"Train Loss: {train_loss:.4f}  "
            f"Val Acc: {val_acc * 100:>6.2f}%  "
            f"({elapsed:.1f}s)"
            f"{marker}\n"
        )

        history.append({
            "epoch"        : epoch,
            "train_loss"   : round(train_loss, 6),
            "val_accuracy" : round(val_acc,    6),
        })

    # ── Save training log ──────────────────────────────────────────────────────
    with open(TRAIN_LOG, "w") as f:
        json.dump(
            {"history": history, "best_val_accuracy": round(best_val_acc, 6)},
            f, indent=2,
        )

    print("=" * 55)
    print(f"  Best val accuracy : {best_val_acc * 100:.2f}%")
    print(f"  Checkpoint        : {CHECKPOINT}")
    print(f"  Training log      : {TRAIN_LOG}")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_model()