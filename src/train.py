"""
train.py — Carnatic Raga CNN Classifier
========================================
Each audio file is loaded once, split into non-overlapping 30-second chunks
in memory, and each chunk is treated as an independent training sample.
No files are written to disk.

Pipeline per audio file
-----------------------
  load_audio(path)  →  full waveform
  split into N × (SAMPLE_RATE * CLIP_DURATION) samples
  compute_logmel(chunk)  →  (128, T)
  pad_or_crop_logmel     →  (128, fixed_T)
  cache all chunks in memory as (1, 128, fixed_T) tensors

Usage:
    python src/train.py
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ── project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config
from env_config import AUDIO_ROOT
from src.features import load_audio, compute_logmel, pad_or_crop_logmel
from src.models import BaselineCNN

# ── output dirs ───────────────────────────────────────────────────────────────
CHECKPOINT = os.path.join(config.CHECKPOINT_DIR, "best_model.pt")
TRAIN_LOG  = os.path.join(config.LOG_DIR, "train_log.json")

os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR,        exist_ok=True)

# ── clip config ───────────────────────────────────────────────────────────────
CLIP_SAMPLES  = config.SAMPLE_RATE * config.CLIP_DURATION   # samples per 30-sec chunk
FIXED_FRAMES  = int(config.CLIP_DURATION * config.SAMPLE_RATE / config.HOP_LENGTH)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset
# ─────────────────────────────────────────────────────────────────────────────

class RagaDataset(Dataset):
    """
    Loads each audio file once, splits it into non-overlapping 30-second
    chunks in memory, and caches all chunks as tensors.

    The index is a flat list of (track_id, chunk_index) pairs, so each chunk
    is treated as an independent sample by the DataLoader.

    Parameters
    ----------
    df        : pd.DataFrame   rows for one split
    label_map : dict           { raga_name: int_index }
    """

    def __init__(self, df: pd.DataFrame, label_map: dict):
        self.label_map = label_map
        self._cache    = {}   # track_id → list of (1, N_MELS, T) tensors

        # Build flat index: list of (track_id, chunk_idx, label)
        self._index = []

        print(f"  [dataset] Loading and splitting {len(df)} audio files …")

        for _, row in df.iterrows():
            track_id   = row["track_id"]
            audio_path = os.path.join(AUDIO_ROOT, row["relative_part"])
            label      = label_map[row["raga"]]

            chunks = self._load_and_split(track_id, audio_path)

            for chunk_idx in range(len(chunks)):
                self._index.append((track_id, chunk_idx, label))

        print(f"  [dataset] Total chunks (samples): {len(self._index)}\n")

    def _load_and_split(self, track_id: str, audio_path: str):
        """
        Load full waveform, split into 30-sec chunks, compute and cache
        log-mel tensors.  Returns the list of tensors for this track.
        """
        if track_id in self._cache:
            return self._cache[track_id]

        if not os.path.exists(audio_path):
            print(f"  [WARN] File not found, skipping: {audio_path}")
            self._cache[track_id] = []
            return []

        try:
            waveform = load_audio(audio_path)        # full waveform, shape (N,)
        except Exception as exc:
            print(f"  [WARN] Could not load {audio_path}: {exc}")
            self._cache[track_id] = []
            return []

        # Split into non-overlapping CLIP_SAMPLES-length chunks
        # Discard the final partial chunk if shorter than CLIP_SAMPLES
        n_chunks = len(waveform) // CLIP_SAMPLES
        if n_chunks == 0:
            # File shorter than one clip — pad to full clip length
            n_chunks = 1

        tensors = []
        for i in range(n_chunks):
            start = i * CLIP_SAMPLES
            end   = start + CLIP_SAMPLES
            chunk = waveform[start:end]

            # Pad if this chunk is shorter than CLIP_SAMPLES
            if len(chunk) < CLIP_SAMPLES:
                chunk = np.pad(chunk, (0, CLIP_SAMPLES - len(chunk)))

            log_mel = compute_logmel(chunk)                          # (N_MELS, T)
            log_mel = pad_or_crop_logmel(log_mel, FIXED_FRAMES)     # (N_MELS, fixed_T)
            tensor  = torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0)  # (1, N_MELS, T)
            tensors.append(tensor)

        self._cache[track_id] = tensors
        return tensors

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        track_id, chunk_idx, label = self._index[idx]
        tensor = self._cache[track_id][chunk_idx]
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
    Prints a progress line after every batch.

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
    print(f"  Clip duration : {config.CLIP_DURATION}s")
    print("=" * 55 + "\n")

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(config.CSV_PATH)
    df["split"] = df["split"].str.strip().str.lower()

    # ── Label map ─────────────────────────────────────────────────────────────
    all_ragas = sorted(df["raga"].unique())
    assert len(all_ragas) == config.NUM_CLASSES, (
        f"Found {len(all_ragas)} unique ragas but config.NUM_CLASSES={config.NUM_CLASSES}"
    )
    label_map = {raga: idx for idx, raga in enumerate(all_ragas)}

    # ── Splits ────────────────────────────────────────────────────────────────
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"].reset_index(drop=True)

    print(f"[data] Train files : {len(train_df)}")
    print(f"[data] Val   files : {len(val_df)}\n")

    # Datasets — all audio is loaded and split here, before training starts
    print("[data] Building train dataset …")
    train_dataset = RagaDataset(train_df, label_map)

    print("[data] Building val dataset …")
    val_dataset   = RagaDataset(val_df,   label_map)

    print(f"[data] Train chunks : {len(train_dataset)}")
    print(f"[data] Val   chunks : {len(val_dataset)}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size  = config.BATCH_SIZE,
        shuffle     = True,
        num_workers = 0,
        pin_memory  = (device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
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