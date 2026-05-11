"""
train_CNN.py

Exploratory 1D MLP baseline (referred to as CNN in the project) for
Carnatic raga identification using only the 24-bin pitch histogram.

Architecture:
    Linear(24 → 64) → ReLU → Dropout(0.3)
    Linear(64 → 32) → ReLU
    Linear(32 → 20)

Training:
    Loss      : CrossEntropyLoss
    Optimizer : Adam
    Epochs    : 30
    Batch size: 16

Workflow:
    1. Load dataset (training_matrix.npz)
    2. Extract pitch_histogram_24 features only  (first 24 dims)
    3. Split into train / val / test using pre-assigned split column
    4. Standardize features  (scaler fitted ONLY on train split)
    5. Train for 30 epochs, printing epoch loss & validation accuracy
    6. Evaluate on test split
    7. Save trained model
"""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.baseline_model.training.utils import (
    load_dataset,
    split_dataset,
    compute_metrics,
    plot_confusion_matrix,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TRAINING_MATRIX = ROOT / "data" / "processed" / "training_matrix.npz"
MODEL_PATH       = ROOT / "models" / "cnn_baseline.pt"
RESULTS_DIR      = ROOT / "results"

# Hyperparameters
INPUT_DIM   = 24
HIDDEN1_DIM = 64
HIDDEN2_DIM = 32
NUM_CLASSES = 20
DROPOUT     = 0.3
EPOCHS      = 30
BATCH_SIZE  = 16
LR          = 1e-3

# Index range for pitch_histogram_24 within the full 36-dim vector
PITCH_HIST_START = 0
PITCH_HIST_END   = 24   # exclusive


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class RagaCNN(nn.Module):
    """
    Lightweight 1D fully-connected baseline operating on the 24-bin
    pitch histogram.

    Despite the 'CNN' name (per project spec), this is an MLP whose
    architecture is suitable as a simple deep baseline.
    """

    def __init__(
        self,
        input_dim: int   = INPUT_DIM,
        hidden1: int     = HIDDEN1_DIM,
        hidden2: int     = HIDDEN2_DIM,
        num_classes: int = NUM_CLASSES,
        dropout: float   = DROPOUT,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        sklearn-compatible predict interface accepting numpy arrays.

        Parameters
        ----------
        X : np.ndarray, shape (N, input_dim)

        Returns
        -------
        np.ndarray, shape (N,)
            Predicted class indices.
        """
        self.eval()
        with torch.no_grad():
            tensor = torch.tensor(X, dtype=torch.float32)
            logits = self.forward(tensor)
            preds  = logits.argmax(dim=1).numpy()
        return preds


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
) -> DataLoader:
    """Build a DataLoader from numpy arrays."""
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)
    ds  = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _validate(
    model: RagaCNN,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
) -> float:
    """Run inference on validation set and return accuracy."""
    model.eval()
    X_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_val, dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(X_t)
        preds  = logits.argmax(dim=1)
        correct = (preds == y_t).sum().item()

    return correct / len(y_t)


def _plot_training_curves(
    train_losses: list[float],
    val_accs: list[float],
    save_dir: Path,
) -> None:
    """Save training-loss and validation-accuracy curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, marker="o", markersize=3)
    ax1.set_title("Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_accs, marker="o", markersize=3, color="darkorange")
    ax2.set_title("Validation Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = save_dir / "cnn_training_curves.png"
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[train_CNN] Training curves saved → {save_path}")


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_cnn(
    training_matrix: Path = TRAINING_MATRIX,
    model_path: Path       = MODEL_PATH,
    results_dir: Path      = RESULTS_DIR,
    epochs: int            = EPOCHS,
    batch_size: int        = BATCH_SIZE,
    lr: float              = LR,
    num_classes: int       = NUM_CLASSES,
) -> None:
    """
    Full CNN (MLP) training and evaluation pipeline.

    Only the 24-bin pitch histogram (first 24 features) is used as input.

    Parameters
    ----------
    training_matrix : Path
        Path to training_matrix.npz built by build_training_matrix.py.
    model_path : Path
        Destination path for the serialized PyTorch model state dict.
    results_dir : Path
        Directory where evaluation artefacts are saved.
    epochs : int
        Number of training epochs.
    batch_size : int
        Mini-batch size.
    lr : float
        Adam learning rate.
    num_classes : int
        Number of output raga classes.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path  = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train_CNN] Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load full dataset
    # ------------------------------------------------------------------
    print("[train_CNN] Loading dataset …")
    X_full, y, ids, splits, class_names = load_dataset(training_matrix)
    print(f"[train_CNN] Full X shape : {X_full.shape}")
    print(f"[train_CNN] Classes      : {list(class_names)}")

    # ------------------------------------------------------------------
    # 2. Extract pitch_histogram_24 (first 24 dims)
    # ------------------------------------------------------------------
    X = X_full[:, PITCH_HIST_START:PITCH_HIST_END]
    print(f"[train_CNN] Pitch histogram slice shape : {X.shape}")
    assert X.shape[1] == INPUT_DIM, (
        f"Expected pitch histogram dim {INPUT_DIM}, got {X.shape[1]}"
    )

    # Update num_classes from data if different
    n_classes_data = int(y.max()) + 1
    if n_classes_data != num_classes:
        print(
            f"[train_CNN] WARNING: num_classes override "
            f"{num_classes} → {n_classes_data} (from data)"
        )
        num_classes = n_classes_data

    # ------------------------------------------------------------------
    # 3. Split
    # ------------------------------------------------------------------
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, splits)

    # ------------------------------------------------------------------
    # 4. Feature standardization – fit ONLY on training split
    # ------------------------------------------------------------------
    print("[train_CNN] Fitting StandardScaler on training split …")
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    # ------------------------------------------------------------------
    # 5. Build model, loss, optimizer
    # ------------------------------------------------------------------
    model = RagaCNN(
        input_dim=INPUT_DIM,
        hidden1=HIDDEN1_DIM,
        hidden2=HIDDEN2_DIM,
        num_classes=num_classes,
        dropout=DROPOUT,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_loader = _make_dataloader(X_train, y_train, batch_size=batch_size, shuffle=True)

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print(f"\n[train_CNN] Starting training for {epochs} epochs …")
    train_losses : list[float] = []
    val_accs     : list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        val_acc  = _validate(model, X_val, y_val, device)

        train_losses.append(avg_loss)
        val_accs.append(val_acc)

        print(
            f"  Epoch [{epoch:3d}/{epochs}]  "
            f"loss={avg_loss:.4f}  "
            f"val_acc={val_acc:.4f} ({val_acc * 100:.2f}%)"
        )

    # ------------------------------------------------------------------
    # 7. Training curves
    # ------------------------------------------------------------------
    _plot_training_curves(train_losses, val_accs, results_dir)

    # ------------------------------------------------------------------
    # 8. Final validation evaluation
    # ------------------------------------------------------------------
    print("\n[train_CNN] ── Validation Evaluation ──")
    y_val_pred = model.predict(X_val)
    val_acc_final, val_report = compute_metrics(y_val, y_val_pred, class_names=class_names)
    print(f"  Validation accuracy : {val_acc_final:.4f} ({val_acc_final * 100:.2f}%)")
    print(val_report)

    plot_confusion_matrix(
        y_true=y_val,
        y_pred=y_val_pred,
        class_names=class_names,
        save_path=results_dir / "cnn_val_confusion_matrix.png",
    )

    # ------------------------------------------------------------------
    # 9. Test evaluation
    # ------------------------------------------------------------------
    print("\n[train_CNN] ── Test Evaluation ──")
    y_test_pred = model.predict(X_test)
    test_acc, test_report = compute_metrics(y_test, y_test_pred, class_names=class_names)
    print(f"  Test accuracy : {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(test_report)

    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_test_pred,
        class_names=class_names,
        save_path=results_dir / "cnn_test_confusion_matrix.png",
    )

    # ------------------------------------------------------------------
    # 10. Save model state dict
    # ------------------------------------------------------------------
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim":        INPUT_DIM,
            "hidden1":          HIDDEN1_DIM,
            "hidden2":          HIDDEN2_DIM,
            "num_classes":      num_classes,
            "dropout":          DROPOUT,
            "scaler_mean":      scaler.mean_,
            "scaler_scale":     scaler.scale_,
            "class_names":      class_names,
        },
        model_path,
    )
    print(f"\n[train_CNN] Model saved → {model_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    train_cnn()


if __name__ == "__main__":
    main()