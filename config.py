# config.py

# =========================
# Dataset
# =========================
CSV_PATH = "data/metadata/raga_20_dataset_frozen.csv"
SAMPLE_RATE = 22050
CLIP_DURATION = 10  # (seconds)
NUM_CLASSES = 20

# =========================
# Feature Extraction
# =========================
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 128
FMIN = 0
FMAX = SAMPLE_RATE // 2

# =========================
# Training
# =========================
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 30
DEVICE = "cuda"

# =========================
# Model
# =========================
MODEL_NAME = "baseline_cnn"

# =========================
# Output Paths
# =========================
CHECKPOINT_DIR = "checkpoints/"
LOG_DIR = "logs/"