"""
feature_extraction.py — Main feature extraction pipeline

Reads the frozen 20-raga metadata CSV, loads each audio clip, extracts all
features at 10 ms resolution, saves one .npz per clip, and updates the
metadata CSV with the feature file path.

Run directly in VS Code: press F5.

Output .npz layout per clip
───────────────────────────
    pitch       (T,)    — F0 in Hz; 0 = unvoiced
    velocity    (T,)    — pitch derivative in Hz/s; 0 at unvoiced boundaries
    energy      (T,)    — RMS energy (linear)
    harmonic    (T, H)  — spectral energy at Sa & Pa harmonics (H = 10)
    tonic       ()      — scalar, estimated Sa frequency in Hz

All time-series features share the same T (trimmed to the shortest).

Dependencies
────────────
    pip install praat-parselmouth librosa soundfile pandas tqdm numpy
"""

import logging
import traceback
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm

from pitch    import extract_pitch
from energy   import extract_energy
from harmonic import extract_harmonic
from tonic    import extract_tonic

# ── Paths ─────────────────────────────────────────────────────────────────────

REPO_ROOT    = Path(r"C:/Users/Sragv/MIR Carnatic raga identification")
AUDIO_ROOT   = REPO_ROOT / "data" / "raw_audio"
METADATA_CSV = REPO_ROOT / "data" / "metadata" / "raga_20_dataset_frozen.csv"
FEATURES_DIR = REPO_ROOT / "features"          # one .npz per clip goes here
LOG_FILE     = REPO_ROOT / "feature_extraction.log"

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def align_to_length(arrays: dict[str, np.ndarray], target_T: int) -> dict[str, np.ndarray]:
    """
    Trim or zero-pad every 1-D array in `arrays` to `target_T` frames.
    2-D arrays (T, H) are trimmed/padded along axis 0.
    Scalars are left untouched.
    Logs a warning for every array that needed adjustment.
    """
    aligned = {}
    for name, arr in arrays.items():
        if np.ndim(arr) == 0:           # scalar
            aligned[name] = arr
            continue

        orig_T = arr.shape[0]
        if orig_T == target_T:
            aligned[name] = arr
        elif orig_T > target_T:
            log.warning("  [align] %s: trimming %d → %d frames", name, orig_T, target_T)
            aligned[name] = arr[:target_T]
        else:
            log.warning("  [align] %s: padding %d → %d frames", name, orig_T, target_T)
            pad_shape = (target_T - orig_T,) + arr.shape[1:]
            aligned[name] = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=0)
    return aligned


def clip_name_from_path(audio_path: str) -> str:
    """Derive a safe filename stem from the audio_path column."""
    return Path(audio_path).stem


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    # Validate inputs
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found:\n  {METADATA_CSV}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(METADATA_CSV)

    required_cols = {"relative_part"}
    missing_cols  = required_cols - set(meta.columns)
    if missing_cols:
        raise ValueError(f"Metadata CSV is missing columns: {missing_cols}")

    log.info("Loaded metadata: %d clips", len(meta))

    feature_paths  = []   # parallel list to build the new CSV column
    failed_clips   = []

    for idx, row in tqdm(meta.iterrows(), total=len(meta), unit="clip"):
        rel_audio = row["relative_part"]
        abs_audio = AUDIO_ROOT / rel_audio
        clip      = clip_name_from_path(rel_audio)
        npz_rel   = f"features/{clip}.npz"
        npz_abs   = REPO_ROOT / npz_rel

        # Skip if already extracted
        if npz_abs.exists():
            log.info("[SKIP] %s — .npz already exists", clip)
            feature_paths.append(npz_rel)
            continue

        if not abs_audio.exists():
            log.error("[MISSING] %s — audio file not found, skipping", abs_audio)
            feature_paths.append("")
            failed_clips.append(clip)
            continue

        try:
            # ── 1. Load audio ─────────────────────────────────────────────────
            audio, sr = librosa.load(str(abs_audio), sr=None, mono=True)

            # ── 2. Extract tonic (scalar, needed by harmonic) ─────────────────
            tonic_hz = extract_tonic(audio, sr)
            log.info("[%s] tonic = %.1f Hz", clip, tonic_hz)

            # ── 3. Extract time-series features ──────────────────────────────
            pitch_hz, velocity = extract_pitch(audio, sr)
            energy             = extract_energy(audio, sr)
            harmonic           = extract_harmonic(audio, sr, tonic_hz)

            # ── 4. Align all features to the same T ───────────────────────────
            raw = {
                "pitch":    pitch_hz,
                "velocity": velocity,
                "energy":   energy,
                "harmonic": harmonic,
                "tonic":    np.float32(tonic_hz),
            }
            time_series = {k: v for k, v in raw.items() if np.ndim(v) >= 1}
            lengths = {k: v.shape[0] for k, v in time_series.items()}
            target_T = min(lengths.values())

            if len(set(lengths.values())) > 1:
                log.warning("[%s] length mismatch before alignment: %s", clip, lengths)

            aligned = align_to_length(raw, target_T)

            # ── 5. Save .npz ──────────────────────────────────────────────────
            np.savez_compressed(
                str(npz_abs),
                pitch    = aligned["pitch"],
                velocity = aligned["velocity"],
                energy   = aligned["energy"],
                harmonic = aligned["harmonic"],
                tonic    = aligned["tonic"],
            )
            log.info("[OK] %s → %s  (T=%d)", clip, npz_rel, target_T)
            feature_paths.append(npz_rel)

        except Exception:
            log.error("[FAIL] %s\n%s", clip, traceback.format_exc())
            feature_paths.append("")
            failed_clips.append(clip)

    # ── 6. Update metadata CSV ────────────────────────────────────────────────
    meta["feature_path"] = feature_paths
    meta.to_csv(METADATA_CSV, index=False)
    log.info("Metadata CSV updated: %s", METADATA_CSV)

    # ── 7. Summary ────────────────────────────────────────────────────────────
    n_ok   = sum(1 for p in feature_paths if p)
    n_fail = len(failed_clips)
    log.info("─" * 60)
    log.info("Done.  %d succeeded,  %d failed.", n_ok, n_fail)
    if failed_clips:
        log.warning("Failed clips:\n  %s", "\n  ".join(failed_clips))


if __name__ == "__main__":
    main()
