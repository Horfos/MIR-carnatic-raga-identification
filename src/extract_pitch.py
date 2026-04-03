"""
Pitch Extractor — extracts F0 pitch every 10ms from audio files in a dataset folder.
Outputs a single combined pitch.csv.

Dependencies:
    pip install praat-parselmouth librosa soundfile pandas tqdm

Run directly in VS Code: just press F5 or click "Run Python File".
"""

from pathlib import Path

import pandas as pd
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm


# ─── Configuration ────────────────────────────────────────────────────────────

INPUT_DIR  = Path(r"C:\Users\priya\OneDrive\Documents\GitHub\MIR-carnatic-raga-identification\data\raw_audio\RagaDataset")
OUTPUT_CSV = Path(r"C:\Users\priya\OneDrive\Documents\GitHub\MIR-carnatic-raga-identification\data\metadata\pitch.csv")

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".aif"}

# Pitch analysis settings — tune these to your recordings
PITCH_FLOOR   = 75.0    # Hz — lower bound (75 for general speech, 50 for male bass)
PITCH_CEILING = 600.0   # Hz — upper bound (500–600 for female/child speech)
FRAME_STEP_MS = 10.0    # ms between pitch frames
SILENCE_VALUE = 0.0     # value written for unvoiced/silent frames (use float("nan") if preferred)


# ─── Core extraction ──────────────────────────────────────────────────────────

def extract_pitch_parselmouth(audio_path: Path) -> list[dict]:
    """
    Extract pitch values every FRAME_STEP_MS milliseconds using Praat (parselmouth).
    Returns a list of dicts: [{"time_s": ..., "pitch_hz": ...}, ...]
    Unvoiced frames are filled with SILENCE_VALUE.
    """
    snd = parselmouth.Sound(str(audio_path))

    pitch = call(
        snd,
        "To Pitch (ac)",
        FRAME_STEP_MS / 1000.0,   # time step in seconds
        PITCH_FLOOR,
        15,                        # max candidates
        True,                      # very accurate
        0.03,                      # silence threshold
        0.45,                      # voicing threshold
        0.01,                      # octave cost
        0.35,                      # octave jump cost
        0.14,                      # voiced/unvoiced cost
        PITCH_CEILING,
    )

    n_frames    = call(pitch, "Get number of frames")
    start_time  = call(pitch, "Get start time")
    time_step   = call(pitch, "Get time step")

    rows = []
    for i in range(1, n_frames + 1):
        t   = start_time + (i - 1) * time_step
        f0  = call(pitch, "Get value in frame", i, "Hertz")
        rows.append({
            "time_s":   round(t, 4),
            "pitch_hz": round(f0, 4) if f0 == f0 else SILENCE_VALUE,  # nan-check
        })
    return rows


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def discover_audio_files(input_dir: Path) -> list[Path]:
    """Recursively find all supported audio files under input_dir."""
    files = [
        p for p in input_dir.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files)


# ─── Main pipeline ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Validate input directory
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Audio dataset folder not found:\n  {INPUT_DIR}")

    # Create output directory if it doesn't exist yet
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    # Discover audio files
    audio_files = [
        p for p in INPUT_DIR.rglob("*")
        if p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    audio_files.sort()

    if not audio_files:
        print(f"No supported audio files found under:\n  {INPUT_DIR}")
    else:
        print(f"Found {len(audio_files)} audio file(s).")
        print(f"Extracting pitch at {FRAME_STEP_MS}ms intervals...\n")

        all_rows: list[dict] = []

        for audio_path in tqdm(audio_files, unit="file"):
            try:
                rows = extract_pitch_parselmouth(audio_path)
                for r in rows:
                    r["filename"] = audio_path.name
                    r["raga"]     = audio_path.parent.name   # folder name = raga label
                all_rows.extend(rows)
            except Exception as exc:
                print(f"\n  [WARN] Skipping {audio_path.name}: {exc}")

        if all_rows:
            df = pd.DataFrame(all_rows, columns=["filename", "raga", "time_s", "pitch_hz"])
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\nDone! {len(df):,} rows written to:\n  {OUTPUT_CSV}")