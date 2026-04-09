"""
Pitch Extractor — extracts F0 pitch every 10ms from audio files listed in a metadata CSV.
Outputs a single combined pitch.csv.

Dependencies:
    pip install praat-parselmouth pandas tqdm

Run directly in VS Code: just press F5 or click "Run Python File".
"""

from pathlib import Path

import pandas as pd
import parselmouth
from parselmouth.praat import call
from tqdm import tqdm


# ─── Configuration ────────────────────────────────────────────────────────────

# Root of your git repository — all relative audio_path values are resolved from here
REPO_ROOT = Path(r"C:\Users\priya\OneDrive\Documents\GitHub\MIR-carnatic-raga-identification")

# CSV that lists the audio files to process (must have an "audio_path" column)
METADATA_CSV = REPO_ROOT / "data" / "metadata" / "raga_20_dataset_frozen.csv"

# Where the output pitch CSV will be written
OUTPUT_CSV = REPO_ROOT / "data" / "metadata" / "pitch.csv"

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


# ─── Main pipeline ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Validate metadata CSV
    if not METADATA_CSV.exists():
        raise FileNotFoundError(f"Metadata CSV not found:\n  {METADATA_CSV}")

    # Load metadata and resolve audio paths relative to repo root
    meta = pd.read_csv(METADATA_CSV)
    if "audio_path" not in meta.columns:
        raise ValueError(f"'audio_path' column not found in {METADATA_CSV.name}. "
                         f"Available columns: {list(meta.columns)}")

    # Build list of (absolute_audio_path, row_metadata) tuples
    entries = []
    for _, row in meta.iterrows():
        abs_path = REPO_ROOT / row["audio_path"]
        entries.append((abs_path, row))

    # Warn about any missing files upfront
    missing = [p for p, _ in entries if not p.exists()]
    if missing:
        print(f"[WARN] {len(missing)} file(s) listed in the CSV were not found on disk:")
        for p in missing[:10]:   # show at most 10 to avoid flooding the console
            print(f"  {p}")
        if len(missing) > 10:
            print(f"  ... and {len(missing) - 10} more.")

    runnable = [(p, r) for p, r in entries if p.exists()]
    if not runnable:
        print("No audio files to process. Exiting.")
    else:
        print(f"\nProcessing {len(runnable)} of {len(entries)} file(s) "
              f"({len(missing)} skipped — not found).")
        print(f"Extracting pitch at {FRAME_STEP_MS}ms intervals...\n")

        # Create output directory if it doesn't exist yet
        OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

        all_rows: list[dict] = []

        for audio_path, meta_row in tqdm(runnable, unit="file"):
            try:
                rows = extract_pitch_parselmouth(audio_path)

                # Carry useful columns from the metadata CSV into every pitch row
                for r in rows:
                    r["track_id"] = meta_row.get("track_id", "")
                    r["filename"] = audio_path.name
                    r["raga"]     = meta_row.get("raga", "")
                    r["split"]    = meta_row.get("split", "")

                all_rows.extend(rows)
            except Exception as exc:
                print(f"\n  [WARN] Skipping {audio_path.name}: {exc}")

        if all_rows:
            cols = ["track_id", "filename", "raga", "split", "time_s", "pitch_hz"]
            df = pd.DataFrame(all_rows, columns=cols)
            df.to_csv(OUTPUT_CSV, index=False)
            print(f"\nDone! {len(df):,} rows written to:\n  {OUTPUT_CSV}")