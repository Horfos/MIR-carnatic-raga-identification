## Stage 1: Audio → NPZ Feature Extraction (Priyanshu)

### Overview

Features are extracted from each audio clip at a consistent *10 ms frame resolution* and saved as one compressed .npz file per clip under features/. The metadata CSV is updated in-place with a feature_path column pointing to each clip's .npz.

### Feature Set

Each .npz file contains the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| pitch | (T,) | F0 in Hz per frame; 0.0 for unvoiced frames |
| velocity | (T,) | Pitch derivative in Hz/s; 0.0 at unvoiced boundaries |
| energy | (T,) | RMS energy per frame (linear scale) |
| harmonic | (T, 10) | Spectral energy at Sa harmonics (cols 0–4) and Pa harmonics (cols 5–9) |
| tonic | scalar | Estimated Sa frequency in Hz for the clip |

All time-series features share the same length T (trimmed to the shortest across all features for a given clip).

### Extraction Methods

*Pitch (F0)* — Extracted using Praat's autocorrelation method via praat-parselmouth. Praat slides a copy of the signal over itself every 10 ms and finds the delay at which the signal best matches itself; the inverse of that delay gives the fundamental frequency. Unvoiced and silent frames are returned as 0.0.

*Pitch Velocity* — Computed directly from the pitch array as a frame-to-frame finite difference scaled to Hz/s: (pitch[i] − pitch[i−1]) / 0.01. Set to 0.0 wherever either the current or previous frame is unvoiced, preventing spurious spikes at voiced/unvoiced boundaries.

*Energy* — RMS energy computed by librosa over a 25 ms window centred on each 10 ms frame: sqrt(mean(samples²)). Captures loudness dynamics across the performance.

*Harmonic* — Short-Time Fourier Transform (STFT) computed by librosa at 10 ms hops. The magnitude spectrum is sampled at the harmonic series of Sa (tonic × 1, 2, 3, 4, 5) and Pa (tonic × 1.5, 3.0, 4.5, 6.0, 7.5), giving 10 amplitude values per frame. Tonic must be estimated before this step.

*Tonic* — Estimated per clip in three stages: (1) pYIN F0 tracking collects all voiced F0 values across the clip; (2) all values are folded into a single octave using cent-domain modulo arithmetic and a 120-bin histogram identifies the most consistently sung pitch class (Sa); (3) the winning bin is shifted to the octave nearest to the median voiced F0, yielding a concrete Hz value. Returned as a scalar.


## Stage 2: NPZ → Processed Baseline Features

This stage converts per-clip `.npz` feature files into structured processed representations suitable for machine learning experiments.

Each clip is processed independently and saved as a corresponding processed `.npz` file.

### Processing Steps

1. **Voiced Frame Selection**

   * Frames where pitch ≤ 0 are treated as unvoiced
   * A boolean voiced mask is generated for downstream processing

2. **Tonic Normalization**

   * Pitch values are converted from Hz to cents relative to tonic:

     ```
     cents = 1200 × log2(f0 / tonic)
     ```

   * This produces a tonic-invariant pitch representation

3. **Octave Wrapping**

   * Pitch values are mapped into a single octave using modulo 1200:

     ```
     cents mod 1200
     ```

4. **Pitch Histogram Construction**

   * The octave range `[0, 1200)` is divided into 24 bins
   * A weighted pitch histogram is computed using RMS energy as frame weights
   * The histogram is normalized to sum to 1

5. **Statistical Feature Extraction**

   Additional summary statistics are computed from the processed signals:

   ### Pitch Statistics

   * Mean pitch
   * Standard deviation
   * Median pitch
   * Pitch range

   ### Velocity Statistics

   * Mean absolute velocity
   * Velocity standard deviation
   * Maximum absolute velocity

   ### RMS Statistics

   * Mean RMS energy
   * RMS standard deviation
   * Maximum RMS energy

   ### Harmonic Statistics

   * Mean harmonic energy
   * Harmonic energy standard deviation

---

### Processed Output

Each processed clip `.npz` file contains:

#### Raw Signals

* `f0`
* `rms`
* `velocity`
* `harmonics`
* `tonic`

#### Processed Signals

* `f0_cents`
* `voiced_mask`
* `pitch_histogram_24`

#### Summary Features

* `pitch_stats`
* `velocity_stats`
* `rms_stats`
* `harmonic_stats`

---

### Final Baseline Feature Vector

For baseline model training, the following features are concatenated:

* 24-dimensional pitch histogram
* 4 pitch statistics
* 3 velocity statistics
* 3 RMS statistics
* 2 harmonic statistics

Final baseline feature dimensionality:

```
36 dimensions
```

---

### Notes

* Temporal ordering is not preserved in the baseline representation
* Sequence-level melodic transitions are not modeled
* Gamaka structure is only indirectly represented through velocity statistics
* The processed per-clip representation enables future feature additions without recomputing raw audio features

# Baseline Results

Final feature vector dimension: 36

## Models Evaluated

| Model               | Test Accuracy |
| ------------------- | ------------- |
| Logistic Regression | 60.0%         |
| CNN Baseline        | 72.5%         |
| Random Forest       | 75.0%         |

## Observations

* The baseline performs strongly despite using only global pitch-distribution-based features.
* Confusions are concentrated among musically related ragas.
* Allied ragas such as Harikāmbhōji, Kāṁbhōji, and Karaharapriya remain difficult to separate.
* Current features do not explicitly model melodic movement, phrase structure, or gamakas.

## Next Direction

Planned next steps include:

* transition/movement-based melodic features
* arohana-avarohana-inspired representations
* temporal modeling
* segment-wise analysis using shorter audio excerpts
