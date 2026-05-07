## Stage 1: Audio → NPZ Feature Extraction (Priyanshu)

PRIYANSHU WRITE HERE

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

