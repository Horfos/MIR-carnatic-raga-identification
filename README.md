# Carnatic Raga Identification

## Overview

This project aims to identify Carnatic ragas from audio recordings using machine learning.

The current implementation provides a **baseline pipeline** that converts audio-derived features into fixed-length representations based on **tonic-normalized pitch distributions**.

---

## Dataset (Summary)

* Source: CompMusic Raga Dataset
* Selected ragas: 20
* Total tracks: 235
* Unique artists: 49
* Split: Artist-independent

> Full dataset details are available in `docs/dataset.md`.

---

## Feature Pipeline

The system operates in two stages:

### Stage 1: Audio → NPZ Features

Extracts time-series features such as pitch (F0), RMS energy, harmonics, and tonic.

### Stage 2: NPZ → Fixed-Length Features

Converts each clip into a **24-dimensional pitch distribution vector** using:

* tonic normalization
* octave wrapping
* histogram binning (24 bins)
* RMS-weighted aggregation

> Full pipeline details: `docs/feature_pipeline.md`

---

## Output

Processed features are saved as:

```text
data/processed/features.npz
```

Contents:

* `X`: feature matrix (N × 24)
* `ids`: clip identifiers

---

## Current Status

* Feature extraction pipeline: complete
* Dataset prepared: complete
* Baseline model training: in progress

---

## How to Run

```bash
python src/baseline_model/baseline_feature_extraction/pipeline.py
```

---

## Notes

* Audio files are not included due to permission restrictions
* Users must obtain the CompMusic dataset separately

---
