# Dataset Description

## Overview

This project uses the CompMusic Raga Dataset.

* Tradition: Carnatic
* Selected ragas: 20
* Total tracks: 235
* Unique artists: 49

---

## Selected Ragas

* Tōḍi
* Kalyāṇi
* Śankarābharaṇaṁ
* Karaharapriya
* Harikāmbhōji
* Māyāmāḷavagauḷa
* Kāmavardani
* Bhairavi
* Ānandabhairavi
* Sahānā
* Bēgaḍa
* Kāṁbhōji
* Rītigauḷa
* Mōhanaṁ
* Hamsadhwani
* Hindōḷaṁ
* Madhyamāvati
* Nāṭakurinji
* Kāpi
* Dhanyāsi

---

## Dataset Split Protocol

The dataset is split using an **artist-independent strategy**:

* No artist appears in multiple splits
* Ensures no timbral leakage
* Maintains raga coverage across splits

### Split Statistics

* Train: 80 tracks
* Validation: 77 tracks
* Test: 78 tracks

---

## Dataset Freeze

Version: v1
Freeze Date: 2026-02-28

The file:

```text
data/metadata/raga_20_dataset_frozen.csv
```

is the authoritative dataset definition.

No changes are made to:

* Track selection
* Labels
* Splits

---

## Audio Access

Audio files are not included due to permission restrictions.

To reproduce:

1. Obtain the CompMusic dataset
2. Use `mirdata` or official sources
3. Configure local paths accordingly
