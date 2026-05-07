# Metadata Construction

The metadata CSV (`raga_20_dataset.csv`) was generated using the `mirdata` interface.

## Process

The notebook `notebooks/01_build_metadata.ipynb`:

* Loads dataset using `compiam` / `mirdata`
* Extracts:

  * track_id
  * artist
  * raga label
* Filters selected 20 ragas
* Rewrites audio paths
* Saves final CSV

---

## Environment

Originally executed in Google Colab.

To reproduce locally:

1. Create Python 3.11 environment
2. Install dependencies:

   ```
   pip install compiam mirdata pandas numpy
   ```
3. Set dataset path
4. Run notebook

---
