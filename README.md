# Carnatic Raga Identification 

## Dataset (Frozen v1)
### Dataset Overview
 This project uses the CompMusic Raga Dataset.  
 The audio files are permission-restricted and are **not included** in this repository.  

 - Tradition: Carnatic  
 - Number of slected ragas: 20  
 - Total Tracks: 235  
 - Total Unique artists: 49  

### Selected Ragas (20)

- Tōḍi  
- Kalyāṇi  
- Śankarābharaṇaṁ  
- Karaharapriya  
- Harikāmbhōji  
- Māyāmāḷavagauḷa  
- Kāmavardani  
- Bhairavi  
- Ānandabhairavi  
- Sahānā  
- Bēgaḍa  
- Kāṁbhōji  
- Rītigauḷa  
- Mōhanaṁ  
- Hamsadhwani  
- Hindōḷaṁ  
- Madhyamāvati  
- Nāṭakurinji  
- Kāpi  
- Dhanyāsi  

### Dataset Split Protocol

The dataset was partitioned using an artist-independent splitting strategy to prevent timbral and stylistic leakage between training and evaluation sets. Artists were assigned exclusively to a single split, ensuring that no recordings by the same artist appeared across multiple splits.  

The split was performed at the artist level with raga coverage ensured during assignment. Due to unequal numbers of recordings per artist, exact proportional splits by track count were not enforced. Instead, remaining artists were assigned based on cumulative track counts to achieve approximately balanced training, validation, and test sets.  

- Split type: Artist-independent  
- Splits: Training / Validation / Test  
- Track distribution: approximately balanced (82 / 76 / 77 tracks)  
- Artist overlap across splits: none  

### Dataset Freeze

Version: v1  
Freeze Date: <2026-02-28>

The file `data/metadata/raga_20_dataset_frozen.csv` is considered the authoritative dataset for all subsequent experiments.

No modifications will be made to:
- Track selection
- Labels
- Split assignments

### Audio Files

Audio files are not included in this repository due to permission restrictions.

Users must obtain the CompMusic Raga Dataset separately and configure local paths accordingly.

To reproduce experiments:

1. Obtain access to the CompMusic Raga audio dataset.
2. Download the dataset using `mirdata` or from the official Zenodo source.
3. Place the audio files in the appropriate local directory.

## Metadata Construction

The metadata CSV (`raga_20_dataset.csv`) was generated using the `mirdata` interface for the `compmusic_raga` dataset.

The notebook `notebooks/01_build_metadata.ipynb`:

- Loads the dataset using `compiam` / `mirdata`
- Extracts track_id, artist, and raga labels
- Filters a selected set of 20 ragas
- Rewrites audio paths to point to the local permissioned audio directory
- Exports the final CSV to `data/metadata/`

## Environment

The metadata notebook was executed in Google Colab.

To reproduce locally:

1. Create a Python 3.11 environment
2. Install dependencies:
   pip install compiam mirdata pandas numpy

3. Set `DATA_HOME` to the location of the CompMusic dataset.
4. Run `01_build_metadata.ipynb`

