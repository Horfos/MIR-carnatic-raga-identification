"""
save_processed.py
Utilities for persisting processed clip data to disk.
"""

from pathlib import Path

import numpy as np


def save_processed_clip(
    output_path: str,
    processed_data: dict,
) -> None:
    """
    Save a processed clip dictionary as a compressed .npz file.

    Parent directories are created automatically if they do not exist.

    Parameters
    ----------
    output_path : str
        Destination file path (e.g. 'data/processed/baseline_features/Abhayamba.npz').
    processed_data : dict
        Dictionary whose values are numpy-compatible scalars or arrays.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **processed_data)