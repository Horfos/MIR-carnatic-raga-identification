import numpy as np


REQUIRED_KEYS = {"pitch", "rms", "tonic" , "velocity" , "harmonic"}
FILE_NAMES= {"pitch", "energy", "tonic", "velocity", "harmonic"}

def load_clip(npz_path: str) -> dict:
    """
    Parameters
    npz_path : str
        Path to the .npz file.

    Returns
    dict with keys:
        "f0"    : np.ndarray, shape (T,)
        "rms"   : np.ndarray, shape (T,)
        "tonic" : float
    """
    data = np.load(npz_path, allow_pickle=False)

    missing = FILE_NAMES - set(data.files)
    if missing:
        raise KeyError(
            f"Missing required keys {missing} in file: {npz_path}"
        )

    f0 = data["pitch"].astype(np.float64)
    rms = data["energy"].astype(np.float64)
    tonic = float(data["tonic"])
    velocity = data["velocity"].astype(np.float64)
    harmonics = data["harmonic"].astype(np.float64)

    if f0.ndim != 1:
        raise ValueError(
            f"Expected 'pitch' to be 1-D, got shape {f0.shape} in: {npz_path}"
        )
    if rms.ndim != 1:
        raise ValueError(
            f"Expected 'rms' to be 1-D, got shape {rms.shape} in: {npz_path}"
        )
    if f0.shape[0] != rms.shape[0]:
        raise ValueError(
            f"Length mismatch: pitch={f0.shape[0]}, rms={rms.shape[0]} in: {npz_path}"
        )
    if velocity.ndim != 1:
        raise ValueError(
            f"Expected 'velocity' to be 1-D, got shape {velocity.shape} in: {npz_path}"
        )
    if harmonics.ndim != 2:
        raise ValueError(
            f"Expected 'harmonics' to be 2-D, got shape {harmonics.shape} in: {npz_path}"
        )
    return {"f0": f0, "rms": rms, "tonic": tonic, "velocity": velocity, "harmonics": harmonics}

