import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from src.features import extract_logmel, plot_logmel
import pandas as pd
import env_config

metadata = pd.read_csv("data/metadata/raga_20_dataset_frozen.csv")

# pick 3 samples
samples = metadata.sample(3)

for i, row in samples.iterrows():

    relative = row["relative_part"].strip().replace("/", os.sep)
    audio_path = os.path.join(env_config.AUDIO_ROOT, relative)
    print(f"Processing: {audio_path}") 

    logmel = extract_logmel(audio_path)

    plot_logmel(logmel)
    print(f"Log-mel shape: {logmel.shape}")