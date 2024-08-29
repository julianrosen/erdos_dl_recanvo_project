from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchaudio
from tqdm.notebook import tqdm

bundle = torchaudio.pipelines.HUBERT_BASE
HuBERT = bundle.get_model()

# List of data files
data_files = pd.read_csv("../data/directory_w_train_test.csv")
label_counts = data_files.Label.value_counts()
training_files = data_files.loc[
    data_files.Label.isin(label_counts[label_counts >= 30].index)
    & (data_files.is_test == 0)
].copy()
training_files["session"] = training_files.Filename.apply(
    lambda name: name.split("-")[0][:-3]
)

datadir = Path("../data/wav")
with torch.no_grad:
  for i in tqdm(range(len(data_files))):
    filename = data_files.Filename.iloc[i]
    waveform, sample_rate = torchaudio.load(datadir / filename)
    waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate) #resample the audio to be at the rate HuBERT expects (16KHz).
    features, _ = HuBERT.extract_features(waveform)
    t=features[0].mean((0, 1))
    torch.save(t, Path(f"../data/HuBERT_features/{filename.removesuffix('.wav')}.pt"))