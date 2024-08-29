from typing import Any
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import torchaudio
import torch
from tqdm.notebook import tqdm

data_dir = Path(__file__).parent.parent.parent / "data"
file_list = data_dir / "tt_small_sessions.csv"
wav_dir = data_dir / "wav"


def get_hubert_features(participant: str) -> dict[str, Any]:
    """
    Extract HuBERT features for one participant

    Parameters
    ----------
    participant: str
        "P01", "P02", etc.

    Returns
    -------
    hubert_features: dict
        Dictionary with strings as keys. Values are
        X_tr: 2d torch.tensor with training features
        y_tr: 1d torch.tensor with training targets (labels)
        session_tr: 1d np.ndarray with training sessions
        X_te: 2d torch.tensor with test features
        X_te: 2d torch.tensor with test targets
        session_tr: 1d np.ndarray with test sessions
        label_list: list of labels
    """
    file_df = pd.read_csv(file_list)
    file_df_one_participant = file_df[file_df.Participant == participant]
    label_counts = file_df_one_participant.Label.value_counts()
    label_counts = label_counts[label_counts >= 30]
    label_list = label_counts.index.to_list()
    file_df_subset = file_df_one_participant[
        file_df_one_participant.Label.isin(label_list)
    ]

    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model()

    with torch.no_grad():
        t_list = []
        for filename in tqdm(file_df_subset.Filename):
            waveform, sample_rate = torchaudio.load(wav_dir / filename)
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, bundle.sample_rate
            )

            features, _ = model.extract_features(waveform, num_layers=1)
            t_list.append(features[0].mean((0, 1)))

    X = torch.stack(t_list)

    y = torch.zeros(len(file_df_subset), dtype=torch.long)
    for idx, label in enumerate(label_list):
        y[(file_df_subset.Label == label).values] = idx

    is_train = (file_df_subset.is_test == 0).values
    is_test = (file_df_subset.is_test == 1).values
    X_tr = X[is_train]
    y_tr = y[is_train]
    session_tr = file_df_subset[is_train].Session.values

    X_te = X[is_test]
    y_te = y[is_test]
    session_te = file_df_subset[is_test].Session.values

    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "session_tr": session_tr,
        "X_te": X_te,
        "y_te": y_te,
        "session_te": session_te,
        "label_list": label_list,
    }


def unweighted_f1(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Unweighted f1 score, rounded to 4 decimal places
    """
    return np.round(f1_score(actual, pred, average="macro"), 4)
