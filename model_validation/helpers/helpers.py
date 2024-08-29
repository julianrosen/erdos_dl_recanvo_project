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


def get_hubert_features(
    participant: str, label_list: list = None, subsample: bool = False
) -> dict[str, Any]:
    """
    Extract HuBERT features for one participant

    Parameters
    ----------
    participant: str
        "P01", "P02", etc.
    label_list: list, default None
        List of labels to use. If None, use all labels with 30+ samples
    subsample: bool, default False
        If True, subsample the data as in transfer learning paper


    Returns
    -------
    hubert_features: dict
        Dictionary with strings as keys. Values are
        X_tr: 2d torch.tensor with training features
        y_tr: 1d torch.tensor with training targets (labels)
        session_tr: 1d np.ndarray with training sessions
        waveforms_tr: list of resampled training waveforms
        X_te: 2d torch.tensor with test features
        X_te: 2d torch.tensor with test targets
        session_tr: 1d np.ndarray with test sessions
        waveforms_te: list of resampled test waveforms
        label_list: list of labels
    """
    file_df = pd.read_csv(file_list)
    file_df_one_participant = file_df[file_df.Participant == participant]
    label_counts = file_df_one_participant.Label.value_counts()
    label_counts = label_counts[label_counts >= 30]
    if label_list is None:
        label_list = label_counts.index.to_list()
    file_df_subset = file_df_one_participant[
        file_df_one_participant.Label.isin(label_list)
    ]

    if subsample:
        df_small = pd.DataFrame(
            file_df_subset.groupby(["Session", "Label"])
            .apply(lambda data: data.sample(min(10, len(data))))
            .values,
            columns=file_df_subset.columns,
        )
        vc = df_small.groupby("Label").size()
        file_df_subset = pd.DataFrame(
            df_small.groupby("Label")
            .apply(lambda data: data.sample(vc.min()))
            .values,
            columns=df_small.columns,
        )

    bundle = torchaudio.pipelines.HUBERT_BASE
    model = bundle.get_model()
    waveform_list = []
    with torch.no_grad():
        t_list = []
        for filename in tqdm(file_df_subset.Filename):
            waveform, sample_rate = torchaudio.load(wav_dir / filename)
            waveform = torchaudio.functional.resample(
                waveform, sample_rate, bundle.sample_rate
            )
            waveform_list.append(waveform)

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
    waveforms_tr = [
        waveform
        for waveform, is_test in zip(waveform_list, file_df_subset.is_test)
        if is_test == 0
    ]

    X_te = X[is_test]
    y_te = y[is_test]
    session_te = file_df_subset[is_test].Session.values
    waveforms_te = [
        waveform
        for waveform, is_test in zip(waveform_list, file_df_subset.is_test)
        if is_test == 1
    ]
    return {
        "X_tr": X_tr,
        "y_tr": y_tr,
        "session_tr": session_tr,
        "waveforms_tr": waveforms_tr,
        "X_te": X_te,
        "y_te": y_te,
        "session_te": session_te,
        "waveforms_te": waveforms_te,
        "label_list": label_list,
    }


def unweighted_f1(actual: np.ndarray, pred: np.ndarray) -> float:
    """
    Unweighted f1 score, rounded to 4 decimal places
    """
    return np.round(f1_score(actual, pred, average="macro"), 4)
