import functools
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import (
    cross_val_predict,
    StratifiedKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler, LabelEncoder
)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

import torchaudio
from tqdm.notebook import tqdm


def to_prob(metric):
    @functools.wraps(metric)
    def metric_that_takes_prob(y_actual, y_pred, sample_weight=None):
        return metric(y_actual, y_pred.argmax(1), sample_weight=sample_weight)

    return metric_that_takes_prob


metrics = {
    "accuracy": to_prob(accuracy_score),
    "balanced_accuracy": to_prob(balanced_accuracy_score),
    "unweighted_f1": to_prob(functools.partial(f1_score, average="macro")),
    "UAR": to_prob(functools.partial(recall_score, average="macro")),
    "logloss": log_loss,
}

data_files = pd.read_csv("../data/tt_small_sessions.csv")
training_files = data_files.loc[data_files.is_test == 0].copy()
label_counts=training_files[training_files.Participant == 'P05'].Label.value_counts()
P05_t=training_files.loc[(training_files.Participant =='P05') & (training_files.Label.isin(label_counts[label_counts >= 30].index))]

class UncomputedFeatures(Dataset):
    def __init__(self, df, labels=[], root_dir=Path("../data/wav")): #To construct a Features object, you feed in a dataframe with column "Filename" with the names of the audio files.
        self.df=df
        self.root_dir=root_dir
        if labels==[]:
            labels=self.df.Label.unique() # if no list of labels is provided, take the labels appearing in df
        self.labeldict=dict(zip(labels, range(len(labels)))) #keeps track of the labels and zips them with numerical values 0,1,...


    def __getitem__(self, idx): #idx an index of dataframe
        if torch.is_tensor(idx):
            idx = idx.tolist()        
        filename = self.df.Filename.iloc[idx]
        waveform, sample_rate = torchaudio.load(self.root_dir / filename)
        waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate) #resample the audio to be at the rate HuBERT expects (16KHz).
        features, _ = HuBERT.extract_features(waveform)
        X=features[0].mean((0, 1))
        y=torch.zeros(len(self.labeldict)).detach()
        y[self.labeldict[self.df.Label.iloc[idx]]]=1
        z = self.labeldict[self.df.Label.iloc[idx]] #z is the integer corresponding to our label. We return z for use in train-test splits through sklearn - I had problems making this happy with tensor objects in the stratify variable
        return X.detach(), y , z

    def __len__(self):
        return len(self.df)
    
class PrecomputedFeatures(Dataset):
    def __init__(self, df, labels=[], root_dir=Path("../data/wav")): #To construct a Features object, you feed in a dataframe with column "Filename" with the names of the audio files, as well as a list of expected labels.
        self.df=df
        self.root_dir=root_dir
        if labels==[]:
            labels=self.df.Label.unique() # if no list of labels is provided, take the labels appearing in df
        self.labeldict=dict(zip(labels, range(len(labels)))) #keeps track of the labels and zips them with numerical values 0,1,...


    def __getitem__(self, idx): #idx is an index of our dataframe
        if torch.is_tensor(idx):
            idx = idx.tolist()
        filename = self.df.Filename.iloc[idx]
        X=torch.load('../data/HuBERt_features/'+filename.removesuffix('.wav')+'.pt').detach()
        y=torch.zeros(len(self.labeldict)).detach()
        y[self.labeldict[self.df.Label.iloc[idx]]]=1
        z = self.labeldict[self.df.Label.iloc[idx]]
        return X, y , z #z is the integer corresponding to our label. We return z for use in train-test splits through sklearn - I had problems making this happy with tensor objects in the stratify variable
    def __len__(self):
        return len(self.df)