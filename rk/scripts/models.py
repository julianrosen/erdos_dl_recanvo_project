

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

import itertools as it

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

class two_layer_classifier_do(nn.Module):
    def __init__(self, num_labels, dr=.2): #num_labels is the number of labels appearing in the dataframe df we used Features(df), i.e. len(df.Label.unique()), dr=dropout rate
        super().__init__()
        self.dropout0= nn.Dropout(dr)
        self.norm = nn.BatchNorm1d(768) #including a batch normalization layer instead of StandardScaler() (Julian used StandardScaler() when playing with logistic regression) 
        self.layer1 = nn.Linear(768, 768)
        self.act1 = nn.ReLU()
        self.dropout1=nn.Dropout(dr)
        self.output = nn.Linear(768, num_labels)

    def forward(self, x):
        x = self.dropout0(x)
        x = self.norm(x)
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.output(x)
        return x
    
class three_layer_classifier_do(nn.Module):
    def __init__(self, num_labels, dr=.2): #num_labels is the number of labels appearing in the dataframe df we used Features(df), i.e. len(df.Label.unique()), dr=dropout rate
        super().__init__()
        self.dropout0= nn.Dropout(dr)
        self.norm = nn.BatchNorm1d(768) #including a batch normalization layer instead of StandardScaler() (Julian used StandardScaler() when playing with logistic regression) 
        self.layer1 = nn.Linear(768, 768)
        self.act1 = nn.ReLU()
        self.dropout1=nn.Dropout(dr)
        self.layer2 = nn.Linear(768, 768)
        self.act2 = nn.ReLU()
        self.dropout2=nn.Dropout(dr)
        self.output = nn.Linear(768, num_labels)

    def forward(self, x):
        x = self.dropout0(x)
        x = self.norm(x)
        x = self.act1(self.layer1(x))
        x = self.dropout1(x)
        x = self.act2(self.layer2(x))
        x = self.dropout2(x)
        x = self.output(x)
        return x