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

from scripts.featureDataSets import *


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

label_metrics=list(metrics.keys())
label_metrics.remove('logloss')


data_files = pd.read_csv("../data/tt_small_sessions.csv")
training_files = data_files.loc[data_files.is_test == 0].copy()
label_counts=training_files[training_files.Participant == 'P05'].Label.value_counts()
P05_t=training_files.loc[(training_files.Participant =='P05') & (training_files.Label.isin(label_counts[label_counts >= 30].index))]

def get_all(dataset): #assuming dataset outputs dataset[i] outputs X[i],y[i] where X[i] is a length 768 dim 1 tensor and y[i] is a length num_labels dim 1 tensor
    if type(dataset)==torch.utils.data.dataset.Subset: #need to do this, since dataset.Subset does not inherit the properties of a custom dataset.
        labeldict=dataset.dataset.labeldict
    else:
        labeldict=dataset.labeldict
    X=torch.zeros(len(dataset), len(dataset[0][0]))
    y=torch.zeros(len(dataset), len(labeldict) )
    z=[0]*len(dataset)
    for i in range(len(dataset)):
        X[i], y[i], z[i] = dataset[i]
    return X, y, z

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight.data) #initializes using xavier uniform distribution, unif on [-M,M] where M=6/\sqrt(num_inputs+num_outputs). seems to be standard choice.

# define a cross validation function
def crossvalid(model, #must include model,
               loss=nn.CrossEntropyLoss(), #if using default, model must have outputs a linear layer. If using a custom model, call it custom_loss
               df=P05_t, #if using the default df must have P05_t already loaded in.
               k_fold=5, batch_size=25, n_epochs=100, lr=.01, m=.8, wd=0, random_state=691): 
    dataset=PrecomputedFeatures(df)
    output=dict()
    for metric in metrics.keys(): #populating output with dataframes keeping track of metrics on each epoch, per split
        df=pd.DataFrame()
        df.index.name='Epoch'
        for i in range(k_fold):
            df[f'split_{i}_train']=[0]*n_epochs
            df[f'split_{i}_val']=[0]*n_epochs
        output[metric]=df

    optimizer=optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=m) # change this if you like!
    
    kfold = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_state)
    all_features,all_labels, z =get_all(dataset)
    all_features, all_labels = list(all_features), list(all_labels)
    for i, (train_indices, val_indices) in enumerate(kfold.split(all_features,z)):

        model.apply(weights_init) #reset the weights of the model

        print(f"{i}-th Fold:")
        train_set=torch.utils.data.dataset.Subset(dataset,train_indices) #correct dataset for this i-th split of train-indices
        val_set=torch.utils.data.dataset.Subset(dataset,val_indices) #correct dataset for this i-th split of val-indices
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)
        """Now we train:"""
        for epoch in tqdm(range(n_epochs)):
            model.train()
            for X_batch, y_batch, z_batch in train_loader:
                y_pred = model(X_batch)
                train_loss = loss(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            X_train, y_train, z_train=get_all(train_set)
            X_val, y_val, z_val=get_all(val_set)

            model.eval()  #put in evaluation mode to determine predicted values on train and validation sets
            y_train_pred=model(X_train).detach()
            y_val_pred=model(X_val).detach()
            for metric in label_metrics:
                train_metric=metrics[metric](z_train, y_train_pred) #for label_metrics, score needs the z, i.e the actual label. Note the wrapper around f1 score that finds the argmax of y_pred.
                val_metric= metrics[metric](z_val, y_val_pred)
                output[metric].loc[epoch, f'split_{i}_train']=train_metric
                output[metric].loc[epoch, f'split_{i}_val']=val_metric
            train_logloss= metrics['logloss'](y_train, nn.Softmax(dim=1)(y_train_pred)) #need to take softmax of the predictions to get probabilities
            val_logloss= metrics['logloss'](y_val, nn.Softmax(dim=1)(y_val_pred))
            output['logloss'].loc[epoch, f'split_{i}_train']=train_logloss
            output['logloss'].loc[epoch, f'split_{i}_val']=val_logloss
            #print(f"Epoch {epoch}:", f"Train logloss = {train_logloss};", f" Val logloss = {val_logloss}.")
            #print(f"Train F1 = {train_f1}", f"Val F1 = {val_f1}.")
    
    return output
        
def plot_scores(output, k_fold=5):
    for metric in metrics.keys():
        fig, axes= plt.subplots(ncols=5)
        fig.set_figwidth(5*k_fold)
        fig.set_figheight(5)
        fig.suptitle(("Training: " + metric))

        for i in range(k_fold):
            axes[i].plot(output[metric][f'split_{i}_train'], color='r', label="train")
            axes[i].plot(output[metric][f'split_{i}_val'], color='b', label="val")
            axes[i].legend
            axes[i].set_title(f"Split {i}")
        plt.show()

# var of crossvalid with no loading bar
def crossvalid_nobar(model, #must include model,
               loss=nn.CrossEntropyLoss(), #if using default, model must have outputs a linear layer. If using a custom model, call it custom_loss
               df=P05_t, #if using the default df must have P05_t already loaded in.
               k_fold=5, batch_size=25, n_epochs=100, lr=.01, m=.8, wd=0, random_state=691): 
    dataset=PrecomputedFeatures(df)
    output=dict()
    for metric in metrics.keys(): #populating output with dataframes keeping track of metrics on each epoch, per split
        df=pd.DataFrame()
        df.index.name='Epoch'
        for i in range(k_fold):
            df[f'split_{i}_train']=[0]*n_epochs
            df[f'split_{i}_val']=[0]*n_epochs
        output[metric]=df

    optimizer=optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=m) # change this if you like!
    
    kfold = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_state)
    all_features,all_labels, z =get_all(dataset)
    all_features, all_labels = list(all_features), list(all_labels)
    for i, (train_indices, val_indices) in enumerate(kfold.split(all_features,z)):

        model.apply(weights_init) #reset the weights of the model

        # print(f"{i}-th Fold:")
        train_set=torch.utils.data.dataset.Subset(dataset,train_indices) #correct dataset for this i-th split of train-indices
        val_set=torch.utils.data.dataset.Subset(dataset,val_indices) #correct dataset for this i-th split of val-indices
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)
        """Now we train:"""
        for epoch in range(n_epochs): #removes loading bar
            model.train()
            for X_batch, y_batch, z_batch in train_loader:
                y_pred = model(X_batch)
                train_loss = loss(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            X_train, y_train, z_train=get_all(train_set)
            X_val, y_val, z_val=get_all(val_set)

            model.eval()  #put in evaluation mode to determine predicted values on train and validation sets
            y_train_pred=model(X_train).detach()
            y_val_pred=model(X_val).detach()
            for metric in label_metrics:
                train_metric=metrics[metric](z_train, y_train_pred) #for label_metrics, score needs the z, i.e the actual label. Note the wrapper around f1 score that finds the argmax of y_pred.
                val_metric= metrics[metric](z_val, y_val_pred)
                output[metric].loc[epoch, f'split_{i}_train']=train_metric
                output[metric].loc[epoch, f'split_{i}_val']=val_metric
            train_logloss= metrics['logloss'](y_train, nn.Softmax(dim=1)(y_train_pred)) #need to take softmax of the predictions to get probabilities
            val_logloss= metrics['logloss'](y_val, nn.Softmax(dim=1)(y_val_pred))
            output['logloss'].loc[epoch, f'split_{i}_train']=train_logloss
            output['logloss'].loc[epoch, f'split_{i}_val']=val_logloss
            #print(f"Epoch {epoch}:", f"Train logloss = {train_logloss};", f" Val logloss = {val_logloss}.")
            #print(f"Train F1 = {train_f1}", f"Val F1 = {val_f1}.")
    
    return output

# var of crossvalid with no loading bar, opt=AdamW
def crossvalid_AdamW_nobar(model, #must include model,
               loss=nn.CrossEntropyLoss(), #if using default, model must have outputs a linear layer. If using a custom model, call it custom_loss
               df=P05_t, #if using the default df must have P05_t already loaded in.
               k_fold=5, batch_size=25, n_epochs=100, lr=.01, wd=0, random_state=691): 
    dataset=PrecomputedFeatures(df)
    output=dict()
    for metric in metrics.keys(): #populating output with dataframes keeping track of metrics on each epoch, per split
        df=pd.DataFrame()
        df.index.name='Epoch'
        for i in range(k_fold):
            df[f'split_{i}_train']=[0]*n_epochs
            df[f'split_{i}_val']=[0]*n_epochs
        output[metric]=df

    optimizer=optim.AdamW(model.parameters(), lr=lr, weight_decay=wd) # change this if you like!
    
    kfold = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=random_state)
    all_features,all_labels, z =get_all(dataset)
    all_features, all_labels = list(all_features), list(all_labels)
    for i, (train_indices, val_indices) in enumerate(kfold.split(all_features,z)):

        model.apply(weights_init) #reset the weights of the model

        # print(f"{i}-th Fold:")
        train_set=torch.utils.data.dataset.Subset(dataset,train_indices) #correct dataset for this i-th split of train-indices
        val_set=torch.utils.data.dataset.Subset(dataset,val_indices) #correct dataset for this i-th split of val-indices
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)
        """Now we train:"""
        for epoch in range(n_epochs): #removes loading bar
            model.train()
            for X_batch, y_batch, z_batch in train_loader:
                y_pred = model(X_batch)
                train_loss = loss(y_pred, y_batch)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
            X_train, y_train, z_train=get_all(train_set)
            X_val, y_val, z_val=get_all(val_set)

            model.eval()  #put in evaluation mode to determine predicted values on train and validation sets
            y_train_pred=model(X_train).detach()
            y_val_pred=model(X_val).detach()
            for metric in label_metrics:
                train_metric=metrics[metric](z_train, y_train_pred) #for label_metrics, score needs the z, i.e the actual label. Note the wrapper around f1 score that finds the argmax of y_pred.
                val_metric= metrics[metric](z_val, y_val_pred)
                output[metric].loc[epoch, f'split_{i}_train']=train_metric
                output[metric].loc[epoch, f'split_{i}_val']=val_metric
            train_logloss= metrics['logloss'](y_train, nn.Softmax(dim=1)(y_train_pred)) #need to take softmax of the predictions to get probabilities
            val_logloss= metrics['logloss'](y_val, nn.Softmax(dim=1)(y_val_pred))
            output['logloss'].loc[epoch, f'split_{i}_train']=train_logloss
            output['logloss'].loc[epoch, f'split_{i}_val']=val_logloss
            #print(f"Epoch {epoch}:", f"Train logloss = {train_logloss};", f" Val logloss = {val_logloss}.")
            #print(f"Train F1 = {train_f1}", f"Val F1 = {val_f1}.")
    
    return output