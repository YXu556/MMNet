'''
Todo:
    *
'''
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
from torch.utils.data import Dataset


class SoMoDataset(Dataset):
    def __init__(self, data, mean, std, ids=None, scale=1, transform=None, cluster=None):
        super(SoMoDataset, self).__init__()

        self.X = data[:, :-1]
        self.y = data[:, -1]

        self.mean = mean
        self.std = std
        self.scale = scale

        self.transform = transform
        if ids is not None:
            self.ids = ids
        if cluster is not None:
            self.cluster = cluster

    def __len__(self):
        return self.X.shape[0]

    def update_transform(self, transform):
        self.transform = transform

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]

        X = (X - self.mean) / self.std
        y = y * self.scale

        sample = {
            'X': X,
            'y': y
        }

        if self.transform:
            sample = self.transform(sample)

        X = sample['X']
        y = sample['y']

        return X, y


class SoMoTSDataset(Dataset):
    def __init__(self, data_statics, data_dynamic, y, mean_s, std_s, mean_d, std_d, scale=1, transform=None):
        super(SoMoTSDataset, self).__init__()

        self.X_s = data_statics
        self.X_d = data_dynamic
        if data_statics.shape[1] == data_dynamic.shape[1]:
            self.X = np.concatenate([data_statics.reshape(*data_statics.shape, 1), data_dynamic], axis=2)
        else:
            self.X = np.array(list(zip(data_statics, data_dynamic)), dtype=object)
        self.y = y

        self.mean_s = mean_s
        self.std_s = std_s
        self.mean_d = mean_d.reshape(-1, 1)
        self.std_d = std_d.reshape(-1, 1)
        self.scale = scale

        self.transform = transform

    def __len__(self):
        return self.X_s.shape[0]

    def update_transform(self, transform):
        self.transform = transform

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        if X.shape[0] == 2:
            X_s, X_d = X
        else:
            X_s, X_d = X[:, 0], X[:, 1:]

        X_s = (X_s - self.mean_s) / self.std_s
        X_d = (X_d - self.mean_d) / self.std_d
        y = y * self.scale

        sample = {
            'X_s': X_s,
            'X_d': X_d.transpose(1, 0),
            'y': y
        }

        if self.transform:
            sample = self.transform(sample)

        X_s = sample['X_s']
        X_d = sample['X_d']
        y = sample['y']

        return X_s, X_d, y
