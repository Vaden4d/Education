import pandas as pd
import numpy as np
import os

from utils import download_data, labels

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt

class Data(Dataset):

    def __init__(self, batch_size=5):
        '''
        Arguments:
            Path to train data folder (string): Folder with cell images
        '''

        self.batch_size = batch_size
        self.data_path = ''
        self.metadata_path = ''
        self._metadata = ''
        self.n = 0

    @property
    def metadata(self):
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value
        self.n = len(value)

    def download_metadata(self, path_to_data):
        self.metadata_path = path_to_data
        self.metadata = pd.read_csv(self.metadata_path)

    def __len__(self):
        return self.n

    def __iter__(self):
        n_batches = np.ceil(self.metadata.shape[0] / self.batch_size).astype(int)
        for i in range(n_batches):
            names = self.metadata.iloc[i*(self.batch_size): (i+1)*self.batch_size].Id
            tensor = torch.from_numpy(download_data(names)).permute(1, 0, 2, 3).float()
            yield tensor

class LabeledData(Data):

    def __init__(self):
        Data.__init__(self)

    def download_metadata(self, path_to_data='/home/vaden4d/Documents/kaggles/proteins/train.csv'):
        self.metadata_path = path_to_data
        self.metadata = pd.read_csv(self.metadata_path)
        self.metadata = pd.concat(
            [self.metadata,
            pd.get_dummies(self.metadata.Target.apply(lambda x: x.split()).apply(pd.Series).stack()).sum(level=0).sort_index(axis=1)
            ], axis=1)
        self.metadata = self.metadata.drop(columns='Target')
        self.metadata = self.metadata.rename(columns=labels)

    def __iter__(self):
        n_batches = np.ceil(self.metadata.shape[0] / self.batch_size).astype(int)
        for i in range(n_batches):

            labels = self.metadata.iloc[i*(self.batch_size): (i+1)*self.batch_size].iloc[:, 1:].values
            labels = torch.from_numpy(labels).int()

            names = self.metadata.iloc[i*(self.batch_size): (i+1)*self.batch_size].Id
            tensor = torch.from_numpy(download_data(names)).permute(0, 3, 1, 2).float()

            yield tensor, labels

class Splitter():

    def __init__(self, obj):

        self.obj = obj
        self.train_size = 1
        self.test_size = 0
        self.valid_size = 0

    def train_test_split(self, test_size=0.1, random_state=42):
        self.test_size = test_size
        self.train_size = 1 - test_size

        self.train_max_index = int(self.train_size * self.obj.n)

        metadata = self.obj.metadata.copy()
        metadata = metadata.sample(frac=1, random_state=random_state)

        train_metadata = metadata.iloc[:self.train_max_index]
        test_metadata = metadata.iloc[self.train_max_index:]

        train_obj = LabeledData()
        test_obj = LabeledData()

        train_obj.metadata = train_metadata
        test_obj.metadata = test_metadata
        return train_obj, test_obj


    def train_validation_test_split(self, valid_size=0.2, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.valid_size = valid_size
        self.train_size = 1 - test_size - valid_size

        self.train_max_index = int(self.train_size * self.obj.n)
        self.test_max_index = int((self.train_size + self.test_size) * self.obj.n)

        metadata = self.obj.metadata.copy()
        metadata = metadata.sample(frac=1, random_state=random_state)

        train_metadata = metadata.iloc[:self.train_max_index]
        test_metadata = metadata.iloc[self.train_max_index:self.test_max_index]
        valid_metadata = metadata.iloc[self.test_max_index:]

        train_obj = LabeledData()
        test_obj = LabeledData()
        valid_obj = LabeledData()

        train_obj.metadata = train_metadata
        test_obj.metadata = test_metadata
        valid_obj.metadata = valid_metadata

        return train_obj, test_obj, valid_obj


if __name__ == '__main__':
    obj = LabeledData()
    obj.download_metadata()

    splitter = Splitter(obj)
    train, test, valid = splitter.train_validation_test_split()
