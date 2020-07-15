import os
from pathlib import Path

import torch
import h5py
import dotenv
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as AbstractDataset
dotenv.load_dotenv()


class SliceDataset(AbstractDataset):

    def __init__(self,
                 data,
                 info,
                 labels=['age'],
                 transform=None):
        """Dataset of slices drawn from a numpy array. 

        The dataframe object should contain the following columns (index=slice number,
        subject key, labels ('age','sex'))
        Args:
            data (np.array): Data array (can be memmapped), number of slices x H x W
            info (pandas.DataFrame): Data frame containing the meta information. One row per slice.
            labels (list): Label columns names.
            transform (class, optional): Tranformation per Image. Defaults to None.
        """

        super().__init__()

        # copy over
        self.info = info
        self.data = data
        self.labels = labels
        self.transform = transform
        #hf = h5py.File(self.data, mode='r')
        #self.ds = hf['image']

    def __len__(self):
        return len(self.info)   

    def __getitem__(self, i):
        # subject
        key = self.info.iloc[i]['key']
        sl = self.info.iloc[i]['slice']
        pos = self.info.iloc[i].name
        img = np.copy(self.data[pos])
        sample = {'data':  img[np.newaxis, np.newaxis, :, :],
                  'label': self.info.iloc[i][self.labels].tolist(),
                  'slice': sl,
                  'key':   key}

        # data augmentation
        # data tensor format BxCxHxWxD (B=C=1)
        if self.transform:
            sample = self.transform(**sample)
        sample['data'] = np.squeeze(sample['data'], axis=0)

        return sample
    
