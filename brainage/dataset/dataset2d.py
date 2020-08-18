import os
from sys import getsizeof
from pathlib import Path

import torch
import time
import h5py
import dotenv
import numpy as np
import pandas as pd
import scipy.ndimage
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as AbstractDataset
dotenv.load_dotenv()

class SliceDataset(AbstractDataset):

    def __init__(self,
                 data,
                 info,
                 labels=['age'],
                 image_group='image',
                 preload=True,
                 zoom=None,
                 transform=None):
        """Dataset of slices drawn from a numpy array. 

        The dataframe object should contain the following columns (index=slice number,
        subject key, labels ('age','sex'))
        Args:
            data (np.array): Data array (can be memmapped), number of slices x H x W
            info (pandas.DataFrame): Data frame containing the meta information. One row per slice.
            labels (list): Label columns names.
            image_group (str): Group name of the image datasets. Defaults to 'image'.
            preload (bool): Preload dataset to memory. Defaults to True.
            zoom (float): Zoom image. Defaults to None. 
            transform (class, optional): Tranformation per Image. Defaults to None.
        """

        super().__init__()

        # copy over
        self.info = info
        self.data = data
        self.labels = labels
        self.transform = transform
        self.preload = preload
        self.zoom = zoom

        hf = h5py.File(self.data, mode='r')
        if self.preload:
            t0 = time.perf_counter()
            print('loading data to memory ...')
            self.ds = hf['image'][self.info.index][:]
            print(f'finished {self.ds.nbytes/1e6:.2f} MB - {time.perf_counter() - t0:.2f}s ')
        else:
            self.ds = hf['image']

    def __len__(self):
        return len(self.info)   

    def __getitem__(self, i):
        # subject
        key = self.info.iloc[i]['key']
        sl = self.info.iloc[i]['position']
        pos = self.info.iloc[i].name
        img = self.ds[i] if self.preload else self.ds[pos]
        img = img.astype(np.float32)
        if self.zoom:
            img = scipy.ndimage.zoom(img, self.zoom)

        sample = {'data':  img[np.newaxis, np.newaxis, :, :],
                  'label': self.info.iloc[i][self.labels].tolist(),
                  'position': sl,
                  'key':   key}

        # data augmentation
        # data tensor format BxCxHxWxD (B=C=1)
        if self.transform:
            sample = self.transform(**sample)
        sample['data'] = np.squeeze(sample['data'], axis=0)

        return sample
    
