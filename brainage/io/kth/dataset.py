import pathlib
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import logging
import torch.nn as nn
import torch.optim as optim
import collections
import copy
from torch.utils.data import Dataset as AbstractDataset
from torch.utils.data import DataLoader

class Dataset(AbstractDataset):
    def __init__(self,
                 fname: pathlib.Path,
                 ix_fname: pathlib.Path,
                 age_data,
                 transform=None):

        super().__init__()

        # copy over
        self.fname = fname
        self.ix_fname = ix_fname
        self.age_data = age_data
        self.transform = transform
        self.logger = logging.getLogger(__name__)

        # read indices
        self.ix = sorted(set(self.ix_fname.read_text().strip().split("\n")))[:16]

        self.logger.info('loading dataset to memory ...')
        t = time.perf_counter()
        # reading data from hdf5 to container object
        with h5py.File(self.fname, 'r') as fhandle:
            def load_data():
                for example_ix in self.ix:
                    self.logger.debug(f'loading {example_ix}')
                    # extract data
                    original = fhandle['original'][example_ix][:]
                    age = float(self.age_data[example_ix])
                    data = {'data': original[np.newaxis, np.newaxis, ...],
                            'age': age}
                    yield data
            self.data_container = collections.deque(load_data())
        self.logger.info(f'finished in {time.perf_counter() -t } seconds')

    def __len__(self):
        return len(self.ix)

    def __getitem__(self, i):
        sample = self.data_container[i]

        # data augmentation
        if self.transform:
            sample = self.transform(**sample)

        # add additional dimension
        age = np.expand_dims(np.array(sample['age']), axis=0)
        # squeeze batch dimension
        data = np.squeeze(sample['data'] ,axis=0)
        return torch.tensor(data), torch.tensor(age)