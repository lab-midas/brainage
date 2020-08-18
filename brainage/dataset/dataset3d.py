import time
import logging
import collections
import copy
from pathlib import Path

import torch
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset as AbstractDataset

class BrainDataset(AbstractDataset):
    def __init__(self,
                 data,
                 keys,
                 info,
                 group,
                 column='label',
                 preload=False,
                 transform=None):

        super().__init__()

        # copy over
        self.transform = transform
        self.logger = logging.getLogger(__name__)
        self.preload = preload

        self.logger.info('opening dataset ...')
        info_df = pd.read_csv(info, index_col=0, dtype={'key': 'string', column: np.float32})
        self.keys = keys
        
        fhandle = h5py.File(data, 'r')
        def load_data():
            for key in tqdm(self.keys):
                label = info_df.loc[key][column]
                group_str = group + '/' if group else ''
                if self.preload:
                    data = fhandle[f'{group_str}{key}'][:]
                else:
                    data = fhandle[f'{group_str}{key}']
                sample = {'data': data,
                          'label': label,
                          'key': key}
                yield sample
        self.data_container = collections.deque(load_data())

    def __len__(self):
        return len(self.data_container)

    def __getitem__(self, i):
        ds = self.data_container[i]
        sample = {'data':   ds['data'][:][np.newaxis, np.newaxis, ...].astype(np.float32),
                  'label':  ds['label'],
                  'key':    ds['key']}
        # data augmentation
        # data tensor format B x C X H X W X D (B=C=1)
        if self.transform:
            sample = self.transform(**sample)
        sample['data'] = np.squeeze(sample['data'], axis=0)
        return sample


def get_random_patch_indices(patch_size, img_shape, pos=None):
    ''' Create random patch indices.
    
    Creates (valid) max./min. corner indices of a patch.
    If a specific position is given, the patch must contain
    this index position. If position is None, a random
    patch will be produced.
    
    Args:
        patch_size (np.array): patch dimensions (H,W,D)
        img_shape  (np.array): shape of the image (H,W,D)
        pos (np.array, optional): specify position (H,W,D), wich should be 
                    included in the sampled patch. Defaults to None.
    
    Returns:
        (np.array, np.array): patch corner indices (e.g. first axis 
                              index_ini[0]:index_fin[0])
    '''
    # 3d - image array should have shape H,W,D
    # if idx is given, the patch has to surround this position
    if pos:
        pos = np.array(pos, dtype=np.int)
        min_index = np.maximum(pos-patch_size+1, 0)
        max_index = np.minimum(img_shape-patch_size+1, pos+1)
    else:
        min_index = np.array([0, 0, 0])
        max_index = img_shape-patch_size+1

    # create valid patch boundaries
    index_ini = np.random.randint(low=min_index, high=max_index)
    index_fin = index_ini + patch_size
    box = (slice(index_ini[0], index_fin[0], 1),
           slice(index_ini[1], index_fin[1], 1),
           slice(index_ini[2], index_fin[2],1 ))
    return box


class BrainPatchDataset(AbstractDataset):

    def __init__(self,
                 data,
                 keys,
                 info,
                 patch_size: (int, int, int),
                 group: str,
                 column='label',
                 preload=False,
                 transform=None):

        super().__init__()

        # copy over
        self.transform_patch = transform
        self.patch_size = np.array(patch_size)
        self.logger = logging.getLogger(__name__)
        self.preload = preload

        self.logger.info('opening dataset ...')
        info_df = pd.read_csv(info, index_col=0, dtype={'key': 'string', column: np.float32})
        self.keys = keys
        fhandle = h5py.File(data, 'r')
        def load_data():
            for key in tqdm(self.keys):
                label = info_df.loc[key][column]
                group_str = group + '/' if group else ''
                if self.preload:
                    data = fhandle[f'{group_str}{key}'][:]
                else:
                    data = fhandle[f'{group_str}{key}']
                sample = {'data': data,
                          'label': label,
                          'key': key}
                yield sample
        self.data_container = collections.deque(load_data())

    def __len__(self):
        return len(self.data_container)

    def __getitem__(self, i):
        ds = self.data_container[i]
        box = get_random_patch_indices(self.patch_size, ds['data'].shape)
        data = ds['data'][box[0].start:box[0].stop,
                          box[1].start:box[1].stop,
                          box[2].start:box[2].stop]
                          
        sample = {'data':   data[np.newaxis, np.newaxis, :, :].astype(np.float32),
                  'label':  ds['label'],
                  'key':    ds['key']}

        # data augmentation
        # data tensor format B x C X H X W X D (B=C=1)
        if self.transform_patch:
            sample = self.transform_patch(**sample)

        sample['data'] = np.squeeze(sample['data'], axis=0)
        sample['position'] = np.array([b.start for b in list(box)])
        return sample


# TODO move to unit test
def test_brain_dataset():
    logging.basicConfig(level=logging.DEBUG)

    info = '/mnt/qdata/raheppt1/data/brainage/nako/interim/nako_age_labels.csv'
    keys = '/mnt/qdata/raheppt1/data/brainage/nako/interim/debug1.dat'
    data = '/mnt/qdata/raheppt1/data/brainage/nako/interim/t1_pp_15_stripped.h5'
    epochs = 10
    slice_selection = range(0,3)
    ds = BrainDataset(data, keys, info, 'image', column='age', preload=True)

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=True, num_workers=8)

    for epoch in range(epochs):
        t = time.perf_counter()
        for k, sample in enumerate(dl):
            #print(sample)
            #print(sample['label'])
            print(torch.max(sample['data'][0]))
            a = sample
        print(f'epoche {epoch}: {(time.perf_counter() -t)/k} s')
   
    print(sample['data'].shape)


def test_brainpatch_dataset():
    info = '/home/thepp/data/info/ADNI/ADNI_age_all.csv'
    keys = '/home/thepp/data/info/ADNI/test.dat'
    data = '/media/datastore1/thepp/ADNI_EVAL/original.h5'
    nepochs = 10
    ds = BrainPatchDataset(data, keys, info, group='original', patch_size = (50,50,50), preload=False)

    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=8, shuffle=True)

    for epoch in range(nepochs):
        t = time.perf_counter()
        for k, sample in enumerate(dl):
            print(sample)
        print(f'epoche {epoch}: {(time.perf_counter() -t)/k} s')
   
    print(sample['data'].shape)


if __name__ == '__main__':
    test_brain_dataset()