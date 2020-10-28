import logging
import time
from pathlib import Path
from collections import deque, defaultdict

import h5py
import zarr
import torch
import tracemalloc
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, IterableDataset


class DataReader:

    def read(self, group_key, subj_keys, dtype=True, preload=True):
        pass
    
    def read_data_to_memory(self, subject_keys, group, dtype=np.float16, preload=True):    
        """Reads data from source to memory.
        
        The dataset should be stored using the following structure:
        <data_path>/<group>/<key>... 
        A generator function (data_generator) can be defined to read data respecting this
        structure (implementations for hdf5/zarr/nifti directory are available).

        Args:
            subject_keys (list): identifying keys
            group (str): data group name
            dtype (type, optional): store dtype (default np.float16/np.uint8). Defaults to np.float16.
            preload (bool, optional): if False, data will be loaded on the fly. Defaults to True.
        
        Returns
            object: collections.deque list containing the dataset
        """
        logger = logging.getLogger(__name__)
        logger.info(f'loading group [{group}]...')
        # check timing and memory allocation
        t = time.perf_counter()
        tracemalloc.start()
        data = deque(self.read(subject_keys, group, dtype, preload))
        current, peak = tracemalloc.get_traced_memory()
        logger.debug(f'finished: {time.perf_counter() - t :.3f} s, current memory usage {current / 10**9: .2f}GB, peak memory usage {peak / 10**9:.2f}GB')
        return data
    
    def get_data_shape(self, subject_keys, group):
        pass

    def get_data_attribute(self, subject_keys, group, attribute):
        pass

    def close(self):
        pass


class DataReaderHDF5(DataReader):

    def __init__(self, path_data):
        self.path_data = path_data
        self.hf = h5py.File(str(path_data), 'r')
        self.logger = logging.getLogger(__name__)

    def read(self, subject_keys, group, dtype=np.float16, preload=True):
        for k in tqdm(subject_keys):
            data = self.hf[f'{group}/{k}']
            if preload:
                data = data[:].astype(dtype)
            yield data[np.newaxis, ...]

    def get_data_shape(self, subject_keys, group):
        shapes = {}
        for k in subject_keys:
            shapes[k] = np.array(self.hf[f'{group}/{k}'].shape)
            shapes[k] = np.insert(shapes[k], 0, 1)
        return shapes

    def get_data_attribute(self, subject_keys, group, attribute):
        attr = {}
        for k in subject_keys:
            attr[k] = self.hf[f'{group}/{k}'].attrs[attribute]
        return attr

    def close(self):
        self.hf.close()


def grid_patch_generator(img, patch_size, patch_overlap, **kwargs):
    """Generates grid of overlapping patches.

    All patches are overlapping (2*patch_overlap per axis).
    Cropping the original image by patch_overlap.
    The resulting patches can be re-assembled to the 
    original image shape.
    
    Additional np.pad argument can be passed via **kwargs.

    Args:
        img (np.array): CxHxWxD 
        patch_size (list/np.array): patch shape [H,W,D]
        patch_overlap (list/np.array): overlap (per axis) [H,W,D]
    
    Yields:
        np.array, np.array, int: patch data CxHxWxD, 
                                 patch position [H,W,D], 
                                 patch number
    """
    dim = 3
    patch_size = np.array(patch_size)
    img_size = np.array(img.shape[1:])
    patch_overlap = np.array(patch_overlap)
    cropped_patch_size = patch_size - 2*patch_overlap
    n_patches = np.ceil(img_size/cropped_patch_size).astype(int)
    overhead = cropped_patch_size - img_size % cropped_patch_size
    padded_img = np.pad(img, [[0,0],
                              [patch_overlap[0], patch_overlap[0] + overhead[0]],
                              [patch_overlap[1], patch_overlap[1] + overhead[1]],
                              [patch_overlap[2], patch_overlap[2] + overhead[2]]], **kwargs)
    pos = [np.arange(0, n_patches[k])*cropped_patch_size[k] for k in range(dim)]
    count = -1
    for p0 in pos[0]:
        for p1 in pos[1]:
            for p2 in pos[2]:
                idx = np.array([p0, p1, p2])
                idx_end = idx + patch_size
                count += 1
                patch = padded_img[:, idx[0]:idx_end[0], idx[1]:idx_end[1], idx[2]:idx_end[2]]
                yield patch, idx, count


class GridPatchSampler(IterableDataset):

    def __init__(self,
                 data_path,
                 subject_keys,
                 patch_size, patch_overlap,
                 out_channels=1,
                 out_dtype=np.uint8,
                 image_group='images',
                 ReaderClass=DataReaderHDF5,
                 pad_args={'mode': 'symmetric'}):
        """GridPatchSampler for patch based inference.
        
        Creates IterableDataset of overlapping patches (overlap between neighboring
        patches: 2*patch_overlapping). 
        To assemble the original image shape use add_processed_batch(). The 
        number of channels for the assembled images (corresponding to the 
        channels of the processed patches) has to be defined by num_channels: 
        <num_channels>xHxWxD.

        Args:
            data_path (Path/str): data path (e.g. zarr/hdf5 file)
            subject_keys (list): subject keys
            patch_size (list/np.array): [H,W,D] patch shape
            patch_overlap (list/np.array): [H,W,D] patch boundary
            out_channels (int, optional): number of channels for the processed patches. Defaults to 1.
            out_dtype (dtype, optional): data type of processed patches. Defaults to np.uint8. 
            image_group (str, optional): image group tag . Defaults to 'images'.
            ReaderClass (function, optional): data reader class. Defaults to DataReaderHDF5.
            pad_args (dict, optional): additional np.pad parameters. Defaults to {'mode': 'symmetric'}.
        """
        self.data_path = str(data_path)
        self.subject_keys = subject_keys
        self.patch_size = np.array(patch_size)
        self.patch_overlap = patch_overlap
        self.image_group = image_group
        self.ReaderClass = ReaderClass
        self.out_channels = out_channels
        self.out_dtype = out_dtype
        self.results = zarr.group()
        self.originals = {}
        self.pad_args = pad_args

        # read image data for each subject in subject_keys
        reader = self.ReaderClass(self.data_path)
        self.data_shape = reader.get_data_shape(self.subject_keys, self.image_group)
        self.data_affine = reader.get_data_attribute(self.subject_keys, self.image_group, 'affine')
        self.data_generator = reader.read_data_to_memory(self.subject_keys, self.image_group, dtype=np.float16)
        reader.close()
    
    def add_processed_batch(self, sample):
        """Assembles the processed patches to the original array shape.
        
        Args:
            sample (dict): 'key', 'position', 'data' (C,H,W,D) for each patch  
        """
        for i, key in enumerate(sample['key']):
            # crop patch overlap
            cropped_patch = np.array(sample['data'][i, :,
                                                    self.patch_overlap[0]:-self.patch_overlap[1],
                                                    self.patch_overlap[1]:-self.patch_overlap[1],
                                                    self.patch_overlap[2]:-self.patch_overlap[2]])
            # start and end position
            pos = np.array(sample['position'][i])
            pos_end = np.array(pos + np.array(cropped_patch.shape[1:]))
            # check if end position is outside the original array (due to padding)
            # -> crop again (overhead)
            img_size = np.array(self.data_shape[key][1:])
            crop_pos_end = np.minimum(pos_end, img_size)
            overhead = np.maximum(pos_end - crop_pos_end, [0, 0, 0])
            new_patch_size = np.array(cropped_patch.shape[1:]) - overhead
            # add the patch to the corresponing entry in the result container
            ds_shape = np.array(self.data_shape[key])
            ds_shape[0] = self.out_channels
            ds = self.results.require_dataset(key, shape=ds_shape, chunks=False, dtype=self.out_dtype)
            ds.attrs['affine'] = np.array(self.data_affine[key]).tolist()
            ds[:, pos[0]:pos_end[0],
                  pos[1]:pos_end[1],
                  pos[2]:pos_end[2]] = cropped_patch[:, :new_patch_size[0],
                                                        :new_patch_size[1],
                                                        :new_patch_size[2]].astype(self.out_dtype)

    def get_assembled_data(self):
        """Gets the dictionary with assembled/processed images.
        
        Returns:
            dict: Dictionary containing the processed and assembled images (key=subject_key)
        """
        return self.results

    def grid_patch_sampler(self):
        """Data reading and patch generation.
        
        Yields:
            dict: patch dictionary (subject_key, position, count and data)
        """
        
        # create a patch iterator 
        for subj_idx, sample in enumerate(tqdm(self.data_generator)):
            subject_key = self.subject_keys[subj_idx]
            # create patches
            result_shape = np.array(sample.shape)
            result_shape[0] = self.out_channels
            patch_generator = grid_patch_generator(
                sample, self.patch_size, self.patch_overlap, **self.pad_args)
            for patch, idx, count in patch_generator:
                patch_dict = {'data': patch[: , :, :, :],
                              'key': subject_key,
                              'position': idx,
                              'count': count}
                yield patch_dict

    def __iter__(self):
        return iter(self.grid_patch_sampler())

    def __len__(self):
        return 1