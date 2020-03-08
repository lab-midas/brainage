from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from pathlib import Path
import re
import SimpleITK as sitk

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class AgeDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, subject_ids, filenames, path_info, transform=None):

        df_info = pd.read_csv(path_info, index_col='IXI_ID')
        columns = ['AGE', 'SEX_ID']
        # Select targets from info dataframe. Subtract 1.0 from SEX_ID.
        self.infos = df_info.loc[subject_ids][columns] - [0.0, 1.0]

        self.filenames = filenames
        self.subject_ids = subject_ids
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        id = self.subject_ids[idx]
        filepath = self.filenames[idx]
        info = self.infos[idx]

        img = sitk.ReadImage(filepath)
        img = sitk.GetArrayFromImage(img)
        img = img.transpose([2, 1, 0])

        sample = {'id': id, 'info': info, 'img': img}

        if self.transform:
            sample = self.transform(sample)

        return sample


def _id2nii(subject_list, path_data):
    path_data = Path(path_data)
    files = {}
    for file in path_data.glob('*.nii'):
        m = re.match('smwc1rIXI([0-9]{3})-.*', file.name)
        id = int(m.group(1))
        files[id] = str(file)
    ids, filenames = zip(
        *[(id, files[int(id)]) for id in subject_list if id in files.keys()])
    return ids, filenames

    #filenames = [f'{series_tag}{str(idx).zfill(4)}.tfrec'
    #             for idx in subject_list]

    # Convert to absolute paths, keep only valid paths.
    #filenames = [os.path.join(str(path_data), filename) for filename in filenames]
    #filenames = [file for file in filenames if os.path.exists(file)]
    #return filenames

def main():
    # Path to tfrecord files and xls with subject information.
    # todo change csv files
    path_info = '/mnt/share/raheppt1/project_data/brain/IXI/IXI_cleaned.csv'
    path_data = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/nii3d')
    path_subjects = '/mnt/share/raheppt1/project_data/brain/IXI/IXI_cleaned.csv'

    # Create list with selected subjects from csv file.
    df_subjects = pd.read_csv(path_subjects)
    subject_list = df_subjects['IXI_ID'].values

    # Convert subject indices to tfrec filenames "PP<idx>.tfrec".
    ids, filenames = _id2nii(subject_list, path_data)

    # filenames = _id2tfrecpath(subject_list, series_tag, path_tfrec)
    #
    # # Create ageio.
    # ds_imgs = create_ds_images(filenames)
    # ds_info = create_ds_info(subject_list, path_info)
    # ds_id = tf.data.Dataset.from_tensor_slices(subject_list)
    #
    # # Zip to one dataset.
    # ds = tf.data.Dataset.zip((ds_id, ds_info, ds_imgs))
    #
    # # Create dataset with (slice numbers, 2d slices image and info) elements.
    # ds_slices = ds.flat_map(img_to_slices)
    #
    # # Test 3D dataset.
    # for id, info, image in ds.take(2):
    #      print(f'image shape {tf.shape(image)} : AGE, SEX_ID: {info}')
    #      plt.imshow(image[:, :, 60])
    #      plt.show()
    #
    # # Test 2D dataset.
    # ds_slices = ds_slices.shuffle(10000).batch(32)
    # fig = plt.figure()
    # for id, sl, info, img in ds_slices.take(1):
    #     for i in range(4):
    #         ax = plt.subplot(1, 4, i+1)
    #         ax.imshow(img[i], cmap='gray')
    #         ax.set_title(f'id {id[i]} slice {sl[i]} info {info[i][0]}')
    #         ax.axis('off')
    # plt.tight_layout()
    # plt.show()

if __name__ == '__main__':
    main()