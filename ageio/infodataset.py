import pandas as pd
import re
import numpy as np

import torch
from torchio import INTENSITY, ImagesDataset, Image
from torch.utils.data import Dataset


def _get_id(filename,
            praefix='IXI',
            digits=3):
    """
    Parse subject id from filename
    Args:
        filename: Stem of the nii filepath (begins with <praefix><id>).
        praefix: Preafix of stem.
        digits: Number of digits after praefix to identify the subject.

    Returns: subject id

    """
    # Parse subject id from filename
    pattern = f'{praefix}([0-9]{{{digits}}}).*'
    m = re.match(pattern, filename)
    subject_id = int(m.group(1))
    return subject_id


class InfoDataset(Dataset):

    def __init__(self,
                 filenames,
                 path_info,
                 columns,
                 praefix='IXI',
                 digits=3,
                 transform=None):

        # Read csv file with additional patient information.
        self.df_info = pd.read_csv(path_info, index_col='ID')
        self.columns = columns
        self.transform = transform
        self.filenames = filenames

        # Praefix and number of digits in file stem.
        self.praefix = praefix
        self.digits = digits

        # Create ImageDataset from filenames.
        subject_images = [[Image('img', f, INTENSITY)] for f in filenames]
        self.dataset = ImagesDataset(subject_images, transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.dataset[idx]
        # Get filename to extract ID.
        f = sample['img']['stem']
        # Add info to sample dict.
        subj_id = _get_id(f,
                          praefix=self.praefix,
                          digits=self.digits)
        sample['img']['id'] = subj_id
        # Select target columns from info dataframe and
        # store into values into sample dict.
        info = self.df_info.loc[subj_id][self.columns]
        info = np.array(info.values, dtype=np.float32)
        sample['img']['info'] = torch.tensor(info)
        return sample

