import pandas as pd
from pathlib import Path

import torch
import torchvision

from torchio.transforms import (
    ZNormalization,
    RandomNoise,
    RandomAffine,
    RandomFlip
)

from ageio.infodataset import InfoDataset


def _id2niipath(subject_list,
                path_nii,
                pattern):
    """
    Converts list of subject id to nii filepaths. Only existing paths are added to the returned list.

    Args:
        subject_list: List with subject ids.
        path_nii: Path to nii directory.
        pattern: String template e.g. '*smwc1rIXI{:03d}*'

    Returns: List of filenames.

    """
    path_nii = Path(str(path_nii))

    # Foreach subject get filepath for the associated nii file (if it exists).
    filenames = []
    for id in subject_list:
        file = list(path_nii.glob(pattern.format(id)))
        if file:
            filenames.append(str(file[0]))
    return filenames


def create_IXI_datasets(validation_split=50):
    #validation_split = 50
    # todo use test_split
    test_split = 400

    # Define format of filenames.
    praefix = 'IXI'
    digits  = 3
    pattern = f'{praefix}{{:0{digits}}}*nii*'

    # Path to nii files and csv with subject information.
    #path_nii = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/W_IXIT1')
    #path_info = '/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/IXI_PP_cleaned.csv'

    path_info = '/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/IXI_PP_cleaned.csv'
    path_nii = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/nii3d')

    # Columns to extract target information from.
    columns = ['AGE', 'SEX_ID']
    # Create list with selected subjects from csv file.
    df_subjects = pd.read_csv(path_info)
    subject_list = df_subjects['ID'].values

    # Convert subject indices to nii filenames.
    filenames = _id2niipath(subject_list, path_nii, pattern)

    # Data augmentation.
    #transforms = (
    #    ZNormalization()#
        #RandomAffine(scales=(0.9, 1.1), degrees=5),
        #RandomNoise(std_range=(0, 0.25)),
        #RandomFlip(axes=(0,))
    #    )
    #transform = torchvision.transforms.Compose(transforms)

    # Define training and validation dataset.
    train_dataset = InfoDataset(filenames[validation_split:],
                                path_info, columns,
                                praefix='IXI',
                                digits=3,
                                transform=ZNormalization())

    validate_dataset = InfoDataset(filenames[:validation_split],
                                   path_info, columns,
                                   praefix='IXI',
                                   digits=3,
                                   transform=ZNormalization())

    return train_dataset, validate_dataset