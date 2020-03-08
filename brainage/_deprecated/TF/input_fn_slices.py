import tensorflow as tf
import pandas as pd # needs xlrd for excel
from pathlib import Path
import re
import matplotlib.pyplot as plt
import numpy as np
import SimpleITK as sitk


def IXI_files(filenames):
    """
    Generator to extract id, slice and absolute path from filenames.
    Filenames are assumed to have the following format:
    ./...IXI<id(3)>.../...sl<slice(4)>.nii...
    Args:
        filenames: List with filenames.

    Returns:

    """
    for file in filenames:
        # Extract id from filename.
        m = re.match('.*/smwc1rIXI([0-9]{3}).*', file)
        id = int(m.group(1))
        # Extract slice number from filename.
        m = re.match('.*_sl([0-9]{4}).nii.*', file)
        slice = int(m.group(1))

        yield id, slice, file


def IXI_read_img(path):
    img = sitk.ReadImage(path.numpy().decode())
    imga = sitk.GetArrayFromImage(img)
    imga = imga.transpose([1, 0])
    return imga


def IXI_info(id, columns, df_info):
    id = id.numpy()
    # Select targets from info dataframe. Subtract 1.0 from SEX_ID.
    targets_df = df_info.loc[id][columns] - [0.0, 1.0]
    # Generate dataset with info elements.
    values = tf.convert_to_tensor(targets_df.values, dtype=tf.float32)
    return values


def _id2filenames(subject_list,
                  path_rootdir,
                  verbose=True):
    """

    Args:
        subject_list:
        path_rootdir:

    Returns: List of filenames.

    """
    path_rootdir = Path(str(path_rootdir))

    # Checks if a corresponding subject directory exists.
    # In this case the path will be appended.
    subject_directories = []
    for id in subject_list:
        id_dir = path_rootdir.glob(f'smwc1rIXI{str(id).zfill(3)}*')
        id_dir = list(id_dir)
        if len(id_dir) > 0:
            subject_directories.append(id_dir[0])
        else:
            if verbose:
                print(f'This id {id} has no directory.')

    # Creates list with all nii filepaths (as string) for all
    # valid subject directories.
    filenames = []
    for dir in subject_directories:
        for file in dir.glob('*nii*'):
            filenames.append(str(file))

    return filenames


def main():
    # Path to nii slices and xls with subject information.
    columns = ['AGE', 'SEX_ID']

    path_rootdir = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/nii2dz')
    path_info = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/IXI_PP_cleaned.csv')
    path_subjects = path_info

    # Create list with selected subjects from csv file.
    df_subjects = pd.read_csv(str(path_subjects))
    subject_list = df_subjects['IXI_ID'].values
    filenames = _id2filenames(subject_list,
                              path_rootdir)

    # Create dataset from filepath generator.
    ds = tf.data.Dataset.from_generator(lambda: IXI_files(filenames),
                                        output_types=(tf.int64, tf.int64, tf.string))

    # Shuffle slices (filenames).
    ds = ds.shuffle(10000)

    # Read csv file with additional patient information.
    df_info = pd.read_csv(path_info, index_col='IXI_ID')

    # Add the subject information (age, sex, ..) for each slice.
    def py_fnc_load_info(id):
        # Loads the subject info from the dataframe.
        return tf.py_function(lambda x: IXI_info(x, columns, df_info),
                              inp=[id],
                              Tout=tf.float32)
    ds = ds.map(lambda id, sl, path: (id, sl, py_fnc_load_info(id), path))

    # Load image data.
    ds = ds.map(lambda id, sl, info, path:
                (id, sl, info, tf.py_function(IXI_read_img,
                                              inp=[path],
                                              Tout=tf.float32)))

    # Test 2D slice dataset.
    ds_slices = ds.batch(32)
    for id, sl, info, img in ds_slices.take(3):
        for i in range(4):
            ax = plt.subplot(1, 4, i + 1)
            print(np.sum(img[i]))
            ax.imshow(img[i], cmap='gray')
            ax.set_title(f'id {id[i]}slice {sl[i]} info {info[i][0]}')
            ax.axis('off')
            print(id)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()




