import tensorflow as tf
import pandas as pd # needs xlrd for excel
from pathlib import Path
import deprecated.TF.tfrec
import matplotlib.pyplot as plt
import os.path


def create_ds_info(subject_list,
                   path_info):
    """ Creates tf Dataset with selected information from the
    dataframe (e.g. age).

    Args:
        subject_list: List with subject indices.
        path_excel: Path to IXI excel sheet.

    Returns: Dataset with selected information.

    """
    columns = ['AGE', 'SEX_ID']
    # Read csv file with additional patient information.
    df = pd.read_csv(path_info, index_col='IXI_ID')
    # Select targets from info dataframe. Subtract 1.0 from SEX_ID.
    targets_df = df.loc[subject_list][columns] - [0.0, 1.0]
    # Generate dataset with info elements.
    values = tf.convert_to_tensor(targets_df.values, dtype=tf.float32)
    ds_info = tf.data.Dataset.from_tensor_slices(values)

    return ds_info


def create_ds_images(filenames):
    """ Creates tf Dataset with the image data stored in the selected
    tfrec files.

    Args:
        subject_list: List with subject indices.
        path_tfrec: Path to tfrec directory.
        series_tag: Tag to identify the series from tfrec filenames.
        <series_tag><idx>.tfrec

    Returns: Dataset with 3d image tensors.

    """

    # Parsing images from tfrec files.
    ds_imgs = tf.data.TFRecordDataset(filenames=filenames)
    ds_imgs = ds_imgs.map(deprecated.TF.tfrec.parse_example)

    return ds_imgs


def img_to_slices(id, info, img):
    """ Creates dataset with numbered image slices for flat_map.
    The info tensor is repeated foreach slice.
    The 2nd image dimension ist used for slicing.

    Args:
        id: subject id
        img: 3D image tensor.
        info: Tensor with additional information (e.g. age)

    Returns: Zipped ageio for slice numbers, image slices and information.

    """

    # Slice along <slice_dim>, std-value 2.
    slice_dim = 2
    slice_perm = [2, 0, 1]
    number_of_slices = tf.shape(img)[slice_dim]

    # Number slices from 0 to (num_slices - 1).
    slice_numbers = tf.range(tf.shape(img)[slice_dim], dtype=tf.float32)
    slice_numbers = tf.data.Dataset.from_tensor_slices(slice_numbers)

    # Repeat info for all slices.
    info = tf.data.Dataset.from_tensors(info)
    info = info.repeat(tf.cast(number_of_slices, tf.int64))
    id = tf.data.Dataset.from_tensors(id)
    id = id.repeat(tf.cast(number_of_slices, tf.int64))

    # Transpose image tensor, because from_tensor_slices operates on
    # the first dimension
    img = tf.transpose(img, perm=slice_perm)
    img_slices = tf.data.Dataset.from_tensor_slices(img)

    # Foreach img tensor two ageio (numbers, image slices) are
    # created.
    return tf.data.Dataset.zip((id, slice_numbers, info, img_slices))


def _id2tfrecpath(subject_list,
                  series_tag,
                  path_tfrec):
    """
    Converts list of subject id to tfrec filepaths: <path_tfrec>/<series_tag><id(4).tfrec
    Only existing paths are added to the returned list.

    Args:
        subject_list:
        series_tag:
        path_tfrec:

    Returns: List of filenames.

    """
    filenames = [f'{series_tag}{str(idx).zfill(4)}.tfrec'
                 for idx in subject_list]

    # Convert to absolute paths, keep only valid paths.
    filenames = [os.path.join(str(path_tfrec), filename) for filename in filenames]
    filenames = [file for file in filenames if os.path.exists(file)]
    return filenames


def main():
    # Path to tfrecord files and xls with subject information.
    path_info = '/mnt/share/raheppt1/project_data/brain/IXI/IXI_cleaned.csv'
    path_tfrec = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/tfrec3d')
    path_subjects = '/mnt/share/raheppt1/project_data/brain/IXI/IXI_cleaned.csv'

    # Create list with selected subjects from csv file.
    df_subjects = pd.read_csv(path_subjects)
    subject_list = df_subjects['IXI_ID'].values

    # Convert subject indices to tfrec filenames "PP<idx>.tfrec".
    series_tag = 'IXIPP_'
    filenames = _id2tfrecpath(subject_list, series_tag, path_tfrec)

    # Create ageio.
    ds_imgs = create_ds_images(filenames)
    ds_info = create_ds_info(subject_list, path_info)
    ds_id = tf.data.Dataset.from_tensor_slices(subject_list)

    # Zip to one dataset.
    ds = tf.data.Dataset.zip((ds_id, ds_info, ds_imgs))

    # Create dataset with (slice numbers, 2d slices image and info) elements.
    ds_slices = ds.flat_map(img_to_slices)

    # Test 3D dataset.
    for id, info, image in ds.take(2):
         print(f'image shape {tf.shape(image)} : AGE, SEX_ID: {info}')
         plt.imshow(image[:, :, 60])
         plt.show()

    # Test 2D dataset.
    ds_slices = ds_slices.shuffle(10000).batch(32)
    fig = plt.figure()
    for id, sl, info, img in ds_slices.take(1):
        for i in range(4):
            ax = plt.subplot(1, 4, i+1)
            ax.imshow(img[i], cmap='gray')
            ax.set_title(f'id {id[i]} slice {sl[i]} info {info[i][0]}')
            ax.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()


