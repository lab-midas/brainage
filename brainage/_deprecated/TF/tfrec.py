import SimpleITK as sitk
import tensorflow as tf
import numpy as np
import re
from pathlib import Path


def serialize_image(img, idtag=''):
    """

    Args:
        img:
        idtag:

    Returns:

    """
    # Get data array from sitk img object.
    img_data = sitk.GetArrayFromImage(img)
    img_data = np.transpose(img_data, [2, 1, 0])
    # Serialize image information.
    return serialize_example(origin=img.GetOrigin(),
                             shape=img.GetSize(),
                             spacing=img.GetSpacing(),
                             direction=img.GetDirection(),
                             data=img_data,
                             idtag=idtag)


def serialize_example(origin, shape, spacing, direction, data, idtag=''):
    """ 
    Creates a tf.Example message ready to be written to a file.
    
    Args:
        origin: 
        shape: 
        spacing: 
        direction: 
        data: 
        idtag:

    Returns:

    """
    
    # Convert data array to flat byte list.
    data_bytes = data.flatten()
    data_bytes = data_bytes.astype(np.float32).tobytes()

    # Create a dictionary mapping the feature name to the tf.Example-compatible
    # data type.
    feature = {
        'id':           tf.train.Feature(bytes_list=tf.train.BytesList(value=[idtag.encode()])),
        'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=shape)),
        'origin':       tf.train.Feature(float_list=tf.train.FloatList(value=origin)),
        'spacing':      tf.train.Feature(float_list=tf.train.FloatList(value=spacing)),
        'direction':    tf.train.Feature(float_list=tf.train.FloatList(value=direction)),
        'data':         tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_bytes]))
    }

    # Create a Features message using tf.train.Example.
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def convert_nii_to_tfrec(path_nii, path_tfrec, idtag=''):
    """

    Args:
        path_nii:
        path_tfrec:
        idtag:

    Returns:

    """

    # Read .nii file to sitk img object.
    reader = sitk.ImageFileReader()
    reader.SetImageIO('NiftiImageIO')
    reader.SetFileName(str(path_nii))
    img = reader.Execute()

    # Write tfrecord file.
    write_tfrec(img, path_tfrec)


def convert_tfrec_to_nii(path_tfrec, path_nii, idtag=''):
    """

    Args:
        path_tfrec:
        path_nii:
        idtag:

    Returns:

    """

    # Read tfrec file and store into sitk image.
    img = read_tfrec(path_tfrec)
    # Save img as nii file.
    writer = sitk.ImageFileWriter()
    writer.SetFileName(str(path_nii))
    writer.Execute(img)


def write_tfrec(img, path_tfrec, idtag=''):
    """

    Args:
        img:
        path_tfrec:
        idtag:
    Returns:

    """

    # Write tfrecord file.
    with tf.io.TFRecordWriter(str(path_tfrec)) as writer:
        example = serialize_image(img, idtag)
        writer.write(example)


def read_tfrec(path_tfrec):
    """

    Args:
        path_tfrec:

    Returns:

    """

    # Read tfrecord file.
    filenames = [str(path_tfrec)]
    raw_dataset = tf.data.TFRecordDataset(filenames)

    for example in raw_dataset.take(1):
        # Parse serialized example.
        content = parse_example_all(example)

        # Convert parsed content to sitk img object.
        img = sitk.Image(content['shape'].numpy().tolist(),
                         sitk.sitkFloat32)
        # Create sitk image object from data array.
        data_array = content['data'].numpy()
        data_array = np.transpose(data_array, [2, 1, 0])
        new_img = sitk.GetImageFromArray(data_array)
        # Restore img coordinate system.
        img.SetOrigin(content['origin'].numpy().astype(np.float64))
        img.SetSpacing(content['spacing'].numpy().astype(np.float64))
        img.SetDirection(content['direction'].numpy().astype(np.float64))

        return img, content['id']


feature_description = {
        'id':           tf.io.FixedLenFeature([], tf.string),
        'shape':        tf.io.FixedLenFeature([3], tf.int64),
        'origin':       tf.io.FixedLenFeature([3], tf.float32),
        'spacing':      tf.io.FixedLenFeature([3], tf.float32),
        'direction':    tf.io.FixedLenFeature([9], tf.float32),
        'data':         tf.io.FixedLenFeature([], tf.string)
}

def parse_example_all(example):
    """

    Args:
        example:

    Returns:

    """

    # Parse the input `tf.Example` proto using the dictionary above.
    content = tf.io.parse_single_example(serialized=example, features=feature_description)

    # Get image tensor with correct shape from raw data.
    data = tf.io.decode_raw(content['data'], tf.float32)
    data = tf.reshape(data, content['shape'])
    content['data'] = data

    return content


def parse_example(example):
    """

        Args:
            example:

        Returns:

    """
    # Parse the input `tf.Example` proto using the dictionary above.
    content = tf.io.parse_single_example(serialized=example, features=feature_description)

    # Get image tensor with correct shape from raw data.
    data = tf.io.decode_raw(content['data'], tf.float32)
    data = tf.reshape(data, content['shape'])

    return data



def main():

    seq_name = 'PP'
    path_nii_dir = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/nii3d')
    path_tfrec_dir = Path('/mnt/share/raheppt1/project_data/brain/IXI/IXI_PP/tfrec3d')

    def convert_ixi_nii2tfrec(path_nii,
                              path_tfrec_dir,
                              seq_name=''):
        # Get pat_id from filename ...IXI<pat_id>...
        pat_id_str = path_nii.name
        m = re.match('.*IXI([0-9]{3}).*', pat_id_str)
        pat_id_str = m.group(1)
        pat_id = int(pat_id_str)
        print(f'ID: {pat_id}')

        path_tfrec = path_tfrec_dir.joinpath(f'IXI{seq_name}_{str(pat_id).zfill(4)}.tfrec')
        convert_nii_to_tfrec(path_nii, path_tfrec)

    for path_nii in path_nii_dir.glob('*.nii*'):
        print(path_nii)
        convert_ixi_nii2tfrec(path_nii, path_tfrec_dir, seq_name)

    #path_out_nii = path_tfrec_dir.joinpath(str(pat_id) + 'out.nii')
    #convert_tfrec_to_nii(path_tfrec, path_out_nii)


if __name__ == '__main__':
    main()

