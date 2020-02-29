import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import tensorflow as tf


def init_tensorboard(praefix):
    """
    Create tensoboard summary writer in logdir
     .logs/<praefix>log<timestamp>
    Args:
        praefix:

    Returns:

    """

    import datetime
    log_dir = Path('./logs/tf')
    ts = datetime.datetime.now().timestamp()
    readable = datetime.datetime.fromtimestamp(ts).isoformat()
    log_dir = str(log_dir.joinpath(praefix+'log_' + readable))
    writer = SummaryWriter(log_dir)
    return writer


def init_gpu(gpu_device):
    print(tf.__version__)

    # Define GPU device.
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_device

    # Activate memory growth.
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def vis_brain(image, slice_sel, axis_sel=1, normalize=False):
    if normalize:
        image = (image - np.mean(image)) / (np.std(image))

    if axis_sel == 1:
        image = image.transpose([1, 2, 0])
    elif axis_sel == 2:
        image = image.transpose([2, 0, 1])

    fig = plt.figure(figsize=(7., 7.))
    plt.imshow(image[slice_sel, :, :], cmap='gray')
    plt.axis('off')
    plt.show()