import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


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


def add_dataset_config(config,
                       dataset='IXI',
                       split='0',
                       ADNI_test_selection='split',
                       ADNI_group='ADNL'):
    
    data_dir = os.getenv("DATA_DIR")

    if dataset == 'IXI':
        # IXI
        config['base_folder'] = os.path.join(data_dir, 'IXI/IXI_T1/PP_IXIT1')
        config['file_prefix'] = 'fcmnorm_brain_mni_IXI'
        config['file_suffix'] = '_T1_restore'
        config['file_ext'] = '.nii.gz'
        config[
            'path_training_csv'] =os.path.join(data_dir, f'IXI/IXI_T1/config/IXI_T1_train_split{split}.csv')
        config[
            'path_validation_csv'] = os.path.join(data_dir,f'IXI/IXI_T1/config/IXI_T1_val_split{split}.csv')
        config['path_test_csv'] = os.path.join(data_dir,'IXI/IXI_T1/config/IXI_T1_test.csv')
        config['path_info_csv'] = os.path.join(data_dir,'IXI/IXI_T1/config/IXI_T1_age.csv')

    if dataset == 'ADNI':
        # ADNI
        config['base_folder'] = os.path.join(data_dir, 'ADNI/ADNI_T1')
        config['file_prefix'] = 'fcmnorm_brain_mni_ADNI_'
        config['file_suffix'] = '_T1'
        config['file_ext'] = '.nii.gz'
        config['path_training_csv'] = os.path.join(data_dir, f'ADNI/config/GR_{ADNI_group}/{ADNI_group}_ADNI_T1_train_split{split}.csv')
        config['path_validation_csv'] = os.path.join(data_dir, f'ADNI/config/GR_{ADNI_group}/{ADNI_group}_ADNI_T1_val_split{split}.csv')
        config['path_test_csv'] = os.path.join(data_dir, f'ADNI/config/GR_{ADNI_group}/{ADNI_group}_ADNI_T1_test_{ADNI_test_selection}.csv')
        config['path_info_csv'] = os.path.join(data_dir, 'ADNI/config/ADNI_age_only.csv')

    return config
