from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import yaml
import numpy as np
import datetime
import argparse
from pathlib import Path

# Tested with tensorflow-gpu 2.0.0
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import *

from dataset import AgeData
from models.models3d import age_siamese_regression
from misc.utils import init_gpu


def create_config(save_path=None):

    config_dict = {
        # General parameters
        'run_name': 'new_siamese_03',
        'image_size': [100, 120, 100],
        'image_spacing': [1.5, 1.5, 1.5],
        # Training parameters
        'batch_size': 16,
        'max_epochs': 3000,
        'learning_rate': 0.0001,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'shuffle_buffer_size': 32,
        # Regularization
        'lambda_l2': 0.00005,
        'b_dropout': True,
        # Paths
        'logroot_dir': './logs/keras/',
        'checkpoint_dir': '/mnt/share/raheppt1/tf_models/age/keras',
        # Dataset paths
        'base_folder': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/PP_IXIT1',
        'file_prefix': 'fcmnorm_brain_mni_IXI',
        'file_suffix': '_T1_restore',
        'file_ext': '.nii.gz',
        'path_training_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_train_split1.csv',
        'path_validation_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_val_split1.csv',
        'path_test_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_test.csv',
        'path_info_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_age.csv',
    }

    if not save_path:
        save_path = './config/' + config_dict['run_name'] + '.yaml'

    with open(save_path, 'w') as f:
        yaml.dump(config_dict, f)

    return config_dict


def train_model(config):
    # General parameters
    run_name = config['run_name']
    image_size = config['image_size']
    image_spacing = config['image_spacing']

    # Training and model parameters
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    learning_rate = config['learning_rate']
    beta_1 = config['beta_1']
    beta_2 = config['beta_2']
    shuffle_buffer_size = config['shuffle_buffer_size']

    # Regularization
    lambda_l2 = config['lambda_l2']
    b_dropout = config['b_dropout']

    # Paths
    logroot_dir = Path(config['logroot_dir'])
    checkpoint_dir = Path(config['checkpoint_dir']).joinpath(run_name)
    checkpoint_dir.mkdir(exist_ok=True)

    # Load training and validation data.
    age_data = AgeData(config,
                       shuffle_training_images=True,
                       save_debug_images=False)
    dataset_train = age_data.dataset_train()
    train_samples = dataset_train.num_entries()
    dataset_val = age_data.dataset_val()
    val_samples = dataset_val.num_entries()

    # Define training and validation datasets from generators.
    def train_gen():
        data = dataset_train
        i = 0
        while i < data.num_entries():
            sample = data.get_next()
            # DHWC tensor format
            image = sample['generators']['image'].transpose([1, 2, 3, 0])
            image = image.astype('float32')
            age = sample['generators']['age']
            yield image, age
            i += 1

    def val_gen():
        data = dataset_val
        i = 0
        while i < data.num_entries():
            sample = data.get_next()
            image = sample['generators']['image'].transpose([1, 2, 3, 0])
            image = image.astype('float32')
            age = sample['generators']['age']
            yield image, age
            i += 1

    ds_train = tf.data.Dataset.from_generator(train_gen,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape((None, None, None, None)),
                                                             tf.TensorShape((1,))))

    ds_train = ds_train.repeat().shuffle(buffer_size=shuffle_buffer_size).batch(batch_size=batch_size)

    ds_val = tf.data.Dataset.from_generator(val_gen,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape((None, None, None, None)),
                                                           tf.TensorShape((1,))))
    ds_val = ds_val.repeat().shuffle(shuffle_buffer_size).batch(batch_size=batch_size)

    # Initialize tensorboard logdir.
    logdir = logroot_dir.joinpath(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + run_name)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(logdir),
                                                       histogram_freq=1)

    # Save checkpoints. Early stopping, only save the best checkpoint.
    checkpoint_path = checkpoint_dir.joinpath('cp{epoch:02d}-{mae:.2f}.tf')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path),
                                                     monitor='val_mse',
                                                     save_weights_only=False,
                                                     save_best_only=True,
                                                     verbose=1)
    # Define metrics and loss.

    def mse(y_true, y_pred):
        n_size = tf.shape(y_true)[0]
        y_true_a = y_true[:n_size // 2]
        y_true_b = y_true[n_size // 2:]
        y_true_diff = y_true_a - y_true_b

        mse_loss = tf.losses.MSE(tf.squeeze(y_pred[:, 0]),
                                 tf.squeeze(y_true_diff[:, 0]))
        tf.print(mse_loss)
        return mse_loss

    def mae(y_true, y_pred):
        n_size = tf.shape(y_true)[0]
        y_true_a = y_true[:n_size // 2]
        y_true_b = y_true[n_size // 2:]
        y_true_diff = y_true_a - y_true_b

        return tf.losses.MAE(tf.squeeze(y_pred[:, 0]),
                             tf.squeeze(y_true_diff[:, 0]))


    # Build model.

    # Network outputs
    # todo add quantile regression
    n_outputs = 1
    siamese_model = age_siamese_regression.build_siamese_model(image_size + [1],
                                                               lambda_l2=lambda_l2,
                                                               dropout=b_dropout,
                                                               outputs=n_outputs)

    siamese_model.compile(loss=mse,
                          optimizer=Adam(learning_rate,
                                         beta_1=beta_1,
                                         beta_2=beta_2),
                          metrics=[mse, mae])
    print(siamese_model.summary())

    # Train the model.
    siamese_model.fit(ds_train,
                      epochs=max_epochs,
                      verbose=1,
                      validation_data=ds_val,
                      steps_per_epoch=train_samples // batch_size + 1,
                      validation_steps=val_samples // batch_size + 1,
                      callbacks=[tensorboard_callback,
                                 cp_callback])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train age predicition model.')
    parser.add_argument('--cfg', nargs='?')
    parser.add_argument('--gpu', type=int)
    args = parser.parse_args()

    if not args.cfg:
        config = create_config(args.cfg)
    else:
        with open(args.cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    print(config)

    # Init GPU.
    gpu = args.gpu
    if not gpu:
        init_gpu(gpu_device='0')
    else:
        init_gpu(gpu_device=str(gpu))

    print(config)

    train_model(config)
