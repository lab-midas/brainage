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
from tensorflow.keras.optimizers import Adma, SGD

from dataset import AgeData
from models.models3d import age_regression
from misc.utils import init_gpu


def create_config(save_path=None):

    config_dict = {
        # General parameters
        'run_name': 'ADNI_ADNL_quantile_01',
        'image_size': [100, 120, 100],
        'image_spacing': [1.5, 1.5, 1.5],
        # Loss
        'loss_type': 'quantile',
        'b_heteroscedastic': False,
        'quantiles': [0.5, 0.25, 0.75],
        # Training parameters
        'batch_size': 8,
        'max_epochs': 3000,
        'learning_rate': 0.0001,
        'sel_optimizer': 'Adam',
        # SGD parameters
        'sgd_momentum': 0.0,
        'sgd_nesterov': False,
        'lr_decay': True,
        'lr_decay_factor': 0.1,
        'lr_decay_start_epoch': 10,
        # ADAM parameters
        'adam_beta_1': 0.9,
        'adam_beta_2': 0.999,
        # Regularization
        'lambda_l2': 0.00005,
        'b_dropout': True,
        # Checkpoint/Tensoboard paths
        'logroot_dir': './logs/keras/',
        'checkpoint_dir': '/mnt/share/raheppt1/tf_models/age/keras',
        # Dataset paths
        # IXI
        #'base_folder': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/PP_IXIT1',
        #'file_prefix': 'fcmnorm_brain_mni_IXI',
        #'file_suffix': '_T1_restore',
        #'file_ext': '.nii.gz',
        #'path_training_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_train_split0.csv',
        #'path_validation_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_val_split0.csv',
        #'path_test_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_test.csv',
        #'path_info_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_age.csv',
        # ADNI
        'base_folder': '/mnt/share/raheppt1/project_data/brain/ADNI/ADNI_T1',
        'file_prefix': 'fcmnorm_brain_mni_ADNI_',
        'file_suffix': '_T1',
        'file_ext': '.nii.gz',
        'path_training_csv': '/mnt/share/raheppt1/project_data/brain/ADNI/config/GR_ADNL/ADNL_ADNI_T1_train_split0.csv',
        'path_validation_csv': '/mnt/share/raheppt1/project_data/brain/ADNI/config/GR_ADNL/ADNL_ADNI_T1_val_split0.csv',
        'path_test_csv': '/mnt/share/raheppt1/project_data/brain/ADNI/config/GR_NL/NL_ADNI_T1_test_split.csv ',
        'path_info_csv': '/mnt/share/raheppt1/project_data/brain/ADNI/config/ADNI_age_only.csv',

        # Data augmentation
        'default_processing': False,
        'train_intensity_shift': 0.0,
        'train_intensity_scale': 1.0,
        'train_intensity_clamp_min': 0.0,
        'train_intensity_clamp_max': 2.0,
        'train_intensity_random_scale': 0.4,
        'train_intensity_random_shift': 0.25,
        'val_intensity_shift': 0.0,
        'val_intensity_scale': 1.0,
        'val_intensity_clamp_min': 0.0,
        'val_intensity_clamp_max': 2.0,
        'augmentation_flip':        [0.5, 0.0, 0.0],
        'augmentation_scale':       [0.1, 0.1, 0.1],
        'augmentation_translation': [10.0, 10.0, 10.0],
        'augmentation_random':      [0.1, 0.1, 0.1]
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

    loss_type = config['loss_type']
    b_heteroscedastic = config['b_heteroscedastic']
    quantiles = config['quantiles']

    # Training and model parameters
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    learning_rate = config['learning_rate']
    adam_beta_1 = config['adam_beta_1']
    adam_beta_2 = config['adam_beta_2']
    shuffle_buffer_size = config['shuffle_buffer_size']

    # Regularization
    lambda_l2 = config['lambda_l2']
    b_dropout = config['b_dropout']

    # Checkpoint/Tensorboard paths
    logroot_dir = Path(config['logroot_dir'])
    checkpoint_dir = Path(config['checkpoint_dir']).joinpath(run_name)
    checkpoint_dir.mkdir(exist_ok=True)

    # Data augmentation
    # If true, default augmentation/processing parameters will be used.
    # Otherwise, they will be loaded explicitly from the config dict.
    default_processing = config['default_processing']

    # Load training and validation data.
    age_data = AgeData(config,
                       shuffle_training_images=True,
                       save_debug_images=False,
                       default_processing=default_processing)
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

    ds_train = ds_train.repeat().batch(batch_size=batch_size)

    ds_val = tf.data.Dataset.from_generator(val_gen,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape((None, None, None, None)),
                                                           tf.TensorShape((1,))))
    ds_val = ds_val.repeat().batch(batch_size=batch_size)

    # Initialize tensorboard logdir.
    logdir = logroot_dir.joinpath(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + run_name)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(logdir))

    # Save checkpoints. Early stopping, only save the best checkpoint.
    checkpoint_path = checkpoint_dir.joinpath('cp{epoch:02d}-{mae:.2f}.tf')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path),
                                                     monitor='val_mse',
                                                     save_weights_only=False,
                                                     save_best_only=True,
                                                     verbose=1)

    # Define metrics and loss.

    def mse(y_true, y_pred):
        return tf.losses.MSE(tf.squeeze(y_true[:, 0]),
                             tf.squeeze(y_pred[:, 0]))

    def mae(y_true, y_pred):
        return tf.losses.MAE(tf.squeeze(y_true[:, 0]),
                             tf.squeeze(y_pred[:, 0]))

    def get_loss():

        # Quantile loss function.
        def quantile_loss(y_true, y_pred):
            losses = []
            for i, q in enumerate(quantiles):
                f = y_pred[:, i]
                f = f[:, tf.newaxis]
                e = (y_true - f)
                loss = tf.reduce_mean(tf.maximum(q * e, (q - 1) * e), axis=-1)
                losses.append(loss)
            return tf.reduce_mean(tf.add_n(losses))

        # Neglog-likelihood loss for gaussian likelihood.
        def loss_nloglik(y_true, y_pred):
            if b_heteroscedastic:
                log_sigma2 = y_pred[:, 1]
            else:
                log_sigma2 = 0.0

            mean = y_pred[:, 0]
            nloglik = tf.reduce_mean(0.5*tf.math.exp(-log_sigma2) * (tf.squeeze(y_true) - mean) ** 2 + 0.5 * log_sigma2)
            return nloglik

        assert loss_type in ['quantile', 'negloglik']

        if loss_type == 'quantile':
            return quantile_loss
        elif loss_type == 'negloglik':
            return loss_nloglik

    # Define optimizer.
    # Learning rate decay for SGD optimizer.
    def lr_scheduler(epoch):
        if sel_optim == 'SGD':
            if lr_decay:
                if epoch < lr_decay_start_epoch:
                    return learning_rate
                else:
                    return learning_rate * tf.math.exp(lr_decay_factor * (lr_decay_start_epoch - epoch))
            else:
                return learning_rate
        else:
            return learning_rate
    
    if sel_optim == 'SGD':
        sel_optimizer = SGD(learning_rate,
                            momentum=sgd_momentum,
                            nesterov=sgd_nesterov)
    elif sel_optim == 'Adam':
        sel_optimizer = Adam(learning_rate,
                             beta_1=adam_beta_1,
                             beta_2=adam_beta_2)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)


    # Build model.

    # Network outputs
    if loss_type == 'quantile':
        n_outputs = len(quantiles)
    else:
        if b_heteroscedastic:
            n_outputs = 2
        else:
            n_outputs = 1

    simple_model = age_regression.build_simple_model(image_size + [1],
                                                     lambda_l2=lambda_l2,
                                                     dropout=b_dropout,
                                                     outputs=n_outputs)

    simple_model.compile(loss=get_loss(),
                         optimizer=sel_optimizer,
                         metrics=[mse, mae])
    print(simple_model.summary())

    # Train the model.
    simple_model.fit(ds_train,
                     epochs=max_epochs,
                     verbose=1,
                     validation_data=ds_val,
                     steps_per_epoch=train_samples // batch_size + 1,
                     validation_steps=val_samples // batch_size + 1,
                     callbacks=[tensorboard_callback,
                                cp_callback,
                                lr_callback])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train age prediction model.')
    parser.add_argument('--cfg', nargs='?')
    parser.add_argument('--gpu', type=int)
    parser.add_argument('--default', action='store_true')
    args = parser.parse_args()

    if not args.cfg:
        config = create_config(args.cfg)
    else:
        with open(args.cfg) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

    if args.default:
        # Activate default data augmentation.
        config['default_processing'] = True

    print(config)

    # Init GPU.
    gpu = args.gpu
    if not gpu:
        init_gpu(gpu_device='0')
    else:
        init_gpu(gpu_device=str(gpu))

    # Start training.
    train_model(config)
