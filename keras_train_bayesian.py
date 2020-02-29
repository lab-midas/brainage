from __future__ import absolute_import, division, print_function, unicode_literals

# Disable eager execution (otherwise there are problems with prob. conv layers)
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
disable_eager_execution()

import os
import sys
import yaml
import numpy as np
import datetime
import argparse
from pathlib import Path

# Tested with tensorflow-gpu 2.0.0.
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, SGD

from dataset import AgeData
from models.models3d import age_regression
from misc.utils import init_gpu


def create_config(save_path=None):

    config_dict = {
        # General parameters
        'run_name': 'new_bayesian_08',
        'image_size': [100, 120, 100],
        'image_spacing': [1.5, 1.5, 1.5],
        # Loss
        'kl_scaling': 0.001,
        'prior_scale': 1.0,
        'b_flipout': False,
        'b_heteroscedastic': False,
        # Training & model parameters
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
       
        # Checkpoint/Tensoboard paths
        'logroot_dir': './logs/keras/',
        'checkpoint_dir': '/mnt/share/raheppt1/tf_models/age/keras',
        # Dataset paths
        'base_folder': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/PP_IXIT1',
        'file_prefix': 'fcmnorm_brain_mni_IXI',
        'file_suffix': '_T1_restore',
        'file_ext': '.nii.gz',
        'path_training_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_train_split0.csv',
        'path_validation_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_val_split0.csv',
        'path_test_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_test.csv',
        'path_info_csv': '/mnt/share/raheppt1/project_data/brain/IXI/IXI_T1/config/IXI_T1_age.csv',

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

    kl_scaling = config['kl_scaling']
    b_heteroscedastic = config['b_heteroscedastic']

    # Training and model parameters
    batch_size = config['batch_size']
    max_epochs = config['max_epochs']
    learning_rate = config['learning_rate']
    lr_decay = config['lr_decay']
    lr_decay_factor = config['lr_decay_factor']
    lr_decay_start_epoch = config['lr_decay_start_epoch']
    sel_optim = config['sel_optimizer']
    sgd_momentum = config['sgd_momentum']
    sgd_nesterov = config['sgd_nesterov']
    b_flipout = config['b_flipout']
    prior_scale = config['prior_scale']
    adam_beta_1=config['adam_beta_1']
    adam_beta_2=config['adam_beta_2']
    # Paths
    logroot_dir = Path(config['logroot_dir'])
    checkpoint_dir = Path(config['checkpoint_dir'])

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
    checkpoint_path = checkpoint_dir.joinpath(run_name, 'cp{epoch:02d}-{mae:.2f}')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path),
                                                     monitor='val_mse',
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)

    # Build model.

    # Define prior distribution.
    def prior(dtype, shape, name, trainable, add_variable_fn):
        loc = tf.zeros(shape)
        scale = tf.ones(shape) * prior_scale
        prior_dist = tfp.distributions.Normal(loc=loc, scale=scale)
        prior_dist = tfp.distributions.Independent(prior_dist,
                                                   reinterpreted_batch_ndims=tf.size(prior_dist.batch_shape_tensor()))
        return prior_dist

    # Network outputs.
    if b_heteroscedastic:
        n_outputs = 2
    else:
        n_outputs = 1

    bayesian_model = age_regression.build_bayesian_model(image_size + [1],
                                                         prior=prior,
                                                         flipout=b_flipout,
                                                         outputs=n_outputs)

    # Define metrics and loss.

    def loss_nloglik(y_true, y_pred):
        if b_heteroscedastic:
            log_sigma2 = y_pred[:, 1]
        else:
            log_sigma2 = 0.0

        mean = y_pred[:, 0]
        nloglik = tf.reduce_mean(
            0.5 * tf.math.exp(-log_sigma2) * (tf.squeeze(y_true) - mean) ** 2 + 0.5 * log_sigma2)
        return nloglik

    def loss_kl(y_true, y_pred):
        kl_loss = tf.keras.backend.get_value(bayesian_model.losses)
        kl_loss = tf.reduce_sum(kl_loss)
        return kl_loss

    def bayesian_loss(y_true, y_pred):
        nloglik_loss = loss_nloglik(y_true, y_pred)
        kl_loss = loss_kl(y_true, y_pred)
        loss = nloglik_loss + (kl_scaling * 1/train_samples - 1) * kl_loss
        return loss

    def mse(y_true, y_pred):
        return tf.losses.MSE(tf.squeeze(y_true[:, 0]),
                             tf.squeeze(y_pred[:, 0]))

    def mae(y_true, y_pred):
        return tf.losses.MAE(tf.squeeze(y_true[:, 0]),
                             tf.squeeze(y_pred[:, 0]))

    # SGD optimizer with learning rate decay.
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

    # Define optimizer.
    # Learning rate decay for SGD optimizer.
    if sel_optim == 'SGD':
        sel_optimizer = SGD(learning_rate,
                            momentum=sgd_momentum,
                            nesterov=sgd_nesterov)
    elif sel_optim == 'Adam':
        sel_optimizer = Adam(learning_rate,
                             beta_1=adam_beta_1,
                             beta_2=adam_beta_2)
    lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    # Compile model.
    bayesian_model.compile(loss=bayesian_loss,
                           optimizer=sel_optimizer,
                           metrics=[mse, mae, loss_kl, loss_nloglik])
    print(bayesian_model.summary())

    # Train the model.
    bayesian_model.fit(ds_train,
                       epochs=max_epochs,
                       verbose=1,
                       validation_data=ds_val,
                       steps_per_epoch=train_samples // batch_size + 1,
                       validation_steps=val_samples // batch_size + 1,
                       callbacks=[tensorboard_callback,
                                  cp_callback,
                                  lr_callback])


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train age predicition model.')
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

    # Start the training.
    train_model(config)
