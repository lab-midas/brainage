from __future__ import absolute_import, division, print_function, unicode_literals

# Disable eager execution (otherwise there are problems with prob. conv layers)
from tensorflow.python.framework.ops import disable_eager_execution, enable_eager_execution
disable_eager_execution()

import os
import sys
import numpy as np
import datetime
from pathlib import Path

# tensorflow-gpu 2.1.0
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, ReLU
from tensorflow.keras.optimizers import *
import tensorflow_probability as tfp

from dataset import AgeData
from misc.utils import init_gpu


def build_epistemic_model(input_shape=(124, 124, 124, 1)):
    model = Sequential()
    model.add(tfp.layers.Convolution3DFlipout(8,
                                              kernel_size=(3, 3, 3),  padding='same',
                                              input_shape=input_shape,
                                              activation=tf.nn.relu))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(tfp.layers.Convolution3DFlipout(16, kernel_size=(3, 3, 3), padding='same',  activation=tf.nn.relu))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(tfp.layers.Convolution3DFlipout(32, kernel_size=(3, 3, 3), padding='same', activation=tf.nn.relu))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(tfp.layers.Convolution3DFlipout(64, kernel_size=(3, 3, 3), padding='same', activation=tf.nn.relu))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(tfp.layers.Convolution3DFlipout(128, kernel_size=(3, 3, 3), padding='same',  activation=tf.nn.relu))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dense(1024,  activation=tf.nn.relu))
    model.add(Dense(512,  activation=tf.nn.relu))
    model.add(Dense(1)) # , activation='linear'
    model.add(tfp.layers.DistributionLambda(lambda t: tfp.distributions.Normal(loc=t, scale=1)))
    return model


def train_model(run_name):
    # General parameters.
    batch_size = 16
    image_size = [100, 120, 100]
    image_spacing = [1.5, 1.5, 1.5]

    # Training parameters.
    max_epochs = 3000
    learning_rate = 0.001
    lambda_l2 = 0.00005
    beta_1 = 0.9
    beta_2 = 0.999

    # Paths.
    logroot_dir = Path('./logs/keras/')
    checkpoint_dir = Path('/mnt/share/raheppt1/tf_models/age/keras')

    # Load training and validation data.
    age_data = AgeData(image_size,
                       image_spacing,
                       shuffle_training_images=True,
                       save_debug_images=False)
    dataset_train = age_data.dataset_train()
    train_samples = dataset_train.num_entries()
    dataset_val = age_data.dataset_val()
    val_samples = dataset_val.num_entries()
    print(train_samples, val_samples)

    # Define training and validation datasets from generators.
    def train_gen():
        data = dataset_train
        i = 0
        while i < data.num_entries():
            sample = data.get_next()
            # DHWC tensor format
            image = sample['generators']['image'].transpose([1, 2, 3, 0])
            image = image.astype('float32')
            age = sample['generators']['age']/100.0
            yield image, age
            i += 1

    def val_gen():
        data = dataset_val
        i = 0
        while i < data.num_entries():
            sample = data.get_next()
            image = sample['generators']['image'].transpose([1, 2, 3, 0])
            image = image.astype('float32')
            age = sample['generators']['age']/100.0
            yield image, age
            i += 1

    ds_train = tf.data.Dataset.from_generator(train_gen,
                                              output_types=(tf.float32, tf.float32),
                                              output_shapes=(tf.TensorShape((None, None, None, None)),
                                                             tf.TensorShape((1, ))))
    ds_train = ds_train.repeat().shuffle().batch(batch_size=batch_size)

    ds_val = tf.data.Dataset.from_generator(val_gen,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape((None, None, None, None)),
                                                           tf.TensorShape((1, ))))
    ds_val = ds_val.repeat().shuffle().batch(batch_size=batch_size)

    # Initialize tensorboard logdir.
    logdir = logroot_dir.joinpath(datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + run_name)
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=str(logdir))

    # Save checkpoints. Early stopping, only save the best checkpoint.
    checkpoint_path = checkpoint_dir.joinpath(run_name, 'cp.ckt')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path),
                                                     save_weights_only=True,
                                                     save_best_only=True,
                                                     verbose=1)

    # Log likelihood loss function.
    a_step = tf.Variable(0.)

    class WeightCallback(tf.keras.callbacks.Callback):

        def on_train_batch_end(self, batch, logs=None):
            tf.keras.backend.set_value(a_step, 1 / 50.0)  # 2**(16-batch)/(2**16-1))
            return

    def custom_loss(y_true, y_pred, a):
        nloglike_loss = loss_nloglike
        kl_loss = loss_kl(y_true, y_pred)
        loss = nloglike_loss + (a - 1) * kl_loss
        return loss

    def loss_nloglike(y_true, y_pred):
        log_sigma = 0
        neglog = 1/(2*tf.math.exp(log_sigma))*tf.losses.MSE(y_true, y_pred) + log_sigma
        return neglog

    def loss_kl(y_true, y_pred):
        kl_loss = tf.keras.backend.get_value(epistemic_model.losses)
        kl_loss = tf.reduce_sum(kl_loss)
        return kl_loss

    # Build model.
    epistemic_model = build_epistemic_model(image_size + [1])
    epistemic_model.compile(loss=lambda y_true, y_pred: custom_loss(y_true, y_pred, 0.0),
                            optimizer=Adam(learning_rate,
                                           beta_1=beta_1,
                                           beta_2=beta_2),
                            metrics=['mae', 'mse'])
    print(epistemic_model.summary())

    # Train the model.
    epistemic_model.fit(ds_train,
                        epochs=max_epochs,
                        verbose=1,
                        validation_data=ds_val,
                        steps_per_epoch=train_samples//batch_size,
                        validation_steps=val_samples//batch_size,
                        callbacks=[tensorboard_callback,
                                   cp_callback])


if __name__ == '__main__':
    init_gpu(gpu_device='2')
    train_model(run_name='epistemic_03')

