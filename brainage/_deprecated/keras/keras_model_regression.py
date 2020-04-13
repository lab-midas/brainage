import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, ReLU, SpatialDropout3D
import tensorflow_probability as tfp

"""---------------------------------------------------------------------------------------------------------------------
 Deterministic CNN
---------------------------------------------------------------------------------------------------------------------"""


def build_simple_model(input_shape=(124, 124, 124, 1),
                       lambda_l2=0.00005,
                       dropout=True,
                       spatial_dropout=False,
                       dropout_rate=0.5,
                       outputs=1):
    model = Sequential()
    model.add(Conv3D(8, kernel_size=(3, 3, 3),  padding='same', input_shape=input_shape))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    if spatial_dropout:
        model.add(SpatialDropout3D(dropout_rate))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    if spatial_dropout:
        model.add(SpatialDropout3D(dropout_rate))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    if spatial_dropout:
        model.add(SpatialDropout3D(dropout_rate))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    if spatial_dropout:
        model.add(SpatialDropout3D(dropout_rate))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    if dropout:
        model.add(Dropout(dropout_rate))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    if dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(512, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    if dropout:
        model.add(Dropout(dropout_rate))
    model.add(Dense(outputs, activation='linear'))
    return model


"""---------------------------------------------------------------------------------------------------------------------
 Bayesian CNN 
---------------------------------------------------------------------------------------------------------------------"""


def build_bayesian_model(input_shape=(124, 124, 124, 1),
                         prior=None,
                         outputs=1,
                         flipout=False,
                         deterministic_conv=False):
    DenseLayer = tfp.layers.DenseReparameterization
    ConvLayer = tfp.layers.Convolution3DReparameterization
    if flipout:
        DenseLayer = tfp.layers.DenseFlipout
        ConvLayer = tfp.layers.Convolution3DFlipout

    if deterministic_conv:
        ConvLayer = Conv3D

    model = Sequential()
    model.add(tfp.layers.Convolution3DReparameterization(8,
                                                         kernel_size=(3, 3, 3),  padding='same',
                                                         input_shape=input_shape))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(ConvLayer(16, kernel_size=(3, 3, 3), padding='same', kernel_prior_fn=prior))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(ConvLayer(32, kernel_size=(3, 3, 3), padding='same', kernel_prior_fn=prior))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(ConvLayer(64, kernel_size=(3, 3, 3), padding='same', kernel_prior_fn=prior))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(ConvLayer(128, kernel_size=(3, 3, 3), padding='same', kernel_prior_fn=prior))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(DenseLayer(1024, activation='relu', kernel_prior_fn=prior))
    model.add(DenseLayer(512, activation='relu', kernel_prior_fn=prior))
    model.add(DenseLayer(outputs, activation='linear', kernel_prior_fn=prior))
    return model
