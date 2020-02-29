import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, ReLU


def create_encoder(input_shape):
    model = Sequential()
    model.add(Conv3D(8, kernel_size=(3, 3, 3),  padding='same', input_shape=input_shape))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), padding='same'))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    return model


def create_common_dense_layers(dropout,
                               lambda_l2,
                               outputs):
    model = Sequential()
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    if dropout:
        model.add(Dropout(0.5))
    model.add(Dense(outputs, activation='linear'))
    return model


def build_siamese_model(input_shape,
                        dropout=False,
                        lambda_l2=0.0,
                        outputs=1):

    encoder = create_encoder(input_shape)
    common_dense = create_common_dense_layers(dropout, lambda_l2, outputs)

    inputs = tf.keras.layers.Input(shape=input_shape, name='img')
    batch_size = tf.shape(inputs)[0]
    x_a = inputs[:batch_size // 2]
    x_b = inputs[batch_size // 2:]
    z_a = encoder(x_a)
    z_b = encoder(x_b)
    z = tf.concat([z_a, z_b], axis=1)
    outputs = common_dense(z)

    model = tf.keras.Model(inputs=inputs, outputs=outputs,
                           name='siamese_model')
    return model
