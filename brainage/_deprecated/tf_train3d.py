from __future__ import absolute_import, division, print_function, unicode_literals

import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, ReLU

from dataset import AgeData

# todo move models to seperate file

def simple_model(input_shape=(124, 124, 124, 1), lambda_l2=0.001):
    model = Sequential()
    model.add(Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same',
                     input_shape=input_shape))
    model.add(Conv3D(8, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), activation='relu', padding='same'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    model.add(Dense(1, activation='linear'))
    return model


def simple_model2(input_shape=(124, 124, 124, 1), lambda_l2=0.001):
    model = Sequential()
    model.add(Conv3D(8, kernel_size=(3, 3, 3),  padding='same', input_shape=input_shape))
    #model.add(tfa.layers.InstanceNormalization(axis=3))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), padding='same'))
    #model.add(tfa.layers.InstanceNormalization(axis=3))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    #model.add(tfa.layers.InstanceNormalization(axis=3))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(64, kernel_size=(3, 3, 3), padding='same'))
    #model.add(tfa.layers.InstanceNormalization(axis=3))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(128, kernel_size=(3, 3, 3), padding='same'))
    #model.add(tfa.layers.InstanceNormalization(axis=3))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    model.add(Dense(1, activation='linear', bias_initializer=tf.constant_initializer(50.0)))
    return model


def simple_model3(input_shape=(124, 124, 124, 1), lambda_l2=0.001):
    model = Sequential()
    model.add(Conv3D(8, kernel_size=(3, 3, 3),  padding='same', input_shape=input_shape))
    #model.add(tfa.layers.InstanceNormalization(axis=3))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(16, kernel_size=(3, 3, 3), padding='same'))
    #model.add(tfa.layers.InstanceNormalization(axis=3))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Conv3D(32, kernel_size=(3, 3, 3), padding='same'))
    #model.add(tfa.layers.InstanceNormalization(axis=3))
    model.add(ReLU())
    model.add(MaxPooling3D(pool_size=(2, 2, 2)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(lambda_l2)))
    model.add(Dense(1, activation='linear', bias_initializer=tf.constant_initializer(50.0)))
    return model


class Trainer3d:

    def __init__(self):

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

        # General parameters
        self.run_name = 'tfconvnet'
        self.run_suffix = 'l2_low_mod3'

        # Image parameters
        image_size = [100, 120, 100]
        image_spacing = [1.5, 1.5, 1.5]

        # Parameters:
        batch_size = 8 # 16 8
        self.max_epochs = 2000
        # Adam
        learning_rate = 0.0001# 0.0001
        # - weight decay
        beta_1 = 0.9
        beta_2 = 0.999
        # l2 regularization
        lambda_l2 = 0.00005 # 0.00005

        # checkpoint directory / run_name
        model_path = '/mnt/share/raheppt1/tf_models/age/3d' + self.run_name + self.run_suffix

        # Load training and validation data.
        age_data = AgeData(image_size,
                           image_spacing,
                           shuffle_training_images=True,
                           save_debug_images=False)

        dataset_train = age_data.dataset_train()
        dataset_val = age_data.dataset_val()

        # Define training and validation datasets from generators.
        def train_gen():
            data = dataset_train
            i = 0
            while i < data.num_entries():
                sample = data.get_next()
                # DHWC tensor format
                image = sample['generators']['image'].transpose([1, 2, 3, 0])
                age = sample['generators']['age']
                yield image, age
                i += 1

        def val_gen():
            data = dataset_val
            i = 0
            while i < data.num_entries():
                sample = data.get_next()
                image = sample['generators']['image'].transpose([1, 2, 3, 0])
                age = sample['generators']['age']
                yield image, age
                i += 1

        ds_train = tf.data.Dataset.from_generator(train_gen, output_types=(tf.float32, tf.float32))
        self.ds_train = ds_train.batch(batch_size=batch_size)

        ds_val = tf.data.Dataset.from_generator(val_gen, output_types=(tf.float32, tf.float32))
        self.ds_val = ds_val.batch(batch_size=batch_size)

        # Loss function
        self.loss_object = tf.keras.losses.MeanSquaredError()
        # Optimizer.
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                                  beta_1=beta_1,
                                                  beta_2=beta_2)

        # Create an instance of the model
        self.model = simple_model3(image_size + [1], lambda_l2)

        # Define metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_mae = tf.keras.metrics.Mean(name='test_mae')

        # Checkpoint manager
        self.best_loss = tf.Variable(1, dtype=tf.float32)
        self.print_interval = 2
        self.ckpt = tf.train.Checkpoint(step=tf.Variable(1, dtype=tf.int64),
                                        optimizer=self.optimizer,
                                        net=self.model)
        self.ckpt_manager = tf.train.CheckpointManager(self.ckpt, model_path, max_to_keep=3)

        # Initialize tensorboard
        self.summary_writer = self._init_tensorboard()

    def _init_tensorboard(self):
        log_dir = './logs/tf/'
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = log_dir + '30d_' + self.run_name + self.run_suffix + '_' + current_time
        return tf.summary.create_file_writer(log_dir)

    def _save_checkpoint(self):
        save_path = self.ckpt_manager.save()
        print("Saved checkpoint for step {}: {}".format(int(self.ckpt.step), save_path))
        print("validation loss {:1.2f}".format(self.best_loss.numpy()))

    @tf.function
    def _train_step(self, images, labels):
        self.ckpt.step.assign_add(1)

        with tf.GradientTape() as tape:
            # training=True to activate dropout
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
            regularization_loss = tf.add_n(self.model.losses)
            loss = loss + regularization_loss
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        # Track metrics.
        self.train_loss(loss)

        if int(self.ckpt.step) % self.print_interval == 0:
            tf.print('.', end='')
            # Log loss to tensorboard.
            with self.summary_writer.as_default():
                tf.summary.scalar('Train/LossMSEStep', self.train_loss.result(),
                                  step=self.ckpt.step)

    @tf.function
    def _test_step(self, images, labels):
        # training=False, dropout deactivated
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)
        mae = tf.keras.losses.MAE(labels, predictions)
        # Track metrics.
        self.test_loss(t_loss)
        self.test_mae(mae)

        # If loss < best_loss, save checkpoint to file.
        if self.ckpt.step == 0:
            self.best_loss.assign(t_loss)
        if t_loss < self.best_loss:
            self.best_loss.assign(t_loss)

    def run(self):
        start_epoch = 0

        # Restore latest checkpoint.
        self.ckpt.restore(self.ckpt_manager.latest_checkpoint)
        if self.ckpt_manager.latest_checkpoint:
            start_epoch = int(str(self.ckpt_manager.latest_checkpoint).split('-')[-1])
            print("Restored from {}".format(self.ckpt_manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        # Training loop
        for epoch in range(start_epoch, self.max_epochs):
            print(epoch)
            # Reset the metrics.
            self.train_loss.reset_states()
            self.test_loss.reset_states()

            # Train.
            for images, labels in self.ds_train:
                self._train_step(images, labels)
            # Log to tensorboard.
            with self.summary_writer.as_default():
                tf.summary.scalar('Train/LossMSE', self.train_loss.result(), step=epoch)

            # Validate.
            for test_images, test_labels in self.ds_val:
                self._test_step(test_images, test_labels)

            # If current loss ist the best loss, save
            # a checkpoint.
            self._save_checkpoint()

            # Log to tensorboard.
            with self.summary_writer.as_default():
                tf.summary.scalar('Val/LossMSE', self.test_loss.result(), step=epoch)
                tf.summary.scalar('Val/MAE', self.test_mae.result(), step=epoch)

            print()
            template = 'Epoch {}, Loss: {}, Test Loss: {}, Test MAE: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.test_loss.result(),
                                  self.test_mae.result()))


def main():
    trainer = Trainer3d()
    trainer.run()


if __name__ == '__main__':
    main()
