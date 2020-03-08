from __future__ import absolute_import, division, print_function, unicode_literals

# Load .env environment variables
from dotenv import load_dotenv
load_dotenv()

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import sys

import numpy as np
import datetime
import pathlib
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# tensorflow-gpu 2.0.0
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout, Conv2D
from tensorflow.keras.layers import BatchNormalization, ReLU
from tensorflow.keras.optimizers import *
import tensorflow_probability as tfp
from scipy.ndimage import gaussian_filter

from brainage.io.dataset import AgeData
from brainage.misc import utils
from brainage.misc.utils import init_gpu, add_dataset_config
from brainage.models import keras_model_regression


def load_bayesian_model(image_size, model_path,
                        outputs=1,
                        flipout=False):
    # Load bayesian model
    prior_scale = 1.0
    def prior(dtype, shape, name, trainable, add_variable_fn):
        loc = tf.zeros(shape)
        scale = tf.ones(shape) * prior_scale
        prior_dist = tfp.distributions.Normal(loc=loc, scale=scale)
        prior_dist = tfp.distributions.Independent(prior_dist,
                                                reinterpreted_batch_ndims=tf.size(prior_dist.batch_shape_tensor()))
        return prior_dist

    bayesian_model = keras_model_regression.build_bayesian_model(image_size + [1],
                                                                 prior=prior,
                                                                 flipout=flipout,
                                                                 outputs=outputs)

    checkpoint_path = Path(model_path)
    latest = tf.train.latest_checkpoint(checkpoint_path)
    print(latest)
    bayesian_model.load_weights(str(latest))
    return bayesian_model


def create_datasets(batch_size,
                    select_dataset,
                    select_group,
                    select_cv_split,
                    select_testset,
                    select_test_part,
                    augmentation=0):

    # Parameters
    config = {
            # General parameters
            'image_size': [100, 120, 100],
            'image_spacing': [1.5, 1.5, 1.5],
        }

    # Load paths
    config = add_dataset_config(config,
                                dataset=select_dataset,
                                split=select_cv_split,
                                ADNI_test_selection=select_test_part,
                                ADNI_group=select_group)

    # Load training and test data.
    age_data = AgeData(config,
                    shuffle_training_images=False,
                    save_debug_images=False,
                    select_testset=select_testset,
                    default_processing=True)
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
            age = sample['generators']['age']
            yield image, age
            i += 1

    def val_gen():
        data = dataset_val
        i = 0
        while i < data.num_entries():
            sample = data.get_next()
            image = sample['generators']['image'].transpose([1, 2, 3, 0])
            # Create blurry/random image if augmentation is activated. 
            #if augmentation==1:
            #    image = gaussian_filter(image, sigma=1.0)
            #elif augmentation==2:
            #    image = gaussian_filter(image, sigma=3.0)
            #elif augmentation==3:
            #    image = gaussian_filter(image, sigma=9.0)
            
            rd_img = np.random.random(image.shape)-0.5
            rd_img = gaussian_filter(rd_img, sigma=1.0)
            rd_img = (rd_img-np.mean(rd_img)) / np.std(rd_img)
            rd_img = rd_img / np.max(rd_img)
            rd_img2 = gaussian_filter(rd_img, sigma=6.0)
            rd_img2 = (rd_img2-np.mean(rd_img2)) / np.std(rd_img2)
            rd_imgc = rd_img*rd_img2
            rd_imgc = (rd_imgc-np.mean(rd_imgc)) / np.std(rd_imgc)

            if augmentation == 1:
                image = image*(1+0.1*rd_imgc * (image > 0))
            elif augmentation == 2:
                image = image*(1+0.2*rd_imgc * (image > 0))
            elif augmentation == 3:
                image = image*(1+0.3*rd_imgc * (image > 0))
            elif augmentation == 4:
                image = np.random.rand(*image.shape)
    
            image = image.astype('float32')
            age = sample['generators']['age']
            yield image, age
            i += 1

    ds_train = tf.data.Dataset.from_generator(train_gen,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape((None, None, None, None)),
                                                            tf.TensorShape((1, ))))
    ds_train = ds_train.batch(batch_size=batch_size)

    ds_val = tf.data.Dataset.from_generator(val_gen,
                                            output_types=(tf.float32, tf.float32),
                                            output_shapes=(tf.TensorShape((None, None, None, None)),
                                                        tf.TensorShape((1, ))))
    ds_val = ds_val.batch(batch_size=batch_size)

    return ds_train, ds_val, config


def run_evaluation(model, 
                   outputs,
                   path_csv_runs,
                   report_csv_name,
                   batch_size = 8,
                   gpu_device='0'):

    # Initialize GPU 
    init_gpu(gpu_device=gpu_device)

    # Parameters
    csv_path = pathlib.Path(__file__).absolute().parent.parent.parent.joinpath('reports', 'predictions', report_csv_name +'.csv')
    csv_path = str(csv_path)
    
    # Read csv with run configurations.
    df = pd.read_csv(path_csv_runs)
    print(df)
    
    # Do multiple forward passes and store prediction results in 
    # a dictionary.
    pred_dict = {'subject': [],
                'sample': [],
                'augmentation': [],
                'label': [],
                'name': [],}
    for k in range(outputs):
                pred_dict[f'prediction_{k}'] = []

    for run in range(df.count()[0]):
        run_dict = df.iloc[run].to_dict()

        # Log run name
        run_name = run_dict['run_name']

        # Predict in training mode? (MC Sampling)
        set_training_on = bool(run_dict['set_training_on'])
        mc_samples = int(run_dict['mc_samples'])
        # Use augmentation/disturb image?
        augmentation = int(run_dict['augmentation'])

        # 'ADNI' / 'IXI' dataset
        select_dataset = run_dict['select_dataset']
        # Select ADNI subgroups 'NL', 'AD', 'ADNL'
        select_group = run_dict['select_group']
        # Select test (True) or validation (False) set
        select_testset = bool(run_dict['select_testset'])
        # 10-fold Cross-validation, select split 0-9
        select_cv_split = int(run_dict['select_cv_split'])
        # Use small seperated testset ('split') or the 
        # whole dataset ('complete') for testing.
        select_test_part = run_dict['select_test_part']

        # Define datasets.
        ds_train, ds_val, config = create_datasets(batch_size=batch_size,
                                                select_dataset=select_dataset,
                                                select_group=select_group,
                                                select_cv_split=select_cv_split,
                                                select_testset=select_testset,
                                                select_test_part=select_test_part,
                                                augmentation=augmentation)

        # Predict
        batch_counter = 0
        for test_images, test_labels in ds_val:
            print('.', end='')
            # Multiple forward runs
            for i in range(mc_samples):
                y = model(test_images, training=set_training_on)
                y_list = [list(y.numpy()[:, k]) for k in range(outputs)]
                for k in range(outputs):
                    pred_dict[f'prediction_{k}'] += y_list[k]
                pred_dict['label'] += list(test_labels.numpy().squeeze())
                pred_dict['sample'] += len(y_list[0])*[i]
                pred_dict['name'] += len(y_list[0])*[run_name]
                pred_dict['augmentation'] += len(y_list[0])*[augmentation]
                pred_dict['subject'] += list(range(batch_counter*batch_size, 
                                                batch_counter*batch_size+len(y_list[0])))
            batch_counter += 1

    print(len(pred_dict['prediction_0']))
    print(len(pred_dict['label']))
    print(len(pred_dict['sample']))
    print(len(pred_dict['subject']))

    # Store data into csv file.
    df = pd.DataFrame.from_dict(pred_dict)
    df.to_csv(csv_path)


if __name__ == '__main__':

    # IXI quantile 01
    #model = tf.saved_model.load('/mnt/share/raheppt1/tf_models/brainage/keras/new_quantile_01/model.tf/')
    #outputs = 3
    
    # IXI simple
    # model = tf.saved_model.load('/mnt/share/raheppt1/tf_models/brainage/keras/IXI_simple/cp719-4.38.tf')
    # outputs = 1
    # ADNI simple
    # model = tf.saved_model.load('/mnt/share/raheppt1/tf_models/brainage/keras/ADNI_NL_simple/cp978-4.71.tf')
    # outputs = 1
    # IXI aleatoric (05)
    # model = tf.saved_model.load('/mnt/share/raheppt1/tf_models/brainage/keras/new_aleatoric_05/cp730-5.05.tf')
    # outputs = 2
    
    # IXI mc dropout 01 
    report_csv_name = 'test_IXI_mc_dropout_01_b'
    path_csv_runs = '/home/raheppt1/projects/age_prediction/reports/predictions/run_mcsampling.csv'
    #model = tf.saved_model.load('/mnt/share/raheppt1/tf_models/brainage/keras/IXI_mc_dropout_01/cp494-5.18.tf')
    #outputs = 1

    # Bayesian model 
    report_csv_name = 'test_IXI_bayesian_06_b'
    path_csv_runs = '/home/raheppt1/projects/age_prediction/reports/predictions/run_mcsampling.csv'
    model = load_bayesian_model(image_size=[100, 120, 100],
                                model_path='/mnt/share/raheppt1/tf_models/brainage/keras/new_bayesian_06/',
                                outputs=1,
                                flipout=False)
    outputs=1
    run_evaluation(model, outputs,
                   path_csv_runs,
                   report_csv_name)
