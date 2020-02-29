import sys
sys.path.insert(0,'/mnt/home/raheppt1/projects/MedicalDataAugmentationTool')

import os
from collections import OrderedDict

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow_train.utils.tensorflow_util
from tensorflow_train.data_generator import DataGenerator
from dataset import Dataset
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.summary_handler import create_summary_placeholder
from utils.io.common import create_directories_for_file_name

class Main():
    def __init__(self, config):
        self.config = config
        self.method = self.config.method
        self.image_size = self.config.image_size
        self.image_spacing = self.config.image_spacing

        self.init_base_variables()

        self.batch_size = self.config.batch_size
        self.max_iter = self.config.max_iter
        self.learning_rate = self.config.learning_rate

        self.test_iter = self.config.test_iter
        self.disp_iter = self.config.disp_iter
        self.snapshot_iter = self.config.snapshot_iter

        self.save_debug_images = False
        self.test_initialization = True

        self.metric_names = ['TestMAD']
        #self.additional_summaries_placeholders_val = dict(
        #    [(name, create_summary_placeholder(name)) for name in self.metric_names])

        self.set_dataset()
        self.init_dataset()

    def init_base_variables(self):
        self.age_type = 'CA'
        self.output_base_folder = '/home/stefan/media/stefan_results/MultiFactorialTest/'  # TODO Modify output path
        self.folder_string = ''
        self.folder_timestamp = self.config.exp_name  # params['exp_name']
        self.output_folder = os.path.join(self.output_base_folder, self.folder_string, self.folder_timestamp)
        self.data_format = 'channels_first'

        self.base_folder = './'  # TODO Modify Base folder for the data sources

        self.reg_constant = 0.0005
        self.initialized = False
        self.test_initialization = True

    def init_dataset_arguments(self):
        dataset_arguments = dict(image_size=self.image_size,
                                 image_spacing=self.image_spacing,
                                 output_folder=self.output_folder,
                                 shuffle_training_images=True,
                                 save_debug_images=self.save_debug_images)
        return dataset_arguments

    def set_dataset(self):
        self.dataset = Dataset

    def init_dataset(self):
        dataset_arguments = self.init_dataset_arguments()
        self.dataset = self.dataset(**dataset_arguments)
        self.dataset_train = self.dataset.dataset_train()
        self.dataset_val = self.dataset.dataset_val()
        self.num_samples = self.dataset_val.num_entries()

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

if __name__ == '__main__':
    config = AttrDict()
    config.learning_rate = 0.0001
    config.cv = 'CA'
    config.method = ''
    config.exp_name = 'exp_1'
    config.batch_size = 8
    config.max_iter = 10000
    config.image_size = [124] * 3
    config.image_spacing = [1] * 3

    config.test_iter = 100
    config.disp_iter = 10
    config.snapshot_iter = 1000

    loop = Main(config)
