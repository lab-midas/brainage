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
from graz.network import Network
from tensorflow_train.train_loop import MainLoopBase
from tensorflow_train.utils.summary_handler import create_summary_placeholder
from utils.io.common import create_directories_for_file_name


class Main(MainLoopBase):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.method = self.config.method
        self.image_size = self.config.image_size
        self.image_spacing = self.config.image_spacing

        self.init_base_variables()
        self.init_changeable_variables()

    def init_base_variables(self):
        self.age_type = 'CA'
        self.output_base_folder = '/home/stefan/media/stefan_results/MultiFactorialTest/'  # TODO Modify output path
        self.folder_string = ''
        self.folder_timestamp = self.config.exp_name  # params['exp_name']
        self.output_folder = os.path.join(self.output_base_folder, self.folder_string, self.folder_timestamp)
        self.data_format = 'channels_first'

        self.base_folder = '/home/stefan/media/stefan_datasets/LBIData/'  # TODO Modify Base folder for the data sources

        self.reg_constant = 0.0005
        self.initialized = False
        self.test_initialization = True

    def init_changeable_variables(self):
        self.batch_size = self.config.batch_size
        self.max_iter = self.config.max_iter
        self.learning_rate = self.config.learning_rate

        self.test_iter = self.config.test_iter
        self.disp_iter = self.config.disp_iter
        self.snapshot_iter = self.config.snapshot_iter

        self.save_debug_images = False
        self.test_initialization = True

        self.metric_names = ['TestMAD']
        self.additional_summaries_placeholders_val = dict(
            [(name, create_summary_placeholder(name)) for name in self.metric_names])

        self.set_dataset()
        self.init_dataset()
        self.init_input_placeholders()

        self.build_graph()

        self.prediction_dict = {}
        self.prediction_dict_train = {}

    def build_graph(self):
        self.network = Network().build_graph

    def init_dataset_arguments(self):
        dataset_arguments = dict(image_size=self.image_size,
                                 image_spacing=self.image_spacing,
                                 output_folder=self.output_folder,
                                 shuffle_training_images=True,
                                 save_debug_images=self.save_debug_images,
                                 age_type=self.age_type)
        return dataset_arguments

    def init_dataset(self):
        dataset_arguments = self.init_dataset_arguments()
        self.dataset = self.dataset(**dataset_arguments)
        self.dataset_train = self.dataset.dataset_train()
        self.dataset_val = self.dataset.dataset_val()
        self.num_samples = self.dataset_val.num_entries()

    def init_input_placeholders(self):
        data_generator_entries = OrderedDict()
        data_generator_entries['image'] = [1] + self.image_size
        data_generator_entries['age'] = [1]

        self.train_queue = DataGenerator(self.dataset_train, self.coord,
                                         data_generator_entries,
                                         batch_size=self.batch_size, n_threads=8)

        self.image_train, self.age_train = self.train_queue.dequeue()
        placeholders = tensorflow_train.utils.tensorflow_util.create_placeholders(
            data_generator_entries, shape_prefix=[None])
        self.image_val = placeholders['image']
        self.age_val = placeholders['age']
        # self.image_val, self.age_val = tensorflow_train.utils.tensorflow_util.create_placeholders(
        #    data_generator_entries, shape_prefix=[None])

    def set_dataset(self):
        self.dataset = Dataset

    def build_graph(self):
        self.network = Network().build_graph

    def loss_function(self, target, prediction):
        return tf.nn.l2_loss(target - prediction) / self.batch_size

    def initNetworks(self):
        network_template = tf.make_template('image', self.network)

        # training networks
        self.prediction = network_template(self.image_train, is_training=True)
        self.loss_net_train = self.loss_function(self.age_train, self.prediction)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            if self.reg_constant > 0:
                regularization_variables = []
                for tf_var in tf.trainable_variables():
                    if 'kernel' in tf_var.name:
                        regularization_variables.append(tf.nn.l2_loss(tf_var))
                self.loss_reg = self.reg_constant * tf.add_n(regularization_variables)
            else:
                self.loss_reg = tf.constant(0, tf.float32)

        self.train_losses = OrderedDict([('loss_net', self.loss_net_train), ('loss_reg', self.loss_reg)])
        self.loss = tf.reduce_sum(list(self.train_losses.values()))

        global_step = tf.Variable(self.current_iter, trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss,
                                                                                           global_step=global_step)

        # validation network
        self.prediction_val = network_template(self.image_val, is_training=False)
        self.loss_net_val = self.loss_function(self.age_val, self.prediction_val)
        self.val_losses = OrderedDict([('loss_net', self.loss_net_val), ('loss_reg', self.loss_reg)])
        self.loss_val = tf.reduce_sum(list(self.val_losses.values()))

    def create_test_batch(self, generator, samples):
        '''
        Loads all images in the test for fast future testing.
        :param generator: Chose from where to load the images.
        :param samples: Choose number of samples to load at once
        :return: Batch of loaded images
        '''
        self.image_batch = []
        self.label_batch = []

        for i in range(samples):
            print(f'Sample {i + 1}')
            data_entry = generator.get_next()
            current_id = data_entry['id']['image_id']
            generated_values = data_entry['generators']

            self.image_batch.append(
                np.expand_dims(generated_values['image'], axis=0))
            current_age = np.expand_dims(generated_values['age'], axis=0)
            self.label_batch.append(current_age)
            self.prediction_dict[current_id] = {'label': current_age[0, 0], 'prediction': []}

            self.prediction_dict_template = self.prediction_dict.copy()
        self.initialized = True

        test_batch_dict = {}
        test_batch_dict[self.image_val] = np.concatenate(self.image_batch, axis=0)
        test_batch_dict[self.age_val] = np.concatenate(self.label_batch, axis=0)

        return test_batch_dict

    def test(self):
        print('Testing...')

        self.prediction_dict = {}

        if not self.initialized:
            self.feed_dict_val = self.create_test_batch(samples=self.num_samples, generator=self.dataset_val)
        else:
            self.prediction_dict = self.prediction_dict_template

        fetches = self.create_test_fetches()

        run_tuple = self.sess.run(fetches + self.val_loss_aggregator.get_update_ops(),
                                  feed_dict=self.feed_dict_val)

        self.prediction_dict = self.unpack_run_tuple(run_tuple)

        create_directories_for_file_name(self.output_file_for_current_iteration('prediction.csv'))
        df = pd.DataFrame.from_dict(self.prediction_dict, orient='index')
        df.to_csv(self.output_file_for_current_iteration('prediction.csv'))

        # finalize loss values
        mad = self.get_mad(self.prediction_dict)
        self.val_loss_aggregator.finalize(self.current_iter, summary_values={'TestMAD': mad})

    def create_test_fetches(self):
        '''
        Define nodes which are evaluated during testing
        :return: Tuple of test fetches
        '''
        return (self.prediction_val,)

    def get_mad(self, prediction_dict):
        return np.mean([np.abs(prediction_dict[key]['prediction'] - prediction_dict[key]['label']) for key in
                        prediction_dict.keys()])

    def unpack_run_tuple(self, run_tuple):
        '''
        Unpacks the result form the current test run.
        :param run_tuple: return of current test run
        :return: Dict which includes all unpacked values
        '''
        temp_dict = {}
        for i, (key, value) in enumerate(self.prediction_dict.items()):
            prediction = np.squeeze(run_tuple[0][i])

            current_age = value['label']
            temp_dict[key] = {'label': current_age, 'prediction': []}
            temp_dict[key]['prediction'] = np.squeeze(prediction)
        return temp_dict


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
    loop.run()