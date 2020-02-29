import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_train.layers.layers import dense, flatten, max_pool3d, conv3d

tfd = tfp.distributions


class Network(object):
    def __init__(self):
        print('==========Base Class==========')
        self.num_features_per_level = [8, 16, 32, 64]
        self.num_features_fc = [128]

        self.dropout_per_level_fc = [0, 0]
        self.padding = 'SAME'
        self.kernel_size = [3, 3, 3]
        self.pool_strides = [2, 2, 2]

        # self.normalization = batch_norm
        # self.normalization_dense = batch_norm_dense
        self.normalization = None
        self.normalization_dense = None

        self.activation = tf.nn.relu

        self.set_layer_types()

    def set_layer_types(self):
        self.conv = conv3d
        self.pool = max_pool3d

    def build_graph(self, images, is_training):
        self.is_training = is_training

        node = self.downsampling_conv2_max_pool(images)
        node = self.fc(node, self.num_features_fc)

        return self.build_output_layer(node)

    def build_output_layer(self, node):
        return dense(node,
                     1,
                     name='prediction',
                     activation=None,
                     normalization=None,
                     is_training=self.is_training,
                     bias_initializer=tf.constant_initializer(16.0))

    def transition_layer(self, node, layer_nr):
        print('Not implemented!')
        return node

    def downsampling_conv2_max_pool(self, node):
        for current_level in range(len(self.num_features_per_level) - 1):
            num_features = self.num_features_per_level[current_level]
            node = self.conv(node,
                             num_features,
                             self.kernel_size,
                             name='conv' + str(current_level) + '_0',
                             normalization=self.normalization,
                             activation=tf.nn.relu,
                             is_training=self.is_training,
                             padding=self.padding)

            node = self.transition_layer(node, 0)
            node = self.conv(node,
                             num_features,
                             self.kernel_size,
                             name='conv' + str(current_level) + '_1',
                             normalization=self.normalization,
                             activation=tf.nn.relu,
                             is_training=self.is_training,
                             padding=self.padding)

            node = self.transition_layer(node, 1)
            node = self.pool(node, self.pool_strides, name='pool' + str(current_level))

        current_level = len(self.num_features_per_level) - 1
        num_features = self.num_features_per_level[current_level]
        node = self.conv(node,
                         num_features,
                         self.kernel_size,
                         name='conv' + str(current_level) + '_0',
                         normalization=self.normalization,
                         activation=tf.nn.relu,
                         is_training=self.is_training,
                         padding=self.padding)

        node = self.transition_layer(node, 0)
        node = self.conv(node,
                         num_features,
                         self.kernel_size,
                         name='conv' + str(current_level) + '_1',
                         normalization=self.normalization,
                         activation=tf.nn.relu,
                         is_training=self.is_training,
                         padding=self.padding)

        node = self.transition_layer(node, 1)
        return node

    def fc(self, node, num_filters_fc):
        node = flatten(node)
        for current_level in range(len(num_filters_fc)):
            dropout = self.dropout_per_level_fc[current_level]
            if dropout > 0:
                node = tf.layers.dropout(node, training=self.is_training, rate=dropout)

            num_features = num_filters_fc[current_level]
            node = dense(node,
                         num_features,
                         name='fc' + str(current_level),
                         normalization=self.normalization_dense,
                         activation=tf.nn.relu,
                         is_training=self.is_training)
        return node