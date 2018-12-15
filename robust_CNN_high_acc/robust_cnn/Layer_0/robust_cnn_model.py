# Some code was borrowed from https://github.com/petewarden/tensorflow_makefile/blob/master/tensorflow/models/image/mnist/convolutional.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.slim as slim

import math

def _dp_mult(dp_epsilon=1.0,
             dp_delta=0.01,
             attack_norm_bound=0.3,
             output_dim=None):
    '''
    Use Gaussian mechanism adding noise
    '''
    dp_eps = dp_epsilon
    dp_del = dp_delta
    
    # Use the Gaussian mechanism
    noise = attack_norm_bound * math.sqrt(2 * math.log(1.25 / dp_del)) / dp_eps
    return noise



# Create model of CNN with slim api
def CNN(inputs, is_training=True):
    batch_norm_params = {'is_training': is_training, 'decay': 0.9, 'updates_collections': None}
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        normalizer_fn=slim.batch_norm,
                        normalizer_params=batch_norm_params):
        x = tf.reshape(inputs, [-1, 28, 28, 1])
        """
        NOISE LAYER
        """
        attack_norm_bound = 0.3
        dp_mult = _dp_mult(attack_norm_bound)
        sensitivity = 1
        # print(dp_mult)
        noise_scale = dp_mult * sensitivity
        # For sensitivity_norm = 'l2'
        noise = tf.random_normal(tf.shape(x), mean=0, stddev=1)
        noise = noise_scale * noise
        net = x + noise
        
        # For slim.conv2d, default argument values are like
        # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
        # padding='SAME', activation_fn=nn.relu,
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer,
        net = slim.conv2d(net, 32, [5, 5], scope='conv1')
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        
        net = slim.conv2d(net, 64, [5, 5], scope='conv2')
        net = slim.max_pool2d(net, [2, 2], scope='pool2')
        net = slim.flatten(net, scope='flatten3')

        # For slim.fully_connected, default argument values are like
        # activation_fn = nn.relu,
        # normalizer_fn = None, normalizer_params = None, <== slim.arg_scope changes these arguments
        # weights_initializer = initializers.xavier_initializer(),
        # biases_initializer = init_ops.zeros_initializer,
        net = slim.fully_connected(net, 1024, scope='fc3')
        net = slim.dropout(net, is_training=is_training, scope='dropout3')  # 0.5 by default
        outputs = slim.fully_connected(net, 10, activation_fn=None, normalizer_fn=None, scope='fco')
    return outputs
