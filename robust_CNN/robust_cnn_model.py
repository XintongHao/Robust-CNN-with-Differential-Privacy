#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import math


def _dp_mult(dp_epsilon=1.0,
             dp_delta=0.01,
             attack_norm_bound,
             output_dim=None):
    '''
    Use Gaussian mechanism adding noise
    '''
    dp_eps = dp_epsilon
    dp_del = dp_delta
    
    # Use the Gaussian mechanism
    noise = attack_norm_bound * math.sqrt(2 * math.log(1.25 / dp_del)) / dp_eps
    return noise

def noise_layer(x, attack_norm_bound):
    """NOISE LAYER"""
    noise_scale = tf.placeholder(tf.float32, shape=(), name='noise_scale')
    # This is a factor applied to the noise layer, used to rampup the noise at the beginning of training.
    dp_mult = _dp_mult(attack_norm_bound)
    sensitivity = 1
    
    noise_scale = noise_scale * dp_mult * sensitivity
    # For sensitivity_norm = 'l2'
    noise = tf.random_normal(tf.shape(x), mean=0, stddev=1)
    noise = noise_scale * noise
    
    return x + noise


#def _maybe_add_noise_layer(x, position):
#    if position == noise_after_n_layers:
#        return self._noise_layer(x, sensitivity_norm, sensitivity_control_scheme)
#            else:
#                return x

def weight_variable(shape):
    initializer = tf.truncated_normal_initializer(dtype=tf.float32, stddev=1e-1)
    return tf.get_variable("weights", shape,initializer=initializer, dtype=tf.float32)

def bias_variable(shape):
    initializer = tf.constant_initializer(0.0)
    return tf.get_variable("biases", shape, initializer=initializer, dtype=tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

class MNISTcnn(object):
    def __init__(self, x, y, conf):
        self.x = x
        self.y = y
        self.keep_prob = tf.placeholder(tf.float32)
        
        attack_norm_bound = 0.3
        # conv1
        with tf.variable_scope('conv1'):
            W_conv1 = weight_variable([5, 5, 1, 32])
            b_conv1 = bias_variable([32])
            h_conv1 = tf.nn.relu(conv2d(self.x, W_conv1) + b_conv1)
            h_pool1 = max_pool_2x2(h_conv1)

        h_pool1 = noise_layer(h_pool1, attack_norm_bound)
        # conv2
        with tf.variable_scope('conv2'):
            W_conv2 = weight_variable([5, 5, 32, 64])
            b_conv2 = bias_variable([64])
            h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
            h_pool2 = max_pool_2x2(h_conv2)

        # fc1
        with tf.variable_scope("fc1"):
            shape = int(np.prod(h_pool2.get_shape()[1:]))
            W_fc1 = weight_variable([shape, 1024])
            b_fc1 = bias_variable([1024])
            h_pool2_flat = tf.reshape(h_pool2, [-1, shape])
            h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # fc2
        with tf.variable_scope("fc2"):
            W_fc2 = weight_variable([1024, 10])
            b_fc2 = bias_variable([10])
            y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=y_conv))
        self.pred = tf.argmax(y_conv, 1)

        self.correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(self.y,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
