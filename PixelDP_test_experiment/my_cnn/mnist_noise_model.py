#define a CNN model for MNIST, same model as in the tensorflow tutorial

import tensorflow as tf
import math
import numpy as np



# def weight_variable(x, shape, name):
#     filter_size = 5
#     in_filters = x.get_shape()[-1]
#     out_filters = 32
#     stride = 1

#     n = filter_size * filter_size * out_filters
#     kernel = tf.get_variable(
#             name,
#             shape,
#             tf.float32,
#             initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
#     return kernel

def weight_variable(shape, name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=name)

def bias_variable(shape, name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=name)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
def place_holders():
	x = tf.placeholder("float", shape=[None, 784])
	y_ = tf.placeholder("float", shape=[None, 10])
	return x, y_

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

# def noise_layer(x, attack_norm_bound=0.3):
#     """NOISE LAYER"""
#     # _noise_scale = tf.placeholder("float", shape=())
#     _noise_scale = tf.placeholder(tf.float32, shape=(), name='noise_scale')
#     # This is a factor applied to the noise layer, used to rampup the noise at the beginning of training.
#     dp_mult = _dp_mult(attack_norm_bound)
#     sensitivity = 1
#     print(dp_mult)
    
#     noise_scale = _noise_scale * dp_mult * sensitivity
#     # For sensitivity_norm = 'l2'
#     noise = tf.random_normal(tf.shape(x), mean=0, stddev=1)

#     noise = noise_scale * noise
#     print(x + noise)
#     return x + noise


#x is an image tensor
def model(x):

    """For sensitivity bounds when they are pre-noise
    """
    # W_conv1 = weight_variable(x, [5, 5, 1, 32], "W_conv1")
    W_conv1 = weight_variable([5, 5, 1, 32], "W_conv1")
    b_conv1 = bias_variable([32], "b_conv1")

    x_image = tf.reshape(x, [-1,28,28,1])


    # filter_size = 5
    # in_filters = x_image.get_shape()[-1]
    # out_filters = 32
    # stride = 1
    # strides = [1, stride, stride, 1]
    # sensitivity_rescaling = math.ceil(filter_size / stride)


    # n = filter_size * filter_size * out_filters
    # kernel = tf.get_variable(
    #         'DW',
    #         [filter_size, filter_size, in_filters, out_filters],
    #         tf.float32,
    #         initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))

    # k = kernel / sensitivity_rescaling

    # W_conv1 = W_conv1 / sensitivity_rescaling

    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    # h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)


    """
    NOISE LAYER
    """
    # initial_noise = tf.truncated_normal([5, 14, 14, 32], stddev=0.1)
  # return tf.Variable(initial, name=name)
    # noise_h_pool1 = tf.Variable(initial_noise, name="noise_layer_pool")
    # add_noise = noise_layer(h_pool1, 0.3)
    # noise_h_pool1 = tf.nn. relu(add_noise)
    # print(noise_h_pool1)
    attack_norm_bound = 0.3

    # _noise_scale = tf.placeholder(tf.float32, shape=(), name='noise_scale')
    # This is a factor applied to the noise layer, used to rampup the noise at the beginning of training.
    dp_mult = _dp_mult(attack_norm_bound)
    sensitivity = 1
    # print(dp_mult)
    
    noise_scale = dp_mult * sensitivity
    # For sensitivity_norm = 'l2'
    noise = tf.random_normal(tf.shape(h_pool1), mean=0, stddev=1)

    noise = noise_scale * noise
    noise_h_pool1 = h_pool1 + noise

    
    # W_conv2 = weight_variable(h_pool1, [5, 5, 32, 64], "W_conv2")
    W_conv2 = weight_variable([5, 5, 32, 64], "W_conv2")
    b_conv2 = bias_variable([64], "b_conv2")

    h_conv2 = tf.nn.relu(conv2d(noise_h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # W_fc1 = weight_variable(h_pool2, [7 * 7 * 64, 1024], "W_fc1")
    W_fc1 = weight_variable([7 * 7 * 64, 1024], "W_fc1")
    b_fc1 = bias_variable([1024], "b_fc1")

    h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # W_fc2 = weight_variable(h_fc1_drop, [1024, 10], "W_fc2")
    W_fc2 = weight_variable([1024, 10], "W_fc2")
    b_fc2 = bias_variable([10], "b_fc2")

    y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    variable_dict = {"W_conv1":W_conv1, "b_conv1":b_conv1, 
    # "noise_scale": noise_var,
    "W_conv2": W_conv2, "b_conv2": b_conv2, 
    "W_fc1": W_fc1, "b_fc1": b_fc1,
     "W_fc2":W_fc2, "b_fc2":b_fc2}
    return y_conv, keep_prob, variable_dict
