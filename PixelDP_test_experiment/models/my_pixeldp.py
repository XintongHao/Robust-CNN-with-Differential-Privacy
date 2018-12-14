#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 21:24:02 2018
code source: https://github.com/columbia/pixeldp/tree/master/models
@author: xin
"""

import tensorflow as tf
import numpy as np
import math

class Model(object):
    """My Pixel CNN model"""
    
    def __init__(self, hps, images, labels, mode):
        """ Model Constructor.
        
        Args: 
            hps: Hyperparameters.
            imgaes: Batches of images. [batch_size, image_size, image_size, 3]
            labels: Batches of labels. [batch_size, num_classes]
            mode: One of 'train' and 'eval'.
        """
        self.hps = hps
        
        if len(self.hps.layer_sensitivity_bounds) == 1 and self.hps.noise_after_n_layers > 1:
            self.layer_sensitivity_bounds =  \
                    self.hps.layer_sensitivity_bounds * self.hps.noise_after_n_layers
        else:
            self.layer_sensitivity_bounds = self.hps.layer_sensitivity_bounds
        
        
        self.mode = mode
        self.images = images
        self.labels = labels
        
        # DP params
        self._image_size = self.hps.image_size
        
        # Book keeping for the noise layer
        self._sensitivities = [1]
        # Extra book keeping for Parseval
        self._parseval_convs = []
        self._parseval_ws = []
        self._extra_train_ops = []
        
        # For noise layer
        self.noise_scale = tf.placeholder(tf.float32, shape=(), name='noise_scale')
        
    def build_graph(self, inputs_tensor=None, labels_tensor=None):
        """Build a whole graph for the model."""
        self.global_step = tf.contrib.framework.get_or_create_global_step()
        
        with tf.variable_scope('model_graph'):
            self._build_model(inputs_tensor, labels_tensor)
            
        if self.mode == 'train':
            self._build_train_op()
            
        self.summaries = tf.summary.merge_all()
        
    def pre_noise_sensitivity(self):
        return tf.reduce_prod(self._sensitivities)
    
    def _stride_arr(self, stride):
        return [1, stride, stride, 1]
    
    def _dp_mult(self, sensitivity_norm, output_dim=None):
        dp_eps = self.hps.dp_epsilon
        dp_del = self.hps.dp_delta
        
        # Use the Gaussian mechanism
        noise = self.hps.attack_norm_bound * math.sqrt(2 * math.log(1.25 / dp_del)) / dp_eps
        return noise
    
    def _build_parseval_update_ops(self):
        beta = self.hps.parseval_step
        
        ops = []
        for kernel in self._parseval_convs:
            shape = kernel.get_shape().as_list()
            w_t = tf.reshape(kernel, [-1, shape[-1]])
            w = tf.transpose(w_t)
            for _ in range(self.hps.parseval_loops):
                w   = (1 + beta) * w - beta * tf.matmul(w, tf.matmul(w_t, w))
                w_t = tf.transpose(w)
                
            op = tf.assign(kernel, tf.reshape(w_t, shape), validate_shape=True)
            
            ops.append(op)
            
        for _W in self._parseval_ws:
            w_t = _W
            w = tf.transpose(w_t)
            
            for _ in range(self.hps.parseval_loops):
                w   = (1 + beta) * w - beta * tf.matmul(w, tf.matmul(w_t, w))
                w_t = tf.transpose(w)
                
            op = tf.assign(_W, w_t, validate_shape=True)
            ops.append(op)
            
        return ops
    
    def _build_train_op(self):
        """Build training specific ops for the graph."""
        self.lrn_rate = tf.constant(self.hps.lrn_rate, tf.float32)
        tf.summary.scalar('learning_rate', self.lrn_rate)
        
        trainable_variables = tf.trainable_variables()
        grads = tf.gradients(self.cost, trainable_variables)
        # optimizer = mom
        optimizer = tf.train.MomentumOptimizer(self.lrn_rate, 0.9)
        
        apply_op = optimizer.apply_gradients(
                zip(grads, trainable_variables),
                global_step=self.global_step, 
                name='train_step')
        
        train_ops = [apply_op] + self._extra_train_ops
        
        previous_ops = [tf.group(*train_ops)]
        
        if len(self._parseval_convs) + len(self._parseval_ws) > 0:
            # Parseval process
            with tf.control_dependencies(previous_ops):
                parseval_update = tf.group(*self._build_parseval_update_ops())
                previous_ops = [parseval_update]
                
        with tf.control_dependencies(previous_ops):
            self.train_op = tf.no_op(name='train')
            
    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find(r'DW') > 0:
                costs.append(tf.nn.l2_loss(var))
                
        return tf.multiply(self.hps.weight_decay_rate, tf.add_n(costs))
    
    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, position=None):
        """CONVOLUTION LAYER, 
        with support for sensitivity bounds when they are pre-noise."""
        assert(strides[1] == strides[2])
        stride = strides[1]
        
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                    'DW',
                    [filter_size, filter_size, in_filters, out_filters],
                    tf.float32,
                    initializer=tf.random_normal_initializer(stddev=np.sqrt(2.0/n)))
            
            if position == None or position > self.hps.noise_after_n_layers:
                # Post noise: no sensitivity control
                return tf.nn.conv2d(x, kernel, strides, padding='SAME')
            
            sensitivity_control_scheme = self.hps.sensitivity_control_scheme
            layer_sensitivity_bound = self.layer_sensitivity_bounds[position-1]
            
            # For layer sensitivity bound = 'l2_l2' 
            self._parseval_convs.append(kernel)
            sensitivity_rescaling = math.ceil(filter_size / stride)
            k = kernel / sensitivity_rescaling
            
            # For sensitivity_control_scheme = 'bound'
            # Compute the sensitivity and keep it. Use kernel as we compensate to the reshape by using k in the conv2d.
            shape = kernel.get_shape().as_list()
            w_t = tf.reshape(kernel, [-1, shape[-1]])
            w = tf.transpose(w_t)
            sing_vals = tf.svd(w, compute_uv=False)
            self._sensitivities.append(tf.reduce_max(sing_vals))
            
            return tf.nn.conv2d(x, k, strides, padding='SAME')
        
    def _noise_layer(self, x, sensitivity_norm, sensitivity_control_scheme):
        """NOISE LAYER"""
        # This is a factor applied to the noise layer, used to rampup the noise at the beginning of training.
        dp_mult = self._dp_mult(sensitivity_norm)
        sensitivity = 1
        
        noise_scale = self.noise_scale * dp_mult * sensitivity
        # For sensitivity_norm = 'l2'
        noise = tf.random_normal(tf.shape(x), mean=0, stddev=1)
        noise = noise_scale * noise
        
        return x + noise
    
    def _maybe_add_noise_layer(self, x, sensitivity_norm, sensitivity_control_scheme, position):
       if position == self.hps.noise_after_n_layers:
           return self._noise_layer(x, sensitivity_norm, sensitivity_control_scheme)
       else:
           return x
       
    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')
    
    def _fully_connected(self, x, out_dim, sensitivity_control=None):
        """FULLY CONNECTED LAYER"""
        x = tf.reshape(x, [self.hps.batch_size * self.hps.n_draws, -1])
        w = tf.get_variable(
                'DW', 
                [x.get_shape()[1], out_dim],
                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', 
                            [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

        
    """
    CONVOLUTIONAL NEURAL NETWORK STRUCTURE
    """    
        
    def _build_model(self, inputs_tensor=None, labels_tensor=None):   
        """CORE MODEL"""
        assert(self.hps.noise_after_n_layers <= 2)
        if inputs_tensor != None:
            self.images = inputs_tensor
        if labels_tensor != None:
            self.labels = labels_tensor
        
        input_layer = self.images
        
        with tf.variable_scope('im_dup'):
            # Duplicate images to get multiple draws from the DP label distribution (each duplicate gets an independent noise draw)
            # before going through the rest of the netowrk.)
            ones = tf.ones([len(input_layer.get_shape())-1], dtype=tf.int32)
            x = tf.tile(input_layer, tf.concat([[self.hps.n_draws], ones], axis=0))
            
        x = self._maybe_add_noise_layer(x,
                                        sensitivity_norm=self.hps.sensitivity_norm,
                                        sensitivity_control_scheme=self.hps.sensitivity_control_scheme,
                                        position=0)
        with tf.variable_scope('init'):
            filter_size = 5
            in_filters = x.get_shape()[-1]
            out_filters = 32
            stride = 2
            strides = self._stride_arr(stride)
            
            x = self._conv('init_conv', 
                           x, 
                           filter_size, 
                           in_filters, 
                           out_filters, 
                           strides,
                           position=1)
        
        if not self.hps.noise_after_activation:
            x = self._relu(x, self.hps.relu_leakiness)
            
        x = self._maybe_add_noise_layer(x,
                                        sensitivity_norm=self.hps.sensitivity_norm,
                                        sensitivity_control_scheme=self.hps.sensitivity_control_scheme,
                                        position=1)           
            
        if self.hps.noise_after_activation:
            x = self._relu(x, self.hps.relu_leakiness)        

        
        x = self._conv('conv2',
                       x,
                       5,
                       out_filters,
                       64,
                       self._stride_arr(2),
                       position=2)
 
        if not self.hps.noise_after_activation:
            x = self._relu(x, self.hps.relu_leakiness)

        x = self._maybe_add_noise_layer(x,
                                        sensitivity_norm=self.hps.sensitivity_norm,
                                        sensitivity_control_scheme=self.hps.sensitivity_control_scheme,
                                        position=2) 

        if self.hps.noise_after_activation:
            x = self._relu(x, self.hps.relu_leakiness)  
             
        with tf.variable_scope('dense'):
            x = self._fully_connected(x, 1024)
            x = self._relu(x, self.hps.relu_leakiness)
            
        with tf.variable_scope('logit'):
            self.logits = self._fully_connected(x, self.hps.num_classes)
            self.predictions = tf.nn.softmax(self.logits)
            
        with tf.variable_scope('label_dup'):
            ones = tf.ones([len(self.labels.get_shape())-1], dtype=tf.int32)
            labels = tf.tile(self.labels, tf.concat([[self.hps.n_draws], ones], axis=0))
            
        with tf.variable_scope('costs'):
            xent = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits,
                                                           labels=labels)
            self.cost = tf.reduce_mean(xent, name='xent')
            self.cost += self._decay()
            
            tf.summary.scalar('cost', self.cost)
