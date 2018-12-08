#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 19:02:08 2018

@author: xin
"""

import time
import six
import sys
import os
import json, math

from models import train
from models import evaluate
from datasets import mnist
import numpy as np
import models.params
#from models import pixeldp_cnn, pixeldp_resnet, madry
from models import my_pixeldp

import tensorflow as tf
#import plots.plot_robust_accuracy
#import plots.plot_accuracy_under_attack
#import plots.plot_robust_precision_under_attack

import attacks
from attacks import train_attack, evaluate_attack, pgd, carlini, params, carlini_robust_precision, evaluate_attack_carlini_robust_prec

from flags import FLAGS

def run_one():
    # Manual runs support cpu or 1 gpu
    if FLAGS.num_gpus == 0:
        dev = '/cpu:0'
    else:
        dev = '/gpu:0'

        steps_num       = 40000
        eval_data_size  = 10000
        
        image_size      = 28
        n_channels      = 1
        num_classes     = 10
        relu_leakiness  = 0.0
        lrn_rate        = 0.01
        lrn_rte_changes = [30000]
        lrn_rte_vals    = [0.01]
        
        batch_size = 128
        n_draws    = 1


    compute_robustness = True

    # See doc in ./models/params.py
    L = 0.1
    hps = models.params.HParams(
            name_prefix="",
            batch_size=batch_size,
            num_classes=num_classes,
            image_size=image_size,
            n_channels=n_channels,
            lrn_rate=lrn_rate,
            lrn_rte_changes=lrn_rte_changes,
            lrn_rte_vals=lrn_rte_vals,
            num_residual_units=4,
            use_bottleneck=False,
            weight_decay_rate=0.0002,
            relu_leakiness=relu_leakiness,
            optimizer='mom',
            image_standardization=False,
            n_draws=n_draws,
            dp_epsilon=1.0,
            dp_delta=0.05,
            robustness_confidence_proba=0.05,
            attack_norm_bound=L,
            attack_norm='l2',
            sensitivity_norm='l2',
            sensitivity_control_scheme='bound',  # bound or optimize
            noise_after_n_layers=1,
            layer_sensitivity_bounds=['l2_l2'],
            noise_after_activation=True,
            parseval_loops=10,
            parseval_step=0.0003,
            steps_num=steps_num,
            eval_data_size=eval_data_size,
    )

    #  atk = pgd
#    atk = carlini
    atk = carlini_robust_precision
    if atk == carlini_robust_precision:
        attack_params = attacks.params.AttackParamsPrec(
            restarts=1,
            n_draws_attack=20,
            n_draws_eval=500,
            attack_norm='l2',
            max_attack_size=5,
            num_examples=1000,
            attack_methodolody=attacks.name_from_module(atk),
            targeted=False,
            sgd_iterations=100,
            use_softmax=False,
            T=0.01
        )
    else:
        attack_params = attacks.params.AttackParams(
            restarts=1,
            n_draws_attack=20,
            n_draws_eval=500,
            attack_norm='l2',
            max_attack_size=5,
            
            num_examples=1000,
            
            attack_methodolody=attacks.name_from_module(atk),
            targeted=False,
            
            sgd_iterations=100,
                                                    
            use_softmax=True
        )
        
    _model = my_pixeldp


    train_attack.train_one(
        'mnist',
        _model,
        hps,
        atk,
        attack_params,
        dev=dev)

    tf.reset_default_graph()




def main(_):
    run_one()

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run()
