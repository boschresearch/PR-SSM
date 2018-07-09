# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

import numpy as np

import tensorflow as tf


def variable_summaries(var, name=None, vector=True):
    """
    Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    """
    with tf.name_scope(name, 'summaries', [var]):
        var = tf.convert_to_tensor(var)
        if vector:
            mean = tf.reduce_mean(var)
            tf.summary.scalar('mean', mean, collections=["stats_summaries"])
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev,
                              collections=["stats_summaries"])
            tf.summary.scalar('max', tf.reduce_max(var),
                              collections=["stats_summaries"])
            tf.summary.scalar('min', tf.reduce_min(var),
                              collections=["stats_summaries"])
            tf.summary.histogram('histogram', var,
                                 collections=["stats_summaries"])
        else:
            tf.summary.scalar('mean', tf.reduce_mean(var),
                              collections=["stats_summaries"])


def backward(y):
    assert not np.any(y <= 1e-6), 'Input to backward transformation should be greater 1e-6'
    result = np.log(np.exp(y - 1e-6) - np.ones(1))
    return np.where(y > 35, y-1e-6, result)


def tf_forward(x, name=None):
    """Forward transform from real number to
    """
    with tf.name_scope(name, 'positive_transform_forward', [x]):
        x = tf.convert_to_tensor(x)
        return tf.nn.softplus(x) + 1e-6
