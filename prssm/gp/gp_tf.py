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

from prssm.utils.utils_tf import tf_forward
from prssm.utils.utils_tf import backward
from prssm.utils.utils_tf import variable_summaries

"""
The following snippets are derived from GPFlow V 1.0
  (https://github.com/GPflow/GPflow)
Copyright 2017 st--, Mark van der Wilk, licensed under the Apache License, Version 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""


class RBF:

    def __init__(self, input_dim, variance=None, lengthscales=None):

        with tf.name_scope('kern'):
            self.input_dim = tf.constant(int(input_dim), name='input_dim')
            with tf.name_scope('variance'):
                variance = variance or 1.0
                self.variance_unc = tf.Variable(backward(np.asarray(variance)),
                                                dtype=tf.float64,
                                                collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'hyperparameters'],
                                                name='signal_variance')
                self.variance = tf_forward(self.variance_unc)
                variable_summaries(self.variance, vector=False)

            with tf.name_scope('lengthscales'):
                lengthscales = lengthscales or 1.0
                self.lengthscales_unc = tf.Variable(backward(np.asarray(lengthscales)) * tf.ones((self.input_dim, ), dtype=tf.float64),
                                                    collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'hyperparameters'],
                                                    name='lengthscales')
                self.lengthscales = tf_forward(self.lengthscales_unc)
                variable_summaries(self.lengthscales)

    def square_dist(self, X, X2):
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, X, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, X2, transpose_b=True) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    def K(self, X, X2=None):
        return self.variance * tf.exp(-self.square_dist(X, X2) / 2)


def conditional(Xnew, X, kern, f, q_sqrt):

    # compute kernel stuff
    num_data = tf.shape(X)[0]
    Kmn = kern.K(X, Xnew)
    Kmm = kern.K(X) + tf.eye(num_data, dtype=tf.float64) * 1e-8
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    fvar = kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
    shape = tf.stack([tf.shape(f)[1], 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # D x N x N or D x N

    # another backsubstitution in the unwhitened case
    A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # D x M x N

    fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # D x N
    fvar = tf.transpose(fvar)  # N x D or N x N x D

    return fmean, fvar
