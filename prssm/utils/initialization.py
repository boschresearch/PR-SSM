# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

import numpy as np

from prssm.utils.utils import retrieve_config


def init_inducing_inputs(config_inducing_inputs, P, D, **kwargs):
    """
    Initialize GP inducing inputs

    :param method: string, any of 'kmeans', 'random', 'randomXpermutedU'
    :param P: number of inducing inputs
    :param D: dimensionality of inducing input
    :return: inducing inputs (P, D)
    """

    method = retrieve_config(config_inducing_inputs, 'method',
                             "init_inducing_inputs: Keyword 'method' required. \
                             Can be 'kmeans', 'random', 'randomXpermutedU'")

    # Cluster GP input data, cluster centers become inducing input
    if method == 'kmeans':
        assert 'X' in kwargs, 'Keyword argument X required: GP training input'
        X = kwargs['X']
        from sklearn.cluster import KMeans
        m = KMeans(n_clusters=P, n_init=50, max_iter=500)
        m.fit(X.copy())
        Z = m.cluster_centers_.copy()

    # Randomly draw inducing inputs i.i.d. from N(0,1)
    if method == 'random':
        noise = retrieve_config(config_inducing_inputs, 'noise',
                                "init_inducing_inputs: Keyword 'noise' required.")
        Z = np.sqrt(noise) * np.random.randn(P, D)

    if method == 'uniform':
        low = retrieve_config(config_inducing_inputs, 'low',
                              "init_inducing_inputs: Keyword 'low' required.")
        high = retrieve_config(config_inducing_inputs, 'high',
                              "init_inducing_inputs: Keyword 'high' required.")
        Z = np.random.uniform(low=low, high=high, size=(P, D))

    # Random inducing inputs on state and random selection of input sequence
    if method == 'randomXpermutedU':
        assert 'X' in kwargs, 'Keyword argument X required: GP training input'
        assert 'dim_u' in kwargs, 'Keyword argument U required: input dim'

        X = kwargs['X']
        dim_u = kwargs['dim_u']

        U_ind = np.random.permutation(X[:, :dim_u])[:P]
        X_ind = 3. * np.random.randn(P, D-dim_u)

        Z = np.concatenate((U_ind, X_ind), axis=1)

    return Z


def init_inducing_outputs(config_inducing_outputs, P=20, D=1, **kwargs):
    """
    Initialize gaussian distribution on GP pseudo outputs

    :param config_inducing_outputs: dict, configuration
    :param P: number of pseudo points
    :param D: output dimensionality
    :return: q_mu, ndarray, NxD means and q_var, ndarray (N, D) variances of
    GP pseudo outputs.
    """
    method = retrieve_config(config_inducing_outputs, 'method',
                             "init_inducing_outputs: Keyword 'method' required. \
                             Can be 'random', 'lstm[1-3]', 'ssm', 'model'")

    # GP output: 0 + noise
    if method == 'zero':
        noise = retrieve_config(config_inducing_outputs, 'noise',
                                "init_inducing_inputs: Keyword 'noise' required.")
        q_mu = np.sqrt(noise) * np.random.rand(P, D)

    # GP output: 1 + noise
    if method == 'one':
        noise = retrieve_config(config_inducing_outputs, 'noise',
                                "init_inducing_inputs: Keyword 'noise' required.")
        q_mu = np.ones((P, D)) + np.sqrt(noise) * np.random.rand(P, D)

    # GP output: identity + noise
    if method == 'identity':
        noise = retrieve_config(config_inducing_outputs, 'noise',
                                "init_inducing_inputs: Keyword 'noise' required.")
        'Z' in kwargs, 'Keyword argument Z required: GP pseudo inputs'
        Z = kwargs['Z']
        assert Z.shape[1] >= D, 'Output dimension larger than input dimension'
        q_mu = Z[:, :D] + np.sqrt(noise) * np.random.rand(P, D)

    # Variance of inducing outputs
    var = retrieve_config(config_inducing_outputs, 'var',
                          "init_latent: Keyword 'var' required.")
    q_var = var * np.ones((P, D))

    return q_mu, q_var
