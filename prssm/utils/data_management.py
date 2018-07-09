# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

import numpy as np


def compute_experiment_normalization(exps):
    """ Compute mean and standard deviation for experimental data

    args:
        exps: list of dicts holding experimental results and meta data
    returns:
        mean and variance of input, output and concatenated input/output
    """
    U = np.concatenate([exp['u'] for exp in exps], axis=0)
    Y = np.concatenate([exp['y'] for exp in exps], axis=0)
    DATA = np.concatenate([exp['data'] for exp in exps], axis=0)

    u_mean = U.mean(axis=0)
    u_std = U.std(axis=0)
    y_mean = Y.mean(axis=0)
    y_std = Y.std(axis=0)
    data_mean = DATA.mean(axis=0)
    data_std = DATA.std(axis=0)

    u_std = np.clip(u_std, 1e-4, None)
    y_std = np.clip(y_std, 1e-4, None)
    data_std = np.clip(data_std, 1e-4, None)

    return u_mean, u_std, y_mean, y_std, data_mean, data_std


def generate_experiment_from_data(y, u, dt=1.0, window_size=None, start_ind=0, u_label=None, y_label=None):
    """
    Generates experiment dict with experimental data and meta data from given data

    args:
        y (ndarray): (N, Dy) 2D matrix with N samples of system outputs (size Dy)
        u (ndarray): (N, Du) 2D matrix with N samples of system inputs (size Du)
        dt (float): discretization of time series signal
        window_size (bool): optional, experiment only from part of data of given size
        start_ind (int): optional, first data index used for experiment
        u_label (list): optional, list of string labels for input signals
        y_label (list): optional, list of string labels for output signals
    """
    assert isinstance(y, np.ndarray), 'Output data should be a numpy matrix.'
    assert isinstance(u, np.ndarray), 'Input data should be a numpy matrix.'

    assert y.ndim==2, 'Output data should be a 2D matrix (samples x output dimensionality).'
    assert u.ndim==2, 'Input data should be a 2D matrix (samples x input dimensionality).'

    H_y, y_dim = y.shape
    H_u, u_dim = u.shape

    assert H_y == H_u, 'Input and output data must be same length (H_y=%d, H_u=%d).'%(H_y,H_u)

    H = H_y

    if window_size is not None:
        assert H >= start_ind + window_size, 'Experimental data too short for requested subtrajectory (H=%d, window=%d:%d)'%(H, start_ind, start_ind+window_size)

    data = np.concatenate((y, u), axis=1)
    if window_size is not None:
        data = data[start_ind:start_ind+window_size, :].copy()
        H_new = window_size
    else:
        data = data[start_ind:, :].copy()
        H_new = H - start_ind

    u_label = u_label or ['In %d' % i for i in range(u_dim)]
    y_label = y_label or ['Out %d' % i for i in range(y_dim)]

    exp = {}

    exp['y'] = data[:, :y_dim]
    exp['u'] = data[:, y_dim:]
    exp['data'] = data

    exp['H'] = H_new
    exp['u_dim'] = u_dim
    exp['y_dim'] = y_dim
    exp['dt'] = dt
    exp['T'] = H_new*dt

    exp['u_label'] = u_label
    exp['y_label'] = y_label

    return exp


def get_subtrajectory_indices(exps, nW=None):
    """ Start index and experiment index for all subtrajectories.

    The function returns a matrix (N, 2) holding all valid combinations of
    the experiment index and data start index for all subtrajectories of length
    nW in the experiments.

    Args:
        exps (list): list of experiments, each experiment is a dict
        nW (int): window length, default: None = no subtrajectory, use all data

    Returns:
        ndarray: (N, 2) each rows has experiment index and data start index
    """

    nE = len(exps)

    # Data-length for each experiment
    H_exp = np.array([exp['H'] for exp in exps], dtype=int)

    if nW is None:
        H_win = H_exp
    else:
        H_win = nW * np.ones_like(H_exp)

    # Number of start indices for each experiment
    T = H_exp - H_win + 1
    T[T < 0] = 0

    ind = np.zeros((np.sum(T), 2), np.int)

    current = 0
    for i in range(nE):
        ind[current:current+T[i], 0] = i
        ind[current:current+T[i], 1] = np.arange(0, T[i])
        current = current + T[i]

    return ind
