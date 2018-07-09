# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

import abc
import numpy as np

from scipy import interpolate


def gLin(m, s, A, b=None):
    if A.ndim == 1:
        dim = 1
    else:
        dim = A.shape[0]

    if b is None:
        if dim == 1:
            b = 0
        else:
            b = np.zeros(dim)

    M = np.dot(A, m) + b

    S = np.dot(A, np.dot(s, A.T))
    if dim > 1:
        S = (S + S.T) / 2
    C = A.T

    return M, S, C


def resample(data, factor):
    """ Up or downsample data by a given factor
    args:
        data: ndarray, (N, D), input data to be resampled along first dimension
        factor: double, >1 = upsample, <1 = downsample
    returns:
        data: ndarray, (floor(N*factor), D) up or downsampled data
    """

    N, D = data.shape

    x = np.linspace(1, N, N)
    x_new = np.linspace(1, N, int(N * factor))

    f = interpolate.interp1d(x, data, kind='cubic', axis=0)

    return f(x_new)


def handle_exception(inst, config, text):
    if 'raise_exception' in config and config['raise_exception'] is True:
        raise
    print()
    print('---------------------------------------------------')
    print('## %s' % (text))
    print(inst)
    print('---------------------------------------------------')


class Configurable(object):

    __metaclass__ = abc.ABCMeta

    def configure(self, config):
        assert type(config) == dict, 'configure method of %s expects dict type config parameter' % (self.__class__)

        # Copy all attributes from config-dict to the class's local space
        for key, value in config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise NotImplementedError('Unknown attribute %s for %s' %
                                          (key, self.name))


def enforce_list(var):
    """ Enforces a list of elements

    If a single, non-list element is given, a list with one element is returned

    args:
        var: list or single element
    returns:
        given list or single element list holding the given var parameter
    """
    if type(var) is not list:
        return [var]
    else:
        return var


def enforce_2d(var):
    """ Enforce list of 2D numpy arrays.

    In case of 1D timeseries (H, ), a singleton dimension is added (H, 1) such
    that timeseries data becomes a column vector.

    args:
        var, list: list of np.ndarrays or Nones
    returns:
        list of np.ndarrays or Nones where each ndarrays is atleast 2D.
    """
    assert type(var) == list, 'enforce_2d expects list type input parameter'
    res = []
    for x in var:
        if x is None:
            res.append(x)
        else:
            assert type(x) == np.ndarray, 'list elements must be ndarray or None'
            if x.ndim < 2:
                res.append(x[:, None])
            else:
                res.append(x)
    return res


def retrieve_config(config, item, error):
    assert item in config, error
    return config[item]


def create_dated_directory(path):
    import time
    import os

    assert(os.path.exists(path))

    date_str = time.strftime('%y%m%d')
    time_str = time.strftime('%H%M')
    run = 0

    dir_path = os.path.join(path, date_str, time_str, 'run_%d' % run)
    path_exists = True

    while path_exists is True:
        if os.path.exists(dir_path):
            path_exists = True
            run += 1
            dir_path = os.path.join(path, date_str, time_str, 'run_%d' % run)
        else:
            os.makedirs(dir_path)
            path_exists = False

    return dir_path
