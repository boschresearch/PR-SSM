# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

import abc
import os
import numpy as np

from prssm.utils.data_management import generate_experiment_from_data
from prssm.utils.utils import Configurable
from prssm.utils.utils import enforce_list
from prssm.utils.utils import enforce_2d
from prssm.utils.utils import resample

"""
The following class is derived from RGP
  (https://github.com/zhenwendai/RGP)
Copyright (c) 2015, Zhenwen Dai, licensed under the BSD 3-clause license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""

class AutoregTask(Configurable):
    __metaclass__ = abc.ABCMeta

    def __init__(self, datapath=None):
        self.datapath = datapath or os.path.join(os.path.dirname(__file__),
                                                 '../datasets')

        # These default/empty task parameters are computed when load_data() is
        # called
        self.dt = 1        # System sampling timestep
        self.Dy = 0        # System output dimensionality
        self.Du = 0        # System input dimensionality
        self.D = 0         # System data (input, output) dimensionality
        self.N_train = 0   # Number of training datasets
        self.N_test = 0    # Number of test datasets
        self.H_train = []  # List of lengths of training datasets
        self.H_test = []   # List of lengths of test datasets

        # Names and units of system inputs/outputs (mainly for debug/plot)
        self.output_names = []
        self.output_units = []
        self.input_names = []
        self.input_units = []

        # Data can be resamples (factor > 1 = upsampling, factor < 1 = downsampling)
        self.resample = False
        self.resample_factor = 1.0

        # List of experimental rollouts used for training and testing
        self.train_exps = []
        self.test_exps = []

    def _data_rectification(self):
        """ Enforces standard format for data loaded by _load_data()
        """
        # Enforce lists of I/O sequences for test and training
        self.data_in_train = enforce_list(self.data_in_train)
        self.data_out_train = enforce_list(self.data_out_train)
        self.data_in_test = enforce_list(self.data_in_test)
        self.data_out_test = enforce_list(self.data_out_test)

        # Each list element should be either None or 2D numpy array
        self.data_in_train = enforce_2d(self.data_in_train)
        self.data_out_train = enforce_2d(self.data_out_train)
        self.data_in_test = enforce_2d(self.data_in_test)
        self.data_out_test = enforce_2d(self.data_out_test)

        self.data_in_train, self.data_out_train = self._replace_none(self.data_in_train, self.data_out_train)
        self.data_in_test, self.data_out_test = self._replace_none(self.data_in_test, self.data_out_test)

    def _data_resampling(self):
        if self.resample is True and self.resample_factor != 1.0:
            self.data_in_train = [resample(data, self.resample_factor) for data in self.data_in_train]
            self.data_out_train = [resample(data, self.resample_factor) for data in self.data_out_train]
            self.data_in_test = [resample(data, self.resample_factor) for data in self.data_in_test]
            self.data_out_test = [resample(data, self.resample_factor) for data in self.data_out_test]

    def _replace_none(self, data1, data2):
        for a, b in zip(data1, data2):
            if a is None and b is not None:
                a = np.ones((b.shape[0], 0), dtype=b.dtype)
            if a is not None and b is None:
                b = np.ones((a.shape[0], 0), dtype=a.dtype)
        return data1, data2

    def _compute_task_parameters(self):
        # System input/output dimensionality
        if self.data_in_train[0] is not None:
            self.Du = self.data_in_train[0].shape[1]
        else:
            self.Du = 0

        if self.data_out_train[0] is not None:
            self.Dy = self.data_out_train[0].shape[1]
        else:
            self.Dy = 0

        self.D = self.Du + self.Dy

        self.N_train = len(self.data_out_train)
        self.N_test = len(self.data_out_test)

        self.H_train = [data.shape[0] for data in self.data_out_train]
        self.H_test = [data.shape[0] for data in self.data_out_test]

    def _check_channels(self, data, channels, message):
        for i, element in enumerate(data):
            if element is not None:
                if element.shape[1] != channels:
                    raise Exception('%s dataset %d: (%d x %d) but expected Du = %d' %
                                    (message, i, element.shape, channels))

    def _check_task_consistency(self):

        # Check all training/test input/output datasets are either None or
        # comply with task Du/Dy dimensionalities
        self._check_channels(self.data_in_test, self.Du, 'Test input')
        self._check_channels(self.data_out_test, self.Dy, 'Test output')
        self._check_channels(self.data_in_train, self.Du, 'Training input')
        self._check_channels(self.data_out_train, self.Dy, 'Training output')

    def load_data(self):
        # Call non-abstract load method of sub-class to load exp data
        res = self._load_data()

        # Return if sub-class couldn't load data
        if res is not True:
            return res

        self._data_rectification()

        self._data_resampling()

        # Concatenate I/O sequences to one data matrix for training and test data
        self.data_train = []
        self.data_test = []
        for data_out, data_in in zip(self.data_out_train, self.data_in_train):
            self.data_train.append(np.concatenate((data_out, data_in), axis=1))
        for data_out, data_in in zip(self.data_out_test, self.data_in_test):
            self.data_test.append(np.concatenate((data_out, data_in), axis=1))

        if not hasattr(self, 'u_label'):
            self.u_label = ['In %d' % i for i in range(self.Du)]
        if not hasattr(self, 'y_label'):
            self.y_label = ['Out %d' % i for i in range(self.Dy)]

        self._compute_task_parameters()
        self._check_task_consistency()

        # Generate training experiments
        self.train_exps = []
        for data_out, data_in in zip(self.data_out_train,
                                     self.data_in_train):
            exp = generate_experiment_from_data(dt=self.dt,
                                                y=data_out,
                                                u=data_in,
                                                u_label=self.u_label,
                                                y_label=self.y_label)
            self.train_exps.append(exp)

        # Generate test experiments
        self.test_exps = []
        for data_out, data_in in zip(self.data_out_test,
                                     self.data_in_test):
            exp = generate_experiment_from_data(dt=self.dt,
                                                y=data_out,
                                                u=data_in,
                                                u_label=self.u_label,
                                                y_label=self.y_label)
            self.test_exps.append(exp)

        return True

    @abc.abstractmethod
    def _load_data(self):
        """ Task specific load routine to gather system I/O data

        This method is implemented by the task sub-class and required to set
        the following variables:
             data_in_train: system input for training (list, ndarray, None)
             data_out_train: system output for training (list, ndarray, None)
             data_in_test: system input for test (list, ndarray, None)
             data_out_test: system output for test (list, ndarray, None)
        """
        return True
