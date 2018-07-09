# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

import os
import numpy as np
import scipy.io

from prssm.tasks.tasks import AutoregTask


class RealWorldTask(AutoregTask):

    def __init__(self):
        datapath = os.path.join(os.path.dirname(__file__),
                                '../../datasets/real_world_tasks')
        super(RealWorldTask, self).__init__(datapath)

#
# Datasets from: http://www.gaussianprocess.org/gpml/data/
#


class SarcosArm(RealWorldTask):
    """
    The data relates to an inverse dynamics problem for a seven
    degrees-of-freedom SARCOS anthropomorphic robot arm. The task is
    to map from a 21-dimensional input space (7 joint positions,
    7 joint velocities, 7 joint accelerations) to the corresponding
    7 joint torques.

    Training data: sarcos_inv, 44484x28, double array
    Test data: sarcos_inv_test, 4449x28, double array

    Data order:
    7 joint pos, 7 joint vel, 7 joint acc, 7 joint torques

    Previously used to learn the mapping from (pos, vel, acc) --> torques

    These data had previously been used in the papers
     - LWPR: An O(n) Algorithm for Incremental Real Time Learning
       in High Dimensional Space, S. Vijayakumar and S. Schaal,
       Proc ICML 2000, 1079-1086 (2000).
     - Statistical Learning for Humanoid Robots, S. Vijayakumar,
       A. D'Souza, T. Shibata, J. Conradt, S. Schaal, Autonomous
       Robot, 12(1) 55-69 (2002)
     - Incremental Online Learning in High Dimensions S. Vijayakumar,
       A. D'Souza, S. Schaal, Neural Computation 17(12) 2602-2634 (2005)
    """

    name = 'sarcosarm'

    def __init__(self):
        super(SarcosArm, self).__init__()

        # All inputs and all outputs
        # self.input_ind = list(range(21, 28))
        # self.output_ind = list(range(0, 21))

        # Predict pos and vel for single joint from all inputs
        # joint = 0
        # self.input_ind = list(range(21, 28))
        # self.output_ind = [0 + joint, 7 + joint]

        # Predict pos and vel for all joints from all inputs
        # self.input_ind = list(range(21, 28))
        # self.output_ind = list(range(0, 14))

        # Predict pos and vel for single joint from single joint input
        # joint = 0
        # self.input_ind = [21]
        # self.output_ind = [0 + joint, 7 + joint]

        # INVERSE DYNAMICS

        # Predict torque from pos, vel, acc
        # self.input_ind = list(range(0, 21))
        # self.output_ind = list(range(21, 28))

        # FORWARD DYNAMICS

        # Predict positions from torques
        self.input_ind = list(range(21, 28))
        self.output_ind = list(range(0, 7))

        self.train_ind = list(range(0, 60))
        self.test_ind = list(range(60, 66))

        # self.test_sub = 50

        self.downsample = 1

    def _load_data(self):
        # Load train data
        data = scipy.io.loadmat(os.path.join(self.datapath, 'sarcos_inv.mat'))
        data = data['sarcos_inv']
        data = data.astype(np.float64)

        H = data.shape[0]
        H_exp = 674

        data_exps = [data[ind:ind+H_exp, :] for ind in range(0, H, H_exp)]

        # Downsample original data
        self.dt = 0.01 * self.downsample
        data_exps_sub = [data[::self.downsample, :] for data in data_exps]

        self.data_in_train = []
        self.data_out_train = []

        self.data_in_test = []
        self.data_out_test = []

        for i, data in enumerate(data_exps_sub):
            # Training data
            if i in self.train_ind:
                self.data_in_train.append(data[:, self.input_ind])
                self.data_out_train.append(data[:, self.output_ind])

            # Test data
            if i in self.test_ind:
                self.data_in_test.append(data[:, self.input_ind])
                self.data_out_test.append(data[:, self.output_ind])

        # Load test data
        # data = scipy.io.loadmat(os.path.join(self.datapath, 'sarcos_inv_test.mat'))
        # data = data['sarcos_inv_test']
        # data = data.astype(np.float64)

        return True

#
# Datasets from: http://www.iau.dtu.dk/nnbook/systems.html
#


class Actuator(RealWorldTask):

    name = 'actuator'
    filename = 'actuator.mat'

    def __init__(self):
        super(Actuator, self).__init__()
        self.split_point = 512

    def _load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))

        split_point = self.split_point
        data_in = data['u'].astype(np.float64)
        data_out = data['p'].astype(np.float64)

        self.data_in_train = data_in[:split_point]
        self.data_out_train = data_out[:split_point]

        self.data_in_test = data_in[split_point:]
        self.data_out_test = data_out[split_point:]

        return True

#
# Datasets from http://homes.esat.kuleuven.be/~smc/daisy/daisydata.html
#


class Ballbeam(RealWorldTask):

    name = 'ballbeam'
    filename = 'ballbeam.dat'

    def _load_data(self):
        data = np.loadtxt(os.path.join(self.datapath, self.filename))

        split_point = 500
        data_in = data[:, 0]
        data_out = data[:, 1]

        self.data_in_train = data_in[:split_point]
        self.data_out_train = data_out[:split_point]

        self.data_in_test = data_in[split_point:]
        self.data_out_test = data_out[split_point:]

        self.dt = 0.1

        return True


class Drive(RealWorldTask):

    name = 'drive'
    filename = 'drive.mat'

    def _load_data(self):
        data = scipy.io.loadmat(os.path.join(self.datapath, self.filename))

        split_point = 250
        data_in = data['u1']
        data_out = data['z1']

        self.data_in_train = data_in[:split_point]
        self.data_out_train = data_out[:split_point]

        self.data_in_test = data_in[split_point:]
        self.data_out_test = data_out[split_point:]

        return True


class Gas_furnace(RealWorldTask):

    name = 'gas_furnace'
    filename = 'gas_furnace.csv'

    def _load_data(self):
        data = np.loadtxt(os.path.join(self.datapath, self.filename),
                          skiprows=1, delimiter=',')

        split_point = 148
        data_in = data[:, 0]
        data_out = data[:, 1]

        self.data_in_train = data_in[:split_point]
        self.data_out_train = data_out[:split_point]

        self.data_in_test = data_in[split_point:]
        self.data_out_test = data_out[split_point:]

        return True


class Dryer(RealWorldTask):

    name = 'dryer'
    filename = 'dryer.dat'

    def _load_data(self):
        data = np.loadtxt(os.path.join(self.datapath, self.filename))

        split_point = 500
        data_in = data[:, 0]
        data_out = data[:, 1]

        self.data_in_train = data_in[:split_point]
        self.data_out_train = data_out[:split_point]

        self.data_in_test = data_in[split_point:]
        self.data_out_test = data_out[split_point:]

        return True
