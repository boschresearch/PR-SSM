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

from prssm.utils.utils import Configurable
from prssm.utils.utils import gLin

from prssm.utils.data_management import compute_experiment_normalization


class DynamicsModel(Configurable):
    """ Base class for system dynamics models

    Provides methods
     - fit(self, exps)
     - freerun(self, exps)
    to fit and test the model on lists of experimental rollouts.

    Data normalization (preprocessing/postprocessing) is taken care of by the
    base-class. Each method (base class and derived sub-classes) can be
    configured based on the Configurable interface.

    The proposed usage is:
    model = DynamicsLearningMethodSubClass()
    model.configure(model_config_dict)
    model.fit(task.train_exps)
    M, S = model.freerun(task.test_exps)

    Each sub-class is required to implement the following abstract methods:
     - _initialize(self)
     - _fit(self, exps)
     - _freerun(self, exps)
    to provide the individual method's capabilities.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.normalize_data = True         # normalize data to zero mean, std one
        self.shift_data = True             # if True shift to zero mean, otherwise only scale
        self.preset_normalization = False  # use predefined data normalization (set_data_normalization())

        self.u_mean = None
        self.u_std = None
        self.y_mean = None
        self.y_std = None
        self.data_mean = None
        self.data_std = None

    def close(self):
        return True

    def fit(self, exps, val_exps=None):
        """ Fit the model to a list of experimental data/rollouts.

        A methode should be configured by calling model.configure(config)
        before calling the fit() method.

        If preprocessing is enabled fit() will employ the list of experiments
        as preprocessing training dataset (e.g. for data normalization)
        """
        exps_prep = self._preprocess_experiments(exps, train=True)
        if val_exps is not None and type(val_exps) == list:
            val_exps_prep = self._preprocess_experiments(val_exps)
            return self._fit(exps_prep, val_exps_prep)
        else:
            return self._fit(exps_prep, None)

    def freerun(self, exps):
        exps_prep = self._preprocess_experiments(exps)

        M, S = self._freerun(exps_prep)

        M_post, S_post = self._postprocess_results(M, S)

        return M_post, S_post

    def reset(self):
        """ Reset method to default settings.

        Results from prior model training is deleted.
        """
        pass

    def set_data_normalization(self, u_mean, u_std, y_mean, y_std, data_mean, data_std):
        self.u_mean = u_mean
        self.u_std = u_std

        self.y_mean = y_mean
        self.y_std = y_std

        self.data_mean = data_mean
        self.data_std = data_std

        self.preset_normalization = True

    def _reverse_trans(self, M, S):
        """
        """
        A = np.diag(self.data_std)
        b = self.data_mean

        M_out = []
        S_out = []
        for m, s in zip(M, S):
            M_unnorm, S_unnorm, _ = gLin(m, s, A, b)
            M_out.append(M_unnorm)
            S_out.append(S_unnorm)

        return np.asarray(M_out), np.asarray(S_out)

    def _postprocess_results(self, M, S):
        """ Apply data postprocessing on free simulation model rollout
        args:
            M: list, list of ndarrays (H, Dy+Du)
            S: list, list of ndarrays (H, Dy+Du, Dy+Du)
        returns:
            M_post, S_post list of postprocessed free simulation results
        """
        if self.normalize_data:
            M_post = []
            S_post = []
            for m, s in zip(M, S):
                M_out, S_out = self._reverse_trans(m, s)
                M_post.append(M_out)
                S_post.append(S_out)
        else:
            M_post, S_post = M, S
        return M_post, S_post

    def _preprocess_experiments(self, exps, train=False):
        """ Apply preprocessing to experimental data

        If data normalization is enabled, normalize system I/O data to zero
        mean, 1 std if train flag is set. Otherwise normalize data based on
        previously seen data.

        args:
            exps: list of experimental rollout data
            train: bool, use experiments as training for preprocessing step
        returns:
            exps_pre: list of preprocessed experiments
        """
        # Preprocessed exps are a copy of the original experiments (shallow)
        exps_pre = [exp.copy() for exp in exps]

        # Compute training data mean and std for data normalization
        if train is True and self.normalize_data is True and self.preset_normalization is False:
            self.u_mean, self.u_std, self.y_mean, self.y_std, self.data_mean, self.data_std = compute_experiment_normalization(exps_pre)

        # Normalize experimenal data
        if self.normalize_data:
            if self.shift_data:
                for exp in exps_pre:
                    exp['u'] = (exp['u'] - self.u_mean) / self.u_std
                    exp['y'] = (exp['y'] - self.y_mean) / self.y_std
                    exp['data'] = (exp['data'] - self.data_mean) / self.data_std
            else:
                for exp in exps_pre:
                    exp['u'] = exp['u'] / self.u_std
                    exp['y'] = exp['y'] / self.y_std
                    exp['data'] = exp['data'] / self.data_std

        return exps_pre

    @abc.abstractmethod
    def _freerun(self, exps):
        """Free model prediction on the rollout data"""
        return None, None

    @abc.abstractmethod
    def _fit(self, exps, val_exps=None):
        """Fit the model to the rollout data. Return True if successful"""
        return True
