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

from prssm.utils.loss import loglik_loss

"""
The following classes are derived from RGP
  (https://github.com/zhenwendai/RGP)
Copyright (c) 2015, Zhenwen Dai, licensed under the BSD 3-clause license,
cf. 3rd-party-licenses.txt file in the root directory of this source tree.
"""


class Evaluation(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def evaluate(self, exps, pred):
        """Compute a scalar for access the performance"""
        return None


class RMSE(Evaluation):
    name = 'RMSE'

    def evaluate(self, exps, M_pred, S_pred):

        assert type(exps) == list
        assert type(M_pred) == list
        assert type(S_pred) == list

        assert len(exps) == len(M_pred)
        assert len(exps) == len(S_pred)

        y_ind = np.arange(exps[0]['y_dim'])

        return np.array([np.sqrt(np.square(exp['y']-pred[:,y_ind]).astype(np.float).mean()) for exp, pred in zip(exps, M_pred)])


class LogLik(Evaluation):
    name = 'LogLik'

    def evaluate(self, exps, M_pred, S_pred):

        assert type(exps) == list
        assert type(M_pred) == list
        assert type(S_pred) == list

        assert len(exps) == len(M_pred)
        assert len(exps) == len(S_pred)

        y_ind = np.arange(exps[0]['y_dim'])

        res = []
        for m_traj, s_traj, exp in zip(M_pred, S_pred, exps):
            loglik = np.array([loglik_loss(m[y_ind], s[np.ix_(y_ind,y_ind)], x, None) for m, s, x in zip(m_traj, s_traj, exp['y'])])
            res.append(np.nansum(loglik))

        return np.array(res)
