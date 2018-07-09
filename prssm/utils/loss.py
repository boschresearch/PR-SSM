# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

import numpy as np


#def lik(m, s, x):
#    n = len(m)
#    return np.exp(-0.5*np.dot((x-m).T, np.linalg.solve(s,(x-m)))) / np.sqrt((2*np.pi)**n * np.linalg.det(s))
#

def loglik_loss(m, s, x, cost):
    D = len(m)
    J = 0
    for i in range(D):
        J = J + 0.5*np.log(2*np.pi) + 0.5*np.log(s[i,i] + 1e-8) + 0.5*(x[i]-m[i])**2/(s[i,i]+1e-8)
    return J

#
#def mean_quadratic_loss(m, s, x, cost):
#    weights = cost['weights']
#    J = np.sum((x - m)**2 * weights)
#    return J
#
#
#def expected_quadratic_loss(m, s, x, cost):
#    weights = cost['weights']
#    J = np.sum(s * weights) + np.sum((m - x)**2 * weights)
#    return J
#
#
#def quadratic_loss_dist(m, s, x, cost):
#    weights = cost['weights']
#    J = np.sum(s * weights) + np.sum((m - x)**2 * weights)
#    J_var = np.sum(2 * weights**2 * s**2) + 4 * np.sum((m - x)**2 * weights**2 * s)
#    return J, J_var
#
#
#def saturated_loss_full(m, s, x, cost):
#
#    # m: Predicted state (D, )
#    # s: Predicted variance (D, D)
#    # x: Target state (D, )
#    # cost: Cost configuration
#    #  'W': Weight matrix (D, D)
#
#    m = np.atleast_1d(m)
#    x = np.atleast_1d(x)
#    s = np.atleast_2d(s)
#
#    D = len(m)
#
#    W = cost['W'] if 'W' in cost else np.eye(D)
#
#
#    SW = np.dot(s, W);
#    iSpW = np.linalg.solve((np.eye(D)+SW).T, W.T)  # matlab: W/(np.eye(D)+SW)
#
#    # 1. Expected cost
#    L = -np.exp(-0.5 * np.dot(np.dot((m-x).T, iSpW),(m-x))) / np.sqrt(np.linalg.det(np.eye(D) + SW)) # in interval [-1,0]
#
#    # 2. Variance of cost
#    # i2SpW = W/(np.eye(D)+2*SW);
#    # r2 = np.exp(- np.dot(np.dot((m-x).T, i2SpW), (m-x)))/ np.sqrt(np.linalg.det(np.eye(D) + 2 * SW));
#    # S = r2 - L**2;
#    # S = 0 if S < 1e-12 else S # for numerical reasons
#
#    # bring cost to the interval [0,1]
#    L = 1 + L;
#
#    # return L, S
#    return L
#
#
#def saturated_loss(m, s, x, cost):
#    w = cost['w']
#    return 1 - np.exp(-(m-x)**2/(2*w**2))
#
#
#def rmse_loss(m, s, x, cost):
#    D = len(m)
#
#    J = 0
#    for i in range(D):
#        J = J + (x[i]-m[i])**2
#
#    return np.sqrt(J/D)
#
#
#def trajectory_loss(m_pred, s_pred, x, cost):
#
#    N, D = m_pred.shape
#
#    J = 0
#
#    for t in range(N):
#        J += cost['fcn'](m_pred[t], s_pred[t], x[t], cost)
#
#    return J/N
#
#
#def experiment_loss(M_pred, S_pred, exps, cost):
#
#    J = 0
#
#    for ind in exps:
#        J += trajectory_loss(M_pred[ind], S_pred[ind], exps[ind]['x'], cost)
#
#    return J/len(exps)
