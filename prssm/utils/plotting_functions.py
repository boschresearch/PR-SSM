# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

import numpy as np
import matplotlib.pyplot as plt

matlab_colors = [[0, 0.4470, 0.7410],
                 [0.8500, 0.3250, 0.0980],
                 [0.9290, 0.6940, 0.1250],
                 [0.4940, 0.1840, 0.5560],
                 [0.4660, 0.6740, 0.1880],
                 [0.3010, 0.7450, 0.9330],
                 [0.6350, 0.0780, 0.1840]]*2

colors = matlab_colors


def plot_task(task, filename=None):
    """ Plot training and test experiments of the given task

    args:
        task: task object holding lists of training and test experiments
        filename: prefix (path + nameprefix) to save plots
    """
    train_exps = task.train_exps
    test_exps = task.test_exps
    task_name = task.name

    # Figure names
    name_train = 'Task: %s, Train' % task_name
    name_test = 'Task: %s, Test' % task_name

    if filename:
        filename_train = filename + '_t_%s_train' % task_name
        filename_test = filename + '_t_%s_test' % task_name
    else:
        filename_train = None
        filename_test = None

    plot_experiments(train_exps, name_train, filename_train)
    plot_experiments(test_exps, name_test, filename_test)


def plot_experiments(exps, name='', filename=None):
    """ Plot individual visualizations for a list of experiments

    For each experiment in the list, a visualization of system input/output
    data is shown.
    If a filename is provided, the figure is saved to disc.
    A figure name can be provided.

    args:
        exps: list of experiments
        name: string (optional), figure name prefix
        filename: string (optional), filename prefix to save figure
    """
    for ind, exp in enumerate(exps):
        # Figure name
        if name is not '':
            name_i = name + ', Exp.: %d of %d' % (ind+1, len(exps))
        else:
            name_i = 'Exp.: %d of %d' % (ind+1, len(exps))
        # Filename if applicable
        if filename is None:
            filename_i = None
        else:
            filename_i = filename + '_e_%d' % (ind)
        # Plot single experiment
        plot_experiment(exp, name_i, filename_i)


def plot_experiment(exp, name='', filename=None):
    """ Plot single experiment's input/output data

    args:
        exp: dict, experimental data and configuration
        name: string (optional), figure name
        filename: string (optional), path + filename if figure should be saved
    """

    u_dim, y_dim = exp['u_dim'], exp['y_dim']
    dt = exp['dt']
    u, y = exp['u'], exp['y']
    H = exp['H']

    u_label = exp['u_label']
    y_label = exp['y_label']

    t = dt*np.arange(H)

    fig = plt.figure(name)
    plt.gcf().clear()

    # Plot all input/output channels
    for i in range(y_dim):
        plt.subplot(u_dim+y_dim, 1, i+1)
        plt.plot(t, y[:, i], color=colors[0])
        plt.xlabel('Time t, dt = %f, H = %d' % (dt, H))
        plt.ylabel(y_label[i])
        if i == 0:
            plt.title(name)

    for i in range(u_dim):
        plt.subplot(u_dim+y_dim, 1, i + y_dim + 1)
        plt.plot(t, u[:, i], color=colors[0])
        plt.xlabel('Time t, dt = %f, H = %d' % (dt, H))
        plt.ylabel(u_label[i])

    # Save figure to file if requested
    if filename is not None:
        plt.savefig(filename + '.png')
        plt.savefig(filename + '.svg')
        plt.close(fig)
    else:
        plt.show()


def plot_predictions_experiments_comparison(M_pred, S_pred, exps, name='',
                                            filename=None):
    """ Plot visualization of predicted and ground-truth data

    Individual visualizations are plotted for each element of the lists of
    predicted and ground-truth experimental data.

    A figurename prefix can be given.
    If a filename-prefix is given, the individual plots are saved to disc.

    args:
        M_pred: list of predicted mean values. [(H, N_batch, D)] or [(H, D)]
        S_pred: list of predicted variances.[(H, N_batch, D, D)] or [(H, D, D)]
        exps: list of experimental result dicts
        name: string (optional) figure tile prefix
        filename: string (optional) path and filename if plot should be saved
    """

    assert type(M_pred) == list, 'Input should be list object'
    assert type(S_pred) == list, 'Input should be list object'
    assert type(exps) == list, 'Input should be list object'

    assert len(M_pred) == len(S_pred) == len(exps), 'Prediction and data lists should be same length (%d, %d, %d)' % (len(M_pred), len(S_pred), len(exps))

    for ind, (exp, M_traj, S_traj) in enumerate(zip(exps, M_pred, S_pred)):
        # Figure name
        if name is not '':
            name_i = name + ', Exp.: %d of %d' % (ind+1, len(exps))
        else:
            name_i = 'Exp.: %d of %d' % (ind+1, len(exps))
        if filename is None:
            filename_i = None
        else:
            filename_i = filename + '_e_%d' % (ind)
        plot_prediction_experiment_comparison(M_traj, S_traj, exp, name_i,
                                              filename_i)


def plot_prediction_experiment_comparison(m, s, exp, name='', filename=None):
    """ Plot visualization of predicted and ground-truth data

    Plot a visualization for a single prediction and ground-truth.

    A figurename prefix can be given.
    If a filename-prefix is given, the individual plots are saved to disc.

    args:
        M_pred: ndarray (H, N_batch, D) or (H, D) predicted mean value.
        S_pred: ndarray (H, N_batch, D, D) or (H, D, D) predicted variance.
        exp: dict, experimental data and configuration
        name: string (optional) figure tile prefix
        filename: string (optional) path and filename if plot should be saved
    """
    if m.ndim == 3:
        _plot_prediction_experiment_comparison_sampled(m, exp, name, filename)
        return

    if m.ndim == 2:
        _plot_prediction_experiment_comparison_gaussian(m, s, exp, name,
                                                        filename)
        return

    raise NotImplementedError('Wrong input format')


def _plot_prediction_experiment_comparison_gaussian(m, s, exp, name='',
                                                    filename=None):
    """ Plot visualization of Gaussian prediction and ground-truth data

    A figurename prefix can be given.
    If a filename-prefix is given, the individual plots are saved to disc.

    args:
        m: ndarray (H, D) predicted mean
        s: ndarray (H, D, D) predicted variance
        exp: dict, experimental data and configuration
        name: string (optional) figure tile prefix
        filename: string (optional) path and filename if plot should be saved
    """
    sigma = np.sqrt(s)

    u_dim, y_dim = exp['u_dim'], exp['y_dim']
    dt = exp['dt']
    H = exp['H']

    t = dt*np.arange(H)

    fig = plt.figure(name)
    plt.gcf().clear()

    for i in range(y_dim):
        mean = m[:, i]
        error = sigma[:, i, i]
        gt = exp['y'][:, i]

        plt.subplot(u_dim + y_dim, 1, i + 1)
        plt.plot(t, gt, label='Exp., Out %d' % i, color=colors[0])
        plt.plot(t, mean, label='Pred., Out %d' % i, color=colors[1])
        plt.fill_between(t, mean - 2*error, mean + 2*error,
                         facecolor=colors[1], alpha=0.25, edgecolor='none')
        plt.xlabel('Time t, dt = %f, H = %d' % (dt, H))
        plt.ylabel('Output')
        plt.legend()
        if name is not None and i == 0:
            plt.title(name)

    for i in range(u_dim):
        mean = m[:, i+y_dim]
        error = sigma[:, i+y_dim, i+y_dim]
        gt = exp['u'][:, i]

        plt.subplot(u_dim + y_dim, 1, i + y_dim + 1)
        plt.plot(t, gt, label='Exp., In %d' % i, color=colors[0])
        plt.plot(t, mean, label='Pred., In %d' % i, color=colors[1])
        plt.fill_between(t, mean - 2*error, mean + 2*error,
                         facecolor=colors[1], alpha=0.25, edgecolor='none')
        plt.xlabel('Time t, dt = %f, H = %d' % (dt, H))
        plt.ylabel('Input')
        plt.legend()

    if filename is not None:
        plt.savefig(filename + '.png')
        plt.savefig(filename + '.svg')
        plt.close(fig)
    else:
        plt.show()


def _plot_prediction_experiment_comparison_sampled(m, exp, name='',
                                                   filename=None):
    """ Plot visualization of predicted samples and ground-truth data

    A figurename prefix can be given.
    If a filename-prefix is given, the individual plots are saved to disc.

    args:
        m: ndarray (H, N_batch, D) samples of predicted mean value.
        exp: dict, experimental data and configuration
        name: string (optional) figure tile prefix
        filename: string (optional) path and filename if plot should be saved
    """
    u_dim, y_dim = exp['u_dim'], exp['y_dim']
    dt = exp['dt']
    H = exp['H']

    H, N_batch, D = m.shape

    # Sample mean and std
    pred_mean = np.mean(m, axis=1)  # (H, D)
    pred_std = np.std(m, axis=1)  # (H, D)

    t = dt*np.arange(H)

    fig = plt.figure(name)
    plt.gcf().clear()

    for i in range(y_dim):
        mean = pred_mean[:, i]
        error = pred_std[:, i]
        gt = exp['y'][:, i]

        plt.subplot(u_dim + y_dim, 1, i + 1)
        plt.plot(t, gt, label='Exp., Out %d' % i, color=colors[0])
        plt.plot(t, mean, label='Pred., Out %d' % i, color=colors[1])
        plt.fill_between(t, mean - 2*error, mean + 2*error,
                         facecolor=colors[1], alpha=0.25, edgecolor='none')
        for j in range(N_batch):
            plt.plot(t, m[:, j, i], color=3*[0.2])
        plt.xlabel('Time t, dt = %f, H = %d' % (dt, H))
        plt.ylabel('Output')
        plt.legend()
        if name is not None and i == 0:
            plt.title(name)

    for i in range(u_dim):
        mean = pred_mean[:, i+y_dim]
        error = pred_std[:, i+y_dim]
        gt = exp['u'][:, i]

        plt.subplot(u_dim + y_dim, 1, i + y_dim + 1)
        plt.plot(t, gt, label='Exp., In %d' % i, color=colors[0])
        plt.plot(t, mean, label='Pred., In %d' % i, color=colors[1])
        plt.fill_between(t, mean - 2*error, mean + 2*error,
                         facecolor=colors[1], alpha=0.25, edgecolor='none')
        for j in range(N_batch):
            plt.plot(t, m[:, j, i+y_dim], color=3*[0.2])
        plt.xlabel('Time t, dt = %f, H = %d' % (dt, H))
        plt.ylabel('Input')
        plt.legend()

    if filename is not None:
        plt.savefig(filename + '.png')
        plt.savefig(filename + '.svg')
        plt.close(fig)
    else:
        plt.show()
