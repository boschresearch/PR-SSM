# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""


# Copyright (c) 2015, Zhenwen Dai
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import print_function

import time

from prssm.benchmarks.outputs import Result
from prssm.utils.utils import handle_exception


def broadcast_method_configs(config):
    nMethod = len(config['methods'])

    # No method configurations given
    if 'method_configs' not in config:
        return [{}] * nMethod

    # Method configuration is None
    if config['method_configs'] is None:
        return [{}] * nMethod

    # Broadcast single configuration for all methods
    if type(config['method_configs']) == dict:
        return [config['method_configs']] * nMethod

    # Check list for Nones
    if type(config['method_configs']) == list:
        return [conf if conf is not None else {} for conf in config['method_configs']]

    raise NotImplementedError


def broadcast_task_configs(config):
    nTask = len(config['tasks'])

    # No method configurations given
    if 'task_configs' not in config:
        return [{}] * nTask

    # Method configuration is None
    if config['task_configs'] is None:
        return [{}] * nTask

    # Broadcast single configuration for all methods
    if type(config['task_configs']) == dict:
        return [config['task_configs']] * nTask

    # Check list for Nones
    if type(config['task_configs']) == list:
        return [conf if conf is not None else {} for conf in config['task_configs']]

    raise NotImplementedError


def broadcast_method_task_configs(config):
    """
    returns:
        (nMethod, nTask)
    """
    nTask = len(config['tasks'])
    nMethod = len(config['methods'])

    # No method configurations given
    if 'method_task_configs' not in config:
        return [[{}] * nTask] * nMethod

    # Method configuration is None
    if config['method_task_configs'] is None:
        return [[{}] * nTask] * nMethod

    # Broadcast single configuration for all methods
    if type(config['method_task_configs']) == dict:
        return [[config['method_task_configs']] * nTask] * nMethod

    # Check list for Nones
    if type(config['method_task_configs']) == list:
        if len(config['method_task_configs']) == nTask:

            method_task_configs = [[{}] * nTask] * nMethod
            for i in range(nTask):
                if type(config['method_task_configs'][i]) == dict:
                    for j in range(nMethod):
                        method_task_configs[j][i] = config['method_task_configs'][i]

            return method_task_configs

    raise NotImplementedError


def run(config):

    nTask = len(config['tasks'])
    nMethod = len(config['methods'])
    nRepeats = int(config['repeats'])

    method_configs = broadcast_method_configs(config)
    task_configs = broadcast_task_configs(config)
    method_task_configs = broadcast_method_task_configs(config)

    results = []

    print()
    print('Setup run script')
    print('---------------------------------------------------')

    # Instantiate object for each method
    print('\t- Instantiate methods')
    methods = [method() for method in config['methods']]

    # Instantiate and initialize tasks
    print('\t- Instantiate tasks')
    tasks = [task() for task in config['tasks']]
    print('\t- Configure tasks')
    [tasks[task_i].configure(task_configs[task_i]) for task_i in range(nTask)]
    print('\t- Load task data')
    [task.load_data() for task in tasks]

    for task_i in range(nTask):
        task = tasks[task_i]

        try:
            print()
            print('Benchmarking on "%s"' % (task.name))
            print('---------------------------------------------------')

            for method_i in range(nMethod):
                method = methods[method_i]

                print('With the method "%s", %d repeats: ' %
                      (method.name, nRepeats), end='')

                try:
                    # Each task/method combi is repeated for nRepeats times
                    for rep_i in range(nRepeats):

                        # Resetn the instance of the current method
                        # (removes results from last iteration)
                        method.reset()

                        # Method configuration
                        # General configuration for this method
                        method.configure(method_configs[method_i])

                        # Task specific configuration for this method
                        method.configure(method_task_configs[method_i][task_i])

                        # Fit model to training data
                        t_st = time.time()
                        if 'validate' in config and config['validate'] == True:
                            results_opt = method.fit(task.train_exps, task.test_exps)
                        else:
                            results_opt = method.fit(task.train_exps)
                        t_fit = time.time() - t_st

                        # Predict (free simulation) on test data
                        if 'result_test_subset' in config:
                            exps = [task.test_exps[i] for i in config['result_test_subset']]
                        else:
                            exps = task.test_exps
                        t_test = time.time()
                        M_test, S_test = method.freerun(exps)
                        t_test = time.time() - t_test

                        # Predict (free simulation) on training data
                        if 'result_train_subset' in config:
                            exps = [task.train_exps[i] for i in config['result_train_subset']]
                        else:
                            exps = task.train_exps
                        t_train = time.time()
                        M_train, S_train = method.freerun(exps)
                        t_train = time.time() - t_train

                        # Add results to output writern
                        result = Result(task_i, method_i, rep_i)
                        result.add('M_train', M_train)
                        result.add('S_train', S_train)
                        result.add('M_test', M_test)
                        result.add('S_test', S_test)
                        result.add('results_opt', results_opt)
                        result.add('t_fit', t_fit)
                        result.add('t_test', t_test)
                        result.add('t_train', t_train)
                        result.add('method', method)   # this is propably not gonna work
                        result.add('task', task)
                        result.add('config', config)
                        results.append(result)

                        # If required write already intermediate results to file
                        if 'intermediate_results' in config and config['intermediate_results'] == True:
                            [out.write_result(result) for out in config['outputs']]

                        print('.', end='', flush=True)
                    print()

                except Exception as inst:
                    handle_exception(inst, config,
                                     'ERROR WHEN EXECUTING TASK: %s, METHOD: %s ##' %
                                     (task.name,
                                      method.name))

        except Exception as inst:
            handle_exception(inst, config,
                             'ERROR WHEN LOADING TASK: %s' %
                             (task.name))

    # Deconstruct method objects (e.g. close matlab engine)
    [method.close() for method in methods]
    # Output results from all tasks and methods together
    [out.write_results(results) for out in config['outputs']]
