# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

from __future__ import print_function
import abc
import os

import numpy as np
from shutil import copy2
from scipy.io import savemat

from prssm.utils.plotting_functions import plot_predictions_experiments_comparison
from prssm.utils.plotting_functions import plot_task

from prssm.benchmarks.evaluation import RMSE
from prssm.benchmarks.evaluation import LogLik


def get_output_name(name, postfix):
    if postfix == '':
        return name
    else:
        return name + '_' + postfix


class Result(object):

    def __init__(self, task_i, method_i, rep_i):
        self.task_i = task_i
        self.method_i = method_i
        self.rep_i = rep_i

    def add(self, name, obj):
        setattr(self, name, obj)


class Output(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, outpath, prjname):
        self.outpath = outpath
        self.prjname = prjname
        self.fname = os.path.join(outpath, prjname)

    def write_results(self, results):
        """ Compute output for a list of results
        """
        for result in results:
            self.write_result(result)

    @abc.abstractmethod
    def write_result(self, result):
        """ Compute output for a single result
        """
        pass


class SaveStartScript(Output):

    def __init__(self, outpath, prjname, filepath):
        super(SaveStartScript, self).__init__(outpath, prjname)
        self.filepath = filepath
        copy2(filepath, outpath)

    def _write_result(self, result):
        pass


class SavePredictionResults(Output):
    """
    Save prediction results from all methods and tasks on the test experiments
    Data is stored as matlab matrix.
    """

    def write_result(self, result):
        task_i = result.task_i
        method_i = result.method_i
        rep_i = result.rep_i

        M_train = result.M_train
        S_train = result.S_train

        M_test = result.M_test
        S_test = result.S_test

        filename = self.fname + '_t_%d_m_%d_r_%d_pred_train' % (task_i, method_i, rep_i)
        if not (M_train is None or S_train is None):
            savemat(filename, {'M_train': M_train, 'S_train': S_train})

        filename = self.fname + '_t_%d_m_%d_r_%d_pred_test' % (task_i, method_i, rep_i)
        if not (M_test is None or S_test is None):
            savemat(filename, {'M_test': M_test, 'S_test': S_test})


class VisualOutput(Output):
    """
    VisualOutput plots training and test data of each task as well as the
    predictions of the individual methods/models on the test datasets.
    """

    def write_result(self, result):
        task_i = result.task_i
        method_i = result.method_i
        rep_i = result.rep_i

        M_train = result.M_train
        S_train = result.S_train

        M_test = result.M_test
        S_test = result.S_test

        task = result.task
        method = result.method
        config = result.config

        train_exps = task.train_exps
        test_exps = task.test_exps

        nRepeats = int(config['repeats'])

        plot_task(task, self.fname)

        name = 'Task: %s, Method: %s, Repeat: %d of %d (Training)'%(task.name,
                                                                    method.name,
                                                                    rep_i + 1,
                                                                    nRepeats)

        filename = self.fname + '_t_%d_m_%d_r_%d_train'%(task_i, method_i, rep_i)

        if not (M_train is None or S_train is None):
            plot_predictions_experiments_comparison(M_train, S_train, train_exps, name, filename)

        name = 'Task: %s, Method: %s, Repeat: %d of %d (Test)'%(task.name,
                                                                method.name,
                                                                rep_i + 1,
                                                                nRepeats)

        filename = self.fname + '_t_%d_m_%d_r_%d_test' % (task_i, method_i, rep_i)

        if not (M_test is None or S_test is None):
            plot_predictions_experiments_comparison(M_test, S_test, test_exps, name, filename)


class PrintRMSE(Output):

    def __init__(self, outpath, prjname):
        super(PrintRMSE, self).__init__(outpath, prjname)
        self.rmse = RMSE()

    def write_results(self, results):
        self._write_header_to_console()
        for result in results:
            self._write_result_to_console(result)

    def write_result(self, result):
        self._write_header_to_console()
        self._write_result_to_console(result)

    def _write_header_to_console(self):
        print()
        print('RMSE results on test data')
        print('-------------------------------------')

    def _write_result_to_console(self, result):
        task_i = result.task_i
        method_i = result.method_i
        rep_i = result.rep_i

        M_train = result.M_train
        S_train = result.S_train

        M_test = result.M_test
        S_test = result.S_test

        task = result.task
        config = result.config

        train_exps = task.train_exps
        test_exps = task.test_exps

        if not (M_train is None or S_train is None):
            error_train = self.rmse.evaluate(train_exps, M_train, S_train)
        else:
            error_train = np.array([np.nan])

        if not (M_test is None or S_test is None):
            error_test = self.rmse.evaluate(test_exps, M_test, S_test)
        else:
            error_test = np.array([np.nan])

        print(' - Task: %s - Method: %s - Repetition %d: test RMSE = %s, train RMSE = %s' %
              (task.name,
               config['methods'][method_i].name,
               rep_i + 1,
               np.array2string(error_test),
               np.array2string(error_train)))


class SaveRMSE(Output):

    def __init__(self, outpath, prjname):
        super(SaveRMSE, self).__init__(outpath, prjname)
        self.rmse = RMSE()

    def write_results(self, results):
        out_name = self.fname + '_all_RMSE.txt'
        with open(out_name, 'w') as f:
            self._write_header_to_file(f)
            for result in results:
                self._write_result_to_file(result, f)

    def write_result(self, result):
        task_i = result.task_i
        method_i = result.method_i
        rep_i = result.rep_i

        out_name = self.fname + '_t_%d_m_%d_r_%d_RMSE.txt' % (task_i, method_i, rep_i)
        with open(out_name, 'w') as f:
            self._write_header_to_file(f)
            self._write_result_to_file(result, f)

    def _write_header_to_file(self, f):
        f.write('RMSE results on test data\n')
        f.write('-------------------------------------\n')
        f.write('Task_ID, Taskname, Method_ID, Methodname, Repetition_ID, RMSE (test), RMSE (train)\n')

    def _write_result_to_file(self, result, file):
        task_i = result.task_i
        method_i = result.method_i
        rep_i = result.rep_i

        M_train = result.M_train
        S_train = result.S_train

        M_test = result.M_test
        S_test = result.S_test

        task = result.task
        config = result.config

        train_exps = task.train_exps
        test_exps = task.test_exps

        if not (M_train is None or S_train is None):
            error_train = self.rmse.evaluate(train_exps, M_train, S_train)
        else:
            error_train = np.array([np.nan])

        if not (M_test is None or S_test is None):
            error_test = self.rmse.evaluate(test_exps, M_test, S_test)
        else:
            error_test = np.array([np.nan])

        file.write('%d, %s, %d, %s, %d, %s, %s\n' %
                   (task_i, task.name,
                    method_i, config['methods'][method_i].name,
                    rep_i,
                    np.array2string(error_test),
                    np.array2string(error_train)))


class PrintLogLik(Output):

    def __init__(self, outpath, prjname):
        super(PrintLogLik, self).__init__(outpath, prjname)
        self.loglik = LogLik()

    def write_results(self, results):
        self._write_header_to_console()
        for result in results:
            self._write_result_to_console(result)

    def write_result(self, result):
        self._write_header_to_console()
        self._write_result_to_console(result)

    def _write_header_to_console(self):
        print()
        print('LogLik results on test data')
        print('-------------------------------------')

    def _write_result_to_console(self, result):
        task_i = result.task_i
        method_i = result.method_i
        rep_i = result.rep_i

        M_train = result.M_train
        S_train = result.S_train

        M_test = result.M_test
        S_test = result.S_test

        task = result.task
        config = result.config

        train_exps = task.train_exps
        test_exps = task.test_exps

        if not (M_train is None or S_train is None):
            error_train = self.loglik.evaluate(train_exps, M_train, S_train)
        else:
            error_train = np.array([np.nan])

        if not (M_test is None or S_test is None):
            error_test = self.loglik.evaluate(test_exps, M_test, S_test)
        else:
            error_test = np.array([np.nan])

        print(' - Task: %s - Method: %s - Repetition %d: test LogLik = %s, train LogLik = %s'%
              (task.name,
               config['methods'][method_i].name,
               rep_i + 1,
               np.array2string(error_test),
               np.array2string(error_train)))


class SaveLogLik(Output):
    """ Save log-likelihood of train and test predictions to file
    """

    def __init__(self, outpath, prjname):
        super(SaveLogLik, self).__init__(outpath, prjname)
        self.loglik = LogLik()

    def write_results(self, results):
        out_name = self.fname + '_all_LogLik.txt'
        with open(out_name, 'w') as f:
            self._write_header_to_file(f)
            for result in results:
                self._write_result_to_file(result, f)

    def write_result(self, result):
        task_i = result.task_i
        method_i = result.method_i
        rep_i = result.rep_i

        out_name = self.fname + '_t_%d_m_%d_r_%d_LogLik.txt' % (task_i, method_i, rep_i)
        with open(out_name, 'w') as f:
            self._write_header_to_file(f)
            self._write_result_to_file(result, f)

    def _write_header_to_file(self, f):
        f.write('RMSE results on test data\n')
        f.write('-------------------------------------\n')
        f.write('Task_ID, Taskname, Method_ID, Methodname, Repetition_ID, RMSE (test), RMSE (train)\n')

    def _write_result_to_file(self, result, file):
        task_i = result.task_i
        method_i = result.method_i
        rep_i = result.rep_i

        M_train = result.M_train
        S_train = result.S_train

        M_test = result.M_test
        S_test = result.S_test

        task = result.task
        config = result.config

        train_exps = task.train_exps
        test_exps = task.test_exps

        if not (M_train is None or S_train is None):
            error_train = self.loglik.evaluate(train_exps, M_train, S_train)
        else:
            error_train = np.array([np.nan])

        if not (M_test is None or S_test is None):
            error_test = self.loglik.evaluate(test_exps, M_test, S_test)
        else:
            error_test = np.array([np.nan])

        file.write('%d, %s, %d, %s, %d, %s, %s\n' %
                   (task_i, task.name,
                    method_i, config['methods'][method_i].name,
                    rep_i,
                    np.array2string(error_test),
                    np.array2string(error_train)))
