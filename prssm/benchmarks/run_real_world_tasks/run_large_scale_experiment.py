# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

import os

from prssm.benchmarks.run import run

from prssm.tasks.real_world_tasks import SarcosArm

from prssm.benchmarks.outputs import VisualOutput
from prssm.benchmarks.outputs import PrintRMSE
from prssm.benchmarks.outputs import SaveRMSE
from prssm.benchmarks.outputs import PrintLogLik
from prssm.benchmarks.outputs import SaveLogLik
from prssm.benchmarks.outputs import SaveStartScript
from prssm.benchmarks.outputs import SavePredictionResults

from prssm.models.prssm import PRSSM

from prssm.utils.utils import create_dated_directory

# Create a directory for experimental results
outpath = create_dated_directory('results/sarcos')
prjname = 'PRSSM'

# Configuration of PR-SSM model and inference
Dx = 14
Dy = 7
Du = 7

# System input: torques, System output: positions
# Select first 60 experiments for training, last 6 for testing
task_config = {
               'input_ind': list(range(21, 28)),
               'output_ind': list(range(0, 7)),
               'downsample': 2,
               'train_ind': list(range(0, 60)),
               'test_ind': list(range(60, 66))
              }


config_inducing_inputs = {
                          'method': 'uniform',
                          'low': -3,
                          'high': 3
                         }

config_inducing_outputs = {
                           'method': 'zero',
                           'noise': 0.1**2,     #
                           'var': 0.01**2,
                          }

PRSSM_config = {
                    'x0_noise': 0.001**2,    # noise variance on initial state
                    'variance': 0.5**2,
                    'lengthscales': [2]*(Dx + Du),   # kernel lengthscales
                    'config_inducing_inputs': config_inducing_inputs,
                    'config_inducing_outputs': config_inducing_outputs,
                    'x0_init_method': 'output',
                    'Dx': Dx,
                    'N_inducing': 50,
                    'N_samples': 20,
                    'maxiter': 1000,
                    'learning_rate': 0.05,
                    'result_dir': outpath,
                    'normalize_data': True,
                    'N_batch': 5,
                    'H_batch': 100,
                    'output_epoch': 50,
                    'var_x_np': [0.002**2]*Dx,
                    'process_noise': False,
                    'var_y_np': [0.05**2]*Dy,      # observation noise variance
                    'optimize_observation_noise': False,
                    'plot_output_ind': [0],
                    'plot_legends': False
                   }

# Configuration of large scale experiment
config = {
          'methods': [PRSSM],
          'method_configs': [PRSSM_config],
          'tasks': [SarcosArm],
          'task_configs': [task_config],
          'repeats': 1,
          'outpath': outpath,
          'prjname': prjname,
          'outputs': [VisualOutput(outpath, prjname),
                      PrintRMSE(outpath, prjname),
                      SaveRMSE(outpath, prjname),
                      PrintLogLik(outpath, prjname),
                      SaveLogLik(outpath, prjname),
                      SaveStartScript(outpath, prjname, os.path.realpath(__file__)),
                      SavePredictionResults(outpath, prjname)],
          'intermediate_results': True,
          'raise_exception': True,
          'validate': True
         }

if __name__ == '__main__':
    run(config)
