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

from prssm.tasks.real_world_tasks import Actuator
from prssm.tasks.real_world_tasks import Ballbeam
from prssm.tasks.real_world_tasks import Drive
from prssm.tasks.real_world_tasks import Gas_furnace
from prssm.tasks.real_world_tasks import Dryer

from prssm.benchmarks.outputs import VisualOutput
from prssm.benchmarks.outputs import PrintRMSE
from prssm.benchmarks.outputs import SaveRMSE
from prssm.benchmarks.outputs import PrintLogLik
from prssm.benchmarks.outputs import SaveLogLik
from prssm.benchmarks.outputs import SaveStartScript
from prssm.benchmarks.outputs import SavePredictionResults

from prssm.models.prssm import PRSSM

from prssm.utils.utils import create_dated_directory

# List of tasks for benchmark experiments
all_tasks = [Actuator,
             Ballbeam,
             Drive,
             Gas_furnace,
             Dryer]

# Create a directory for experimental results
outpath = create_dated_directory('results/benchmark')
prjname = 'PRSSM'

# Configuration of PR-SSM model and inference
Dx = 4
Dy = 1
Du = 1

config_inducing_inputs = {
                          'method': 'uniform',
                          'low': -2,
                          'high': 2
                         }

config_inducing_outputs = {
                           'method': 'zero',
                           'noise': 0.05**2,     #
                           'var': 0.01**2,
                          }

PRSSM_config = {
                'x0_noise': 0.1**2,    # noise variance on initial state
                'var_y_np': [1**2]*Dy,      # observation noise variance
                'var_x_np': [0.002**2]*Dx,
                'variance': 0.5**2,
                'lengthscales': [2]*(Dx + Du),   # kernel lengthscales
                'config_inducing_inputs': config_inducing_inputs,
                'config_inducing_outputs': config_inducing_outputs,
                'x0_init_method': 'conv',
                'Dx': Dx,
                'N_inducing': 20,
                'N_samples': 50,
                'maxiter': 3000,
                'learning_rate': 0.1,
                'result_dir': outpath,
                'normalize_data': True,
                'N_batch': 10,
                'H_batch': 50,
                'output_epoch': 50
               }

# Configuration of benchmark experiment
config = {
          'methods': [PRSSM],
          'method_configs': [PRSSM_config],
          'tasks': all_tasks,
          'repeats': 3,
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
          'raise_exception': False,
          'validate': True
         }

if __name__ == '__main__':
    run(config)
