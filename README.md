# Probabilistic Recurrent State-Space Models

This is the companion code for the dynamics model learning method reported in the paper
Probabilistic Recurrent State-Space Models by Andreas Doerr et al., ICML 2018. The paper can
be found here https://arxiv.org/abs/1801.10395. The code allows the users to
reproduce the PR-SSM results reported in the benchmark and large-scale experiments. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Requirements, how to build, test, install, use, etc.

The PR-SSM code depends on Tensorflow.

### Prerequesits

In order to train a PR-SSM model for a new dataset, a new task has to be derived from the [task base class](prssm/tasks/tasks.py).
See for example [real_world_tasks.py](prssm/tasks/real_world_tasks.py).

A valid path must be provided to store the experimental results and log files.
An example is given in [run_benchmark_experiments.py](prssm/benchmarks/run_real_world_tasks/run_benchmark_experiments.py).

### Reproducing PR-SSM results

The experiments reported in the publication can be run by executing

```
python benchmarks/run_real_world_tasks/run_benchmark_experiments.py
python benchmarks/run_real_world_tasks/run_large_scale_experiment.py
```

The individual datasets have to be provided in the [datasets](datasets) folder.

## License

Probabilistic recurrent state-space models is open-sourced under the MIT license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Benchmarks, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
