# Probabilistic Recurrent State-Space Models

This is the companion code for the dynamics model learning method reported in the paper
Probabilistic Recurrent State-Space Models by Andreas Doerr et al., ICML 2018. The paper can
be found here https://arxiv.org/abs/1801.10395. The code allows the users to
reproduce and the PR-SSM results reported in the benchmark and large-scale experiments. Please cite the
above paper when reporting, reproducing or extending the results.

## Purpose of the project

This software is a research prototype, solely developed for and published as
part of the publication cited above. It will neither be
maintained nor monitored in any way.

## Requirements, how to build, test, install, use, etc.

Depends on Tensorflow.

Experiments can be run by calling

python benchmarks/run_real_world_tasks/run_benchmark_experiments.py
python benchmarks/run_real_world_tasks/run_loarge_scale_experiment.py

## License

Probabilistic recurrent state-space models is open-sourced under the MIT license. See the
[LICENSE](LICENSE) file for details.

For a list of other open source components included in Benchmarks, see the
file [3rd-party-licenses.txt](3rd-party-licenses.txt).
