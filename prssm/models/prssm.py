# -*- coding: utf-8 -*-
"""
Copyright (c) 2018 Robert Bosch GmbH
All rights reserved.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

@author: Andreas Doerr
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from prssm.models.base_model import DynamicsModel

from prssm.utils.utils_tf import tf_forward
from prssm.utils.utils_tf import backward
from prssm.utils.utils_tf import variable_summaries

from prssm.utils.initialization import init_inducing_inputs
from prssm.utils.initialization import init_inducing_outputs

from prssm.utils.data_management import get_subtrajectory_indices

from prssm.gp.gp_tf import RBF
from prssm.gp.gp_tf import conditional


class PRSSM(DynamicsModel):
    """ Gaussian Process State Space Model (GP-SSM)
    based on the paper "Probabilistic Recurrent State-Space Models" (Andreas Doerr et al.)

    The system is modelled as

    x_{t+1} = f(x_t, u_t) + \eps^(x)
    y_{t} = C x_t + \eps^(y)

    with \eps^(x) and \eps^(y) Gaussian noise and $f$ individual GPs for each
    latent state dimension.
    """

    name = 'PRSSM'

    def __init__(self):
        super(PRSSM, self).__init__()
        self.reset()

    def reset(self):
        # 1) Default model configuration
        # ------------------------------
        # Dimensionalities: Latent space, output space, input space
        self.Dx = 4
        self.Dy = 1
        self.Du = 1

        # Number of inducing points for sparse GP
        self.N_inducing = 25

        # Number of sampled latent state trajectories
        self.N_samples = 50

        # Minibatch of subtrajectories configuration
        self.N_batch = None   # number of subtrajectories per minibatch
        self.H_batch = None   # length of subtrajectory (time-steps)

        # 2) Default model initialization
        # -------------------------------
        # Inducing inputs
        self.config_inducing_inputs = {
                                       'method': 'zero',
                                       'noise': 0.01**2,
                                       'var': 0.001**2,
                                      }
        # Inducing outputs
        self.config_inducing_outputs = {
                                        'method': 'zero',
                                        'noise': 0.01**2,
                                        'var': 0.001**2,
                                       }

        # Method used to initialize initial state x_0
        self.x0_noise = 0.05
        self.x0_init_method = 'zero'  # any of: zero, output, nn or conv
        self.x0_init_h = 16

        # Observation noise
        self.var_y_np = 1**2
        self.optimize_observation_noise = True
        # Process noise
        self.var_x_np = 0.002**2
        self.process_noise = True
        # Kernel hyperparameters
        self.variance = 0.2**2
        self.lengthscales = 1

        # 3) Optimization configuration
        # -----------------------------
        # Optimization steps
        self.maxiter = 1000
        # learning rate
        self.learning_rate = 0.03
        # 'direct': optimize all model parameters jointly
        # 'em': model vs hyper-parameters
        self.optimization_method = 'direct'
        # EM configuration: optimization of hyps every optimization_hyps
        # iterations, for optimization_hyps_iter number of iterations
        self.optimization_hyps = 100
        self.optimization_hyps_iter = 20

        self.optimize_inducing_inputs = True

        # 4) Debug configuration
        # ----------------------
        # Print output settings (full output every output_epoch)
        self.output_epoch = 10
        # Save tensorboard logs in
        self.result_dir = 'dsvigpssm'

        # 5) Model state
        # --------------
        # Initialize model from stored parameters
        self.load_variables = False
        self.load_variables_file = None
        # Current best achieved loss
        self.elbo_loss_best = None

        self.plot_legends = True
        self.plot_output_ind = None

    def _fit(self, exps, val_exps=None):
        """Fits DS VI GP-SSM to the given experimental data

        Args:
            exps: 'list', List of experiments, each entry is a dict holding
                          experimental data and meta information
            val_exps: 'list', List of validation experiments, optional.
        Returns:
            bool, True if model was fit successfully, false otherwise
        """

        # Extract input/output dimensionality and sequence length from
        # experimental data should be equal for all training/test experiments
        exp = exps[0]

        self.Dy = exp['y_dim']
        self.Du = exp['u_dim']

        if self.plot_output_ind is None:
            self.plot_output_ind = list(range(self.Dy))

        # If minibatch is not configures (H_batch, N_batch == None) optimize
        # entire trajectory
        if self.H_batch is None:
            self.H_batch = exp['H']
            self.N_batch = 1

        # Construct new model graph according to the new data dimensions
        tf.reset_default_graph()
        self._build_graph()

        # Batch indices contain experiment ID and start ID for each
        # subtrajectory
        self.batch_indices = get_subtrajectory_indices(exps, self.H_batch)
        np.random.shuffle(self.batch_indices)

        Y_test = val_exps[0]['y'][:, None, :]                     # (H, 1, Dy)
        U_test = val_exps[0]['u'][:, None, :]                     # (H, 1, Du)

        with self._graph.as_default():
            with tf.Session() as sess:
                # Restore variables from file, otherwise default initialization
                if self.load_variables == True:
                    self.saver.restore(sess, self.load_variables_file)
                else:
                    sess.run(self.variable_initializer)

                # Optimize loss
                for i in range(self.maxiter):

                    if self.N_batch == 1:
                        U_train, Y_train = exps[0]['u'][:, None, :], exps[0]['y'][:, None, :]
                    else:
                        U_train, Y_train = self.get_minibatch(exps, i)       # (H_batch, N_batch, Dy), (H_batch, N_batch, Du)

                    # 1) Model update
                    # Version 1: update of both model and hyper parameters
                    if self.optimization_method == 'direct':
                        sess.run([self.train_op], feed_dict={self.Y_data: Y_train, self.U_data: U_train})

                    if self.optimization_method == 'em':
                        sess.run([self.train_model], feed_dict={self.Y_data: Y_train, self.U_data: U_train})
                        if i % self.optimization_hyps == 0:
                            for j in range(self.optimization_hyps_iter):
                                sess.run([self.train_hyps], feed_dict={self.Y_data: Y_train, self.U_data: U_train})

                    # 2) Visualization
                    if i % self.output_epoch == 0:
                        # Train + output on train and test data
                        X_train_pred, Y_train_pred, Y_train_mean_pred, Y_train_var_pred, J_train, summary_train = sess.run([self.X_final, self.Y_final, self.Y_mean, self.Y_var, self.elbo_loss, self.stats_summary_op], feed_dict={self.Y_data: Y_train, self.U_data: U_train})
                        X_test_pred, Y_test_pred, Y_test_mean_pred, Y_test_var_pred, J_test, summary_test = sess.run([self.X_final, self.Y_final, self.Y_mean, self.Y_var, self.elbo_loss, self.stats_summary_op], feed_dict={self.Y_data: Y_test, self.U_data: U_test})

                        self.writer_train.add_summary(summary_train, i)
                        self.writer_test.add_summary(summary_test, i)

                        fig = self.get_figure(2)
                        self.plot_prediction(fig, Y_train, U_train, X_train_pred, Y_train_pred, Y_train_mean_pred, Y_train_var_pred)
                        self.figure_to_summary(fig, self.writer_train, i)

                        fig = self.get_figure(3)
                        self.plot_prediction(fig, Y_test, U_test, X_test_pred, Y_test_pred, Y_test_mean_pred, Y_test_var_pred)
                        self.figure_to_summary(fig, self.writer_test, i)
                        print('Iter %d - Training: %f - Test: %f' % (i, J_train, J_test))
                    else:
                        J_train, summary_train = sess.run([self.elbo_loss, self.stats_summary_op], feed_dict={self.Y_data: Y_train, self.U_data: U_train})
                        self.writer_train.add_summary(summary_train, i)
                        print('Iter %d - Training: %f' % (i, J_train))

                    # 3) Save best model
                    if i % self.output_epoch == 0:
                        if self.elbo_loss_best is None:
                            self.elbo_loss_best = J_train
                        if J_train <= self.elbo_loss_best:
                            out_path =  self.result_dir + '/train/best_model%d.ckpt' % i
                            save_path_out = self.saver.save(sess, out_path)
                            print('Best model saved in file: %s' % save_path_out)
                            self.elbo_loss_best = J_train

                # Save final model
                out_path = self.result_dir + '/train/final_model.ckpt'
                save_path_out = self.saver.save(sess, out_path)
                print('Final model saved in file: %s' % save_path_out)

                self.load_variables = True
                self.load_variables_file = out_path

        return True

    def _freerun(self, exps):
        """Compute open loop DS VI GP-SSM predictions for the given experimental data

        Args:
            exps: 'list', List of experiments, each entry is a dict holding
                          experimental data and meta information
        Returns:
            'boolean', True if model was fit successfully, false otherwise
        """
        M = []
        S = []

        with self._graph.as_default():
            with tf.Session() as sess:

                if self.load_variables is True:
                    self.saver.restore(sess, self.load_variables_file)
                else:
                    print('Model is not yet initialized')
                    return M, S

                for exp in exps:
                    # Compute open loop predictions
                    M_pred, S_pred = sess.run([self.IO_mean, self.IO_var], feed_dict={self.Y_data: exp['y'][:,None,:], self.U_data: exp['u'][:,None,:]})
                    M.append(M_pred)
                    S.append(S_pred)

        return M, S

    def retrieve_tf_results(self, exps, var_list):
        """ Evaluate list of tensorflow model variables

        Args:
            exps: 'list', List of experiments, each entry is a dict holding
                          experimental data and meta information
            var_list: 'list', List of tensorflow ops in model graph to be evaluated
        Returns:
            'list', list of evaluated tf ops for each exp in exps
        """
        results = []

        with self._graph.as_default():
            with tf.Session() as sess:

                if self.load_variables is True:
                    self.saver.restore(sess, self.load_variables_file)
                else:
                    print('Model is not yet initialized')
                    return results

                for exp in exps:
                    # Compute open loop predictions
                    result = sess.run(var_list, feed_dict={self.Y_data: exp['y'][:,None,:], self.U_data: exp['u'][:,None,:]})
                    results.append(result)

        return results

    def _initialize_model(self):
        """
        """
        # Model initialization, depends on data dimensions (Du, Dy)
        self.Z_np = init_inducing_inputs(self.config_inducing_inputs,
                                         P=self.N_inducing,
                                         D=self.Dx + self.Du)

        self.zeta_mean_np, self.zeta_var_np = init_inducing_outputs(self.config_inducing_outputs,
                                                                    P=self.N_inducing, D=self.Dx, Z=self.Z_np)

        # initial state (N_batch, N_samples, Dx)
        self.x0_np = np.sqrt(self.x0_noise) * np.random.randn(self.N_batch, self.N_samples, self.Dx)

        # Convert sensor and process noise into numpy array
        self.var_y_np = np.atleast_1d(self.var_y_np)

        if self.process_noise is True:
            self.var_x_np = np.atleast_1d(self.var_x_np)

    def _recognize_conv(self, Y, U):
        """ Recognition model for initial latent state given I/O trajectory

        Args:
            Y: Tensor, (H, N_batch, Dy), System output
            U: Tensor, (H, N_batch, Du), System input

        """
        X = tf.concat((Y, U), axis=2)    # (H, N_batch, Dy + Du)
        X = tf.transpose(X, [1, 0, 2])   # (N_batch, H, Dy + Du)
        X = X[:, :self.x0_init_h, :]     # (N_batch, x0_init_h, Dy + Du)

        X = tf.cast(X, tf.float32)

        variable_summaries(X, 'X_init')

        layer1 = tf.layers.conv1d(X, 5, 3, activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling1d(layer1, 2, 2)
        out1 = tf.reshape(pool1, [self.N_batch_tf, 35])
        dense2 = tf.layers.dense(out1, self.Dx)

        variable_summaries(tf.get_default_graph().get_tensor_by_name(os.path.split(dense2.name)[0] + '/kernel:0'), 'dense1_kernel')
        variable_summaries(tf.get_default_graph().get_tensor_by_name(os.path.split(dense2.name)[0] + '/bias:0'), 'dense1_bias')

        variable_summaries(tf.get_default_graph().get_tensor_by_name(os.path.split(layer1.name)[0] + '/bias:0'), 'layer1_bias')
        variable_summaries(tf.get_default_graph().get_tensor_by_name(os.path.split(layer1.name)[0] + '/kernel:0'), 'layer1_kernel')

        return tf.cast(dense2, tf.float64)   # (N_batch, Dx)

    def _recognize_conv_dist(self, Y, U):
        """ Recognition model for initial latent state given I/O trajectory

        Args:
            Y: Tensor, (H, N_batch, Dy), System output
            U: Tensor, (H, N_batch, Du), System input

        returns:
            x_mean: (N_batch, Dx), mean of multivariate Gaussian
            x_std: (N_batch, Dx), std of multivariate Gaussian
        """
        n = self.x0_init_h

        X = tf.concat((Y, U), axis=2)    # (H, N_batch, Dy + Du)
        X = tf.transpose(X, [1, 0, 2])   # (N_batch, H, Dy + Du)
        X = X[:, :n, :]                  # (N_batch, n, Dy + Du)

        X = tf.cast(X, tf.float32)

        variable_summaries(X, 'X_init')

        # Convolutional layers
        # (batch, x0_init_h, 2) --> (batch, x0_init_h / 2, 8)
        conv1 = tf.layers.conv1d(inputs=X, filters=16, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu)
        max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2, padding='same')

        # (batch, x0_init_h / 2, 8) --> (batch, x0_init_h / 4, 16)
        conv2 = tf.layers.conv1d(inputs=max_pool_1, filters=32, kernel_size=2, strides=1, padding='same', activation = tf.nn.relu)
        max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2, padding='same')

        # Predictions from fully connected layer
        flat = tf.reshape(max_pool_2, (-1, (self.x0_init_h // 4) * 32))
        y = tf.layers.dense(flat, 2*self.Dx)

        # Compute initial state mean and std from prediction
        x_mean = y[:, :self.Dx]
        x_std = tf_forward(y[:, self.Dx:])

        return tf.cast(x_mean, tf.float64), tf.cast(x_std, tf.float64)

    def _recognize(self, Y, U):

        n = 10

        X = tf.concat((Y, U), axis=2)  # (H, N_batch, Dy + Du)
        X = tf.transpose(X, [0,2,1])   # (H, Dy + Du, N_batch)
        X = X[:n, :, :]                # (n, Dy + Du, N_batch)
        X = tf.reshape(X, [(self.Dy + self.Du) * n, self.N_batch_tf])  # (n * (Dy + Du), N_batch)

        W1 = tf.Variable(tf.random_normal((10, 20), mean=0.0, stddev=0.1, dtype=tf.float64), name='W1')
        b1 = tf.Variable(tf.random_normal((10, 1), mean=0.0, stddev=0.1, dtype=tf.float64), name='b1')

        W2 = tf.Variable(tf.random_normal((self.Dx, 10), mean=0.0, stddev=0.1, dtype=tf.float64), name='W2')
        b2 = tf.Variable(tf.random_normal((self.Dx, 1), mean=0.0, stddev=0.1, dtype=tf.float64), name='b2')

        Y = tf.sigmoid(tf.matmul(W2, tf.sigmoid(tf.matmul(W1, X) + b1)) + b2)   # (Dx, N_batch)

        return Y

    def _build_graph(self):

        self._initialize_model()
        self._graph = tf.Graph()

        with self._graph.as_default():

            # Placeholder for (batches) of system input/output data
            # (H, N_batch, Dy)
            self.Y_data = tf.placeholder(dtype=tf.float64,
                                         shape=(None, None, self.Dy),
                                         name='Y_data')
            # (H, N_batch, Du)
            self.U_data = tf.placeholder(dtype=tf.float64,
                                         shape=(None, None, self.Du),
                                         name='U_data')

            # Subtrajectory length (H) and batch size (N_batch) dynamic size
            self.H_tf = tf.shape(self.Y_data)[0]
            self.N_batch_tf = tf.shape(self.Y_data)[1]

            # TensorArrays for while-loop computations
            self.X = tf.TensorArray(dtype=tf.float64, size=0,
                                    dynamic_size=True, clear_after_read=False,
                                    name='X')
            self.U = tf.TensorArray(dtype=tf.float64, size=0,
                                    dynamic_size=True, clear_after_read=False,
                                    name='U')

            # Dublicate input sequence for each sampled trajectory
            U_dub = tf.tile(tf.expand_dims(self.U_data, axis=2, name='add_sampling_dimension'), [1, 1, self.N_samples, 1], name='duplicate_input')
            self.U = self.U.unstack(U_dub, name='U_list')

            # GP inducing inputs (initialized from numpy array) (N_inducing, Dx + Du)
            if self.optimize_inducing_inputs is True:
                self.Z = tf.Variable(self.Z_np,
                                     collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                  'model_variables'],
                                     name='GP_pseudo_inputs')
            else:
                self.Z = tf.constant(self.Z_np, name='GP_pseudo_inputs')

            # GP inducing outputs (N_inducing, Dx) mean and variance for each dimension Dx
            self.zeta_mean = tf.Variable(self.zeta_mean_np,
                                         collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                      'model_variables'],
                                         name='GP_pseudo_outputs_mean')
            self.zeta_var_unc = tf.Variable(backward(self.zeta_var_np),
                                            collections=[tf.GraphKeys.GLOBAL_VARIABLES,
                                                         'model_variables'],
                                            name='GP_pseudo_outputs_var_unconstrained')
            self.zeta_var = tf_forward(self.zeta_var_unc,
                                       name='GP_pseudo_outputs_var_constrained')

            # GP inducing outputs: Dx multivariate normal distributions having diagonal covariance self.zeta_var and mean self.zeta_mean
            self.zeta_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=tf.transpose(self.zeta_mean), scale_diag=tf.sqrt(tf.transpose(self.zeta_var)), name='zeta_dist')

            # Sensor noise as initialized by numpy self.var_y_np
            if self.optimize_observation_noise is True:
                self.var_y_unc = tf.Variable(backward(self.var_y_np), collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'hyperparameters'], name='sensor_noise_unconstrained')
                self.var_y = tf_forward(self.var_y_unc, name='sensor_noise_constrained')
            else:
                self.var_y = tf.constant(self.var_y_np, dtype=tf.float64)

            # Process noise, initialized by numpy self.var_x_np
            if self.process_noise is True:
                self.var_x_unc = tf.Variable(backward(self.var_x_np), collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'hyperparameters'], name='process_noise_unconstrained')
                self.var_x = tf_forward(self.var_x_unc, name='process_noise_constrained')

            # GP kernel
            self.kern = RBF(self.Dx + self.Du, self.variance, self.lengthscales)

            # Initial state initialization
            # 1) LSTM-like (initialize by zero)
            if self.x0_init_method == 'zero':
                x0 = tf.constant(self.x0_np, dtype=tf.float64)
                x0_noise = tf.cast(tf.sqrt(self.x0_noise), tf.float64) * tf.random_normal((self.N_batch_tf, self.N_samples, self.Dx), dtype=tf.float64)
                x0 = x0 + x0_noise                                           # (N_batch, N_samples, Dx)

            # 2) Measured system output in first part of latent state, everything else 0 + noise
            if self.x0_init_method == 'output':
                y0 = self.Y_data[0,:,:]                                              # (N_batch, Dy)
                y0_pad = tf.pad(y0, [[0,0], [0,self.Dx - self.Dy]], 'CONSTANT')      # (N_batch, Dy + Dx - Dy) = (N_batch, Dx)
                y0_pad_exp = tf.expand_dims(y0_pad, axis=1)                          # (N_batch, 1, Dx)
                x0_noise = tf.cast(tf.sqrt(self.x0_noise), tf.float64) * tf.random_normal((self.N_batch_tf, self.N_samples, self.Dx), dtype=tf.float64)
                x0 = y0_pad_exp + x0_noise                                           # (N_batch, N_samples, Dx)
            # 3) Recognition model using fully connected NN
            if self.x0_init_method == 'nn':
                x0 = self._recognize(self.Y_data, self.U_data) # (Dx, N_batch)
            # 4) Recognition model using convolutional NN
            if self.x0_init_method == 'conv':
                x0_noisefree = self._recognize_conv(self.Y_data, self.U_data) # (N_batch, Dx)
                x0 = tf.expand_dims(x0_noisefree, axis=1) + tf.zeros((self.N_batch_tf, self.N_samples, self.Dx), dtype=tf.float64)

            # 5) Recognition model using CNN for mean and variance
            if self.x0_init_method == 'conv_dist':
                x0_mean, x0_std = self._recognize_conv_dist(self.Y_data, self.U_data) # (N_batch, Dx), (N_batch, Dx)
                x0_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=x0_mean, scale_diag=x0_std, name='x0_dist')
                x0 = tf.transpose(x0_dist.sample(self.N_samples), [1, 0, 2])  # (N_batch, N_samples, Dx)

            # Compute variational lower bound
            self.K_prior = self.kern.K(self.Z, self.Z)   # (N, N)
            self.scale_prior = tf.tile(tf.expand_dims(tf.cholesky(self.K_prior), 0), [self.Dx, 1, 1])

            # GP prior: N(0, K) for each latent state dimension (Dx, P)
            self.zeta_prior = tf.contrib.distributions.MultivariateNormalTriL(loc=tf.zeros((self.Dx, self.N_inducing), dtype=tf.float64), scale_tril=self.scale_prior, name='zeta_prior')

            var_y_exp = tf.expand_dims(tf.expand_dims(tf.expand_dims(self.var_y, 0), 0), 0)          # (1, 1, 1, Dy)
            self.var_full = tf.tile(var_y_exp, [self.H_tf, self.N_batch_tf, self.N_samples, 1])      # (H_batch, N_batch, N_samples, Dy)

            # Write initial latent state x0
            t = 0
            self.X = self.X.write(t, x0, name='set_initial_state')

            # Loop over entire input/output trajectory
            self.U_final, self.X_final, self.t_final = tf.while_loop(self.condition, self.body, [self.U, self.X, t], parallel_iterations=1)

            # Stack TensorArray into Tensor
            self.X_final = self.X_final.stack()   # (H, N_batch, N_samples, Dx)
            self.U_final = self.U_final.stack()   # (H, N_batch, N_samples, Du)

            # First Dy dimensions from latent state are observations
            self.Y_final = self.X_final[:,:,:,:self.Dy]     # (H, N_batch, N_samples, Dy)

            # Compute mean and standard from sampled distributions
            self.X_mean, self.X_var = tf.nn.moments(self.X_final, axes=[2]) # (H, N_batch, N_samples, Dx) --> (H, N_batch, Dx)
            self.Y_mean, self.Y_var = tf.nn.moments(self.Y_final, axes=[2]) # (H, N_batch, N_samples, Dy) --> (H, N_batch, Dy)
            # Add sensor noise to observation
            self.Y_var = tf.add(self.Y_var, self.var_y, name='add_sensor_noise')   # (H, N_batch, Dy) + (1) --> (H, N_batch, Dy)

            # Compute log likelihood of data given samples
            Y_dist = tf.contrib.distributions.MultivariateNormalDiag(loc=self.Y_final, scale_diag=tf.sqrt(self.var_full), name='Y_dist')
            loglik = tf.reduce_sum(Y_dist.log_prob(tf.tile(tf.expand_dims(self.Y_data, 2), [1, 1, self.N_samples, 1])))

            # General output format: Mean and variance for system output + input (variance as full matrix)
            self.IO_mean = tf.squeeze(tf.concat((self.Y_mean, self.U_data), axis=2))   # (H, N_batch, Dy + Du) or (H, Dy + Du) if N_batch == 1
            self.IO_var = tf.matrix_diag(tf.squeeze(tf.concat((self.Y_var, tf.zeros_like(self.U_data)), axis=2)))  # (H, N_batch, Dy + Du, Dy + Du)

            # Compute variational lower bound
            self.elbo = 0
            # KL divergence between variational inducing output distribution and GP prior
            self.elbo += -tf.reduce_sum(tf.contrib.distributions.kl_divergence(self.zeta_dist, self.zeta_prior))
            # Expectation over observation log likelihoods approximated from samples
            self.elbo += loglik

            # Compute loss
            # 1) Negative VI lower bound
            self.elbo_loss = tf.negative(self.elbo, name='elbo_loss')
            # 2) Negative log likelihood
            self.loglik_loss = tf.negative(loglik, name='loglik_loss')
            # 3) L2-loss
            self.l2_loss = tf.reduce_mean(tf.square(self.Y_mean - self.Y_data), name='l2_loss')
            # 4) RMSE-loss
            self.rmse_loss = tf.sqrt(self.l2_loss, name='rmse_loss')

            # Set up optimizer
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_model = tf.train.AdamOptimizer(self.learning_rate)
            self.optimizer_hyps = tf.train.AdamOptimizer(self.learning_rate)

            # Ops for training steps on model parameters, hyperparameters or both
            self.train_op = tf.contrib.training.create_train_op(self.elbo_loss, self.optimizer, summarize_gradients=True)
            self.train_model = tf.contrib.training.create_train_op(self.elbo_loss, self.optimizer_model, variables_to_train=tf.get_collection('model_variables'), summarize_gradients=True)
            self.train_hyps = tf.contrib.training.create_train_op(self.elbo_loss, self.optimizer_hyps, variables_to_train=tf.get_collection('hyperparameters'), summarize_gradients=True)

            # Set up prediction plotting as tensorboard summary
            fig = self.get_figure()
            self.image_placeholder = tf.placeholder(tf.uint8, self.fig2rgb_array(fig).shape)
            self.image_summary = tf.summary.image('prediction_plot', self.image_placeholder, collections=["image_summaries"])

            logdir = self.result_dir + '/'

            self.writer_train = tf.summary.FileWriter(logdir + 'train/', tf.get_default_graph())
            self.writer_test = tf.summary.FileWriter(logdir + 'test/')

            self.saver = tf.train.Saver()

            # Add summaries for tensorboard visualization
            variable_summaries(self.var_y, 'var_y')
            if self.process_noise is True:
                variable_summaries(self.var_x, 'var_x')
            variable_summaries(self.Z, 'Z')
            variable_summaries(self.zeta_mean, 'zeta_mean')
            variable_summaries(self.zeta_var, 'zeta_var')
            variable_summaries(self.elbo, 'elbo', vector=False)
            variable_summaries(self.elbo_loss, 'elbo_loss', vector=False)
            variable_summaries(self.loglik_loss, 'loglik_loss', vector=False)
            variable_summaries(self.l2_loss, 'l2_loss', vector=False)
            variable_summaries(self.rmse_loss, 'rmse_loss', vector=False)
            variable_summaries(x0, 'x0')

            self.stats_summary_op = tf.summary.merge_all(key="stats_summaries")

            self.variable_initializer = tf.global_variables_initializer()

    def get_minibatch(self, exps, i):

        # Select batch size subtrajectories, start from beginning if list is exhausted
        start = (i*self.N_batch) % (self.batch_indices.shape[0] - self.N_batch)
        current_batch_indices = self.batch_indices[start:start+self.N_batch]

        U = []
        Y = []

        for i in range(self.N_batch):

            exp_ind = current_batch_indices[i, 0]
            start_ind = current_batch_indices[i, 1]

            U.append(exps[exp_ind]['u'][start_ind:start_ind+self.H_batch])
            Y.append(exps[exp_ind]['y'][start_ind:start_ind+self.H_batch])

        U = np.asarray(U)
        Y = np.asarray(Y)

        U = np.swapaxes(U, 0, 1)
        Y = np.swapaxes(Y, 0, 1)

        return U, Y

    def mean_function(self, X, name=None):
        """Identity mean function.
        Computes GP mean function from N datapoints (rows of X) of length Dx+Du
        First Dx elements of each datapoint are returned.

        Args:
            X: 'Tensor', input data (..., Dx + Du)
        Returns:
            'Tensor' (..., Dx)
        """
        with tf.name_scope(name, 'mean_function', [X]):
            X = tf.convert_to_tensor(X)
            return X[..., :self.Dx]

    def transition_model(self, U, X, name=None):
        """Compute next latent state given current state and input

        Args:
            U: 'Tensor', System input (..., Du)
            X: 'Tensor', System state (..., Dx)
        Returns:
            'Tensor', (..., Dx) Sample from p(y*|x*, X, Y)
        """
        with tf.name_scope(name, 'transition_model', [U, X]):
            U = tf.convert_to_tensor(U)
            X = tf.convert_to_tensor(X)

            # Input to dynamics model: (..., Dx + Du)
            Xnew = tf.concat((X, U), axis=2, name='merge_xu_for_gp')

            Xnew = tf.reshape(Xnew, (self.N_batch_tf * self.N_samples, self.Dx + self.Du))

            fmean, fvar = conditional(Xnew, self.Z, self.kern, self.zeta_mean, tf.sqrt(self.zeta_var))
            fmean = tf.add(fmean, self.mean_function(Xnew), name='add_mean_function')

            if self.process_noise is True:
                fvar = tf.add(fvar, self.var_x, name='add_process_noise')

            fmean = tf.reshape(fmean, (self.N_batch_tf, self.N_samples, self.Dx))
            fvar = tf.reshape(fvar, (self.N_batch_tf, self.N_samples, self.Dx))

            eps = tf.tile(tf.random_normal((self.N_batch_tf, self.N_samples, 1), dtype=tf.float64), [1, 1, self.Dx])
            x_next = tf.add(fmean, tf.multiply(eps, fvar, name='sample_variance'), name='add_sample_variance')

            return x_next

    def body(self, U, X, t, name=None):
        """One-step GP-SSM prediction
        """
        with tf.name_scope(name, 'GP-SSM-Prediction', [U, X]):

            x_t = X.read(t, name='current_input_mean')   # (N_batch, N_samples, Dx)
            u_t = U.read(t, name='current_input_var')    # (N_batch, N_samples, Du)

            x_next = self.transition_model(u_t, x_t)

            t_next = tf.add(t, 1, name='increase_loop_counter')

            X = X.write(t_next, x_next, name='next_state')

            return (U, X, t_next)

    def condition(self, U, X, t, name=None):
        """Loop until time horizon H is reached
        """
        with tf.name_scope(name, 'loop_condition', [U, X]):
            return t < (self.H_tf - 1)

    def plot_prediction(self, fig, Y, U, X_pred, Y_pred, Y_mean_pred,
                        Y_var_pred, stride=1, batch_id=0, name=None):
        """Create tensor holding plot image of prediction and groundtruth

        Args:
            fig:
            Y: ndarray (H, N_batch, Dy), batch of measured system output trajectories
            U: ndarray (H, N_batch, Du), batch of system input trajectories
            X_pred: ndarray (H, N_batch, N_samples, Dx), predicted latent state samples for all trajectories in batch
            Y_pred: ndarray (H, N_batch, N_samples, Dy), predicted output samples for all trajectories in batch
            Y_mean_pred: ndarray (H, N_batch, Dy), mean over all samples for output predictions
            Y_var_pred: ndarray (H, N_batch, Dy), variance over all samples for output predictions (including sensor noise)
        """
        H, N_batch, Dy = Y.shape
        N_samples = X_pred.shape[2]

        X_mean = np.mean(X_pred, axis=2)  # (H, N_batch, N_samples, Dx) --> (H, N_batch, Dx)
        X_std = np.std(X_pred, axis=2)  # (H, N_batch, N_samples, Dx) --> (H, N_batch, Dx)

        ind = np.arange(H)[::stride]

        for batch_id in range(N_batch):

            # Visualize output
            ax = fig.add_subplot(3, N_batch, batch_id + 1)
            for dim in range(self.Dy):
                if dim in self.plot_output_ind:
                    # Draw observed output
                    ax.plot(ind, Y[ind, batch_id, dim], '+-', color='C%d' % (dim % 10), label='Output $y_%d$'%dim)

                    # Draw predicted output mean + std
                    y = Y_mean_pred[ind, batch_id, dim]
                    error = 2 * np.sqrt(Y_var_pred[ind, batch_id, dim])
                    ax.fill_between(ind, y-error, y+error, color='C%d'% ((self.Dy + dim) % 10), alpha=0.5)
                    ax.errorbar(ind, y, yerr=error, color='C%d' % ((self.Dy + dim) % 10), label='Prediction $y_%d$'%dim)
            if self.plot_legends is True:
                ax.legend()
            ax.grid()
            plt.xlim(0, H)

            # Visualize latent state
            ax2 = fig.add_subplot(3, N_batch, N_batch + batch_id + 1)
            for dim in range(self.Dx):
                # Draw latent state samples
                for sample in range(N_samples):
                    ax2.plot(ind, X_pred[ind, batch_id, sample, dim], '-', color=[0.8,0.8,0.8], alpha=0.2)

                # Draw latent state mean + std
                y = X_mean[ind, batch_id, dim]
                error = 2 * X_std[ind, batch_id, dim]

                ax2.fill_between(ind, y-error, y+error, color='C%d' % (dim % 10), alpha=0.5)
                ax2.errorbar(ind, y, yerr=error, color='C%d'%(dim % 10), label='State $x_%d$'%dim)
            if self.plot_legends is True:
                ax2.legend()
            ax2.grid()
            plt.xlim(0, H)

            # Visualize input
            ax3 = fig.add_subplot(3, N_batch, 2*N_batch + batch_id + 1)
            for dim in range(self.Du):
                ax3.plot(ind, U[ind, batch_id, dim], '+-', color='C%d'%(dim % 10), label='Input $u_%d$'%dim)
            if self.plot_legends is True:
                ax3.legend()
            ax3.grid()
            plt.xlim(0, H)

        plt.tight_layout()

    def get_figure(self, num=0):
        fig = plt.figure(num=num, figsize=(15, 10), dpi=72)
        fig.clf()
        return fig

    def fig2rgb_array(self, fig, expand=True):
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        shape = (nrows, ncols, 3) if not expand else (1, nrows, ncols, 3)
        return np.fromstring(buf, dtype=np.uint8).reshape(shape)

    def figure_to_summary(self, fig, summary_writer, iteration):
        image = self.fig2rgb_array(fig)
        summary_writer.add_summary(self.image_summary.eval(feed_dict={self.image_placeholder: image}), iteration)

    def load_model(self, exps, config, name):
        """ Load model variables from file
        Args:
            name: 'string', path and name of the saver file
        """
        self.configure(config)

        # Extract input/output dimensionality and sequence length from
        # experimental data should be equal for all training/test experiments
        exp = exps[0]

        self.Dy = exp['y_dim']
        self.Du = exp['u_dim']

        # If minibatch is not configures (H_batch, N_batch == None) optimize
        # entire trajectory
        if self.H_batch is None:
            self.H_batch = exp['H']
            self.N_batch = 1

        # Construct new model graph according to the new data dimensions
        self._build_graph()

        self.load_variables = True
        self.load_variables_file = name
