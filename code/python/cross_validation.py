#!/usr/bin/env python
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Functions for conducting k-fold cross-validation.
#
###############################################################################

import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import timeit
from utils import *
from helper_layers import *
from read_data import *
from base_class import BaseClass
from mirt_vae import *

# Find optimal number of factors based on cross-validated log-likelihoods.
def find_best_n_factors(cv_res,
                        n_factors_grid,
                        one_stdev = False):
    """
    Args:
        cv_res         (dict): Dictionary holding cross-validation results.
        n_factors_grid (list of int): Grid of numbers of factors.
    """
    mean_ll = [np.mean(ll) for ll in cv_res["log_likelihood"]]
    std_ll = [np.std(ll) for ll in cv_res["log_likelihood"]]
    min_ll = min(mean_ll)
    min_ll_idx = mean_ll.index(min_ll)
    
    if one_stdev:
        best_idx = min([idx for idx, ll in enumerate(mean_ll) if
                        ll < min_ll + std_ll[min_ll_idx]])
    else:
        best_idx = min_ll_idx
    
    return n_factors_grid[min_ll_idx]

# Conduct k-fold cross-validation.
def cross_validation(n_folds,
                     n_factors_grid,
                     df,
                     batch_size,
                     input_dim,
                     n_cats,
                     learning_rate,
                     device,
                     log_interval,
                     steps_anneal,
                     kwargs = {},
                     max_iter = 3000,
                     mc_samples = 1,
                     iw_samples = 1,
                     correlated_factors = False,
                     seed = 1,
                     cv_seed = 1):
    """
    Args:
        n_folds             (int): Number of cross validation folds.
        n_factors_grid      (list of int): Grid of numbers of factors.
        df                  (DataFrame): Pandas data frame containing data set.
        batch_size          (int): Batch size for stochastic gradient descent.
        input_dim           (int): Input vector dimension.
        n_cats              (list of int):  List containing number of categories for each observed variable.
        learning_rate       (float): Learning rate for stochastic gradient descent.
        device              (str): String specifying whether to run on CPU or GPU.
        log_interval        (int): Number of batches to wait before printing progress report.
        steps_anneal        (int): Number of batches over which to linearly anneal KL divergence.
        max_iter            (float): Maximum possible number of fitting iterations; may be fractional.
        kwargs              (dict): Dictionary containing CUDA parameters, if used.
        mc_samples          (int): Number of Monte Carlo samples to draw at each fitting iteration.
        iw_samples          (int): Number of importance-weighted samples to draw at each fitting iteration.
        correlated_factors  (Boolean): Whether or not factors should be correlated.
        seed                (int): Starting seed for model fitting.
        cv_seed             (int): Seed for cross_validation splits.
    """
    # Set random seeds.
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Define dictionary for holding results.
    keys = ["BIC", "log_likelihood", "ELBO", "loadings", "run_time"]
    cv_res = {key : [[] for _ in range(len(n_factors_grid))] for key in keys}

    # Generate cross-validation folds.
    kf = KFold(n_splits = n_folds,
               shuffle = True,
               random_state = cv_seed)

    # Outer loop over cross-validation folds.
    for cur_fold_idx, cur_data_idxs in enumerate(kf.split(df)):        
        # Create train set data loader.
        train_idx = cur_data_idxs[0]
        train_tensor = torch.from_numpy(df.iloc[train_idx, :].to_numpy())
        train_loader = torch.utils.data.DataLoader(tensor_dataset(train_tensor),
                                                   batch_size = batch_size,
                                                   shuffle = True, **kwargs)

        # Create test set data loader.
        test_idx = cur_data_idxs[1]
        test_tensor = torch.from_numpy(df.iloc[test_idx, :].to_numpy())
        test_loader = torch.utils.data.DataLoader(tensor_dataset(test_tensor),
                                                  batch_size = batch_size,
                                                  shuffle = True, **kwargs)

        # Inner loop over regularization hyperparameter grid.
        for factor_idx, n_factors in enumerate(n_factors_grid):
            # Fit model while ensuring numerical stability.
            nan_output = True
            while nan_output:
                # Re-set random seeds.
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Initialize model.
                vae = MIRTVAEClass(input_dim = input_dim,
                                   encoder_dims = [int(np.floor((input_dim + 2 * n_factors) / 2))],
                                   latent_dim = n_factors,
                                   n_cats = n_cats,
                                   learning_rate = learning_rate,
                                   device = device,
                                   log_interval = log_interval,
                                   correlated_factors = correlated_factors,
                                   steps_anneal = steps_anneal)

                # Train model.
                start = timeit.default_timer()
                vae.run_training(train_loader, train_loader, mc_samples = mc_samples, iw_samples = iw_samples, max_epochs = max_iter)
                stop = timeit.default_timer()

                # Get fitted values.
                loadings = vae.model.loadings.weight.data.numpy()
                ints = vae.model.intercepts.bias.data.numpy()

                # Evaluate model.
                elbo, elbo_components = vae.test(test_loader, mc_samples, iw_samples, print_result = False)

                if np.isnan(elbo):
                    nan_output = True
                    seed += 1
                else:
                    nan_output = False
                    bic = vae.bic(test_loader, 1)

            # Store results.
            cv_res["BIC"][factor_idx].append(bic[0])
            cv_res["log_likelihood"][factor_idx].append(-bic[1])
            cv_res["ELBO"][factor_idx].append(elbo)
            cv_res["loadings"][factor_idx].append(loadings)
            cv_res["run_time"][factor_idx].append(stop - start)

            print("Stored values for P = {:}\tLL = {:.2f}".format(n_factors, -bic[1]))

            seed += 1

        print("Factor loop", cur_fold_idx + 1, "done")
                
    return cv_res