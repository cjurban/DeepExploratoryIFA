#!/usr/bin/env python
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Functions for conducting simulations.
#
###############################################################################

import torch
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
from scipy.linalg import block_diag
from sklearn.model_selection import train_test_split
import timeit
from functools import reduce
import matplotlib.pyplot as plt
from pylab import setp
import matplotlib.patches as mpatches
from matplotlib.transforms import Affine2D
from utils import *
from helper_layers import *
from read_data import *
from base_class import BaseClass
from mirt_vae import *

# Dummy code a vector.
def dummy_code(vec,
               max_val):
    """
    Args:
        vec     (Tensor): Vector with ordinal entries.
        max_val (int): Maximum possible value for ordinal entries.
    """
    dummy_vec = torch.FloatTensor(vec.size(0), max_val)
    dummy_vec.zero_()
    
    return dummy_vec.scatter_(1, vec, 1)

# Simulate MIRT data.
def sim_mirt(n_obs,
             distribution,
             loadings,
             intercepts,
             n_cats):
    """
    Args:
        n_obs        (int): Number of observations to simulate.
        distribution (Distribution): Distribution of latent variables.
        loadings     (Tensor/array): Factor loadings matrix.
        intercepts   (Tensor/array): Vector of intercepts.
        n_cats       (list of int): List containing number of categories for each observed variable.
    """
    # Define block diagonal loadings matrix.
    ones = [np.ones((n_cat - 1, 1)) for n_cat in n_cats]
    D = torch.from_numpy(block_diag(*ones)).float()
    loadings = torch.mm(D, loadings)
    
    # Sample factor scores.
    scores = distribution.sample(torch.Size([n_obs]))
    
    # Compute cumulative probailities.
    activations = F.linear(scores, loadings, intercepts)
    cum_probs = activations.sigmoid()

    # Compute item response probabilities.
    one_idxs = np.cumsum(n_cats) - 1
    zero_idxs = one_idxs - (np.asarray(n_cats) - 1)
    upper_probs = torch.ones(cum_probs.size(0), cum_probs.size(1) + (len(n_cats)))
    lower_probs = torch.zeros(cum_probs.size(0), cum_probs.size(1) + (len(n_cats)))
    upper_probs[:, torch.from_numpy(np.delete(np.arange(0, upper_probs.size(1), 1), one_idxs))] = cum_probs
    lower_probs[:, torch.from_numpy(np.delete(np.arange(0, lower_probs.size(1), 1), zero_idxs))] = cum_probs
    probs = (upper_probs - lower_probs).clamp(min = 1e-16)
    
    # Simulate data.
    idxs = np.concatenate((np.zeros(1), np.cumsum(n_cats)))
    ranges = [torch.from_numpy(np.arange(int(l), int(u))) for l, u in zip(idxs, idxs[1:])]
    data = torch.cat([dummy_code(torch.multinomial(probs[:, rng], 1), n_cats[i]) for
                      i, rng in enumerate(ranges)], dim = 1)
    
    return data, scores

# Make data generating parameters for simulations.
def make_gen_params(loadings,
                    intercepts,
                    orig_n_cats,
                    new_n_cats,
                    cov = None,
                    factor_mul = 1,
                    item_mul = 1):
    # If data generating parameters are Tensors, convert to Numpy arrays.
    if isinstance(cov, torch.Tensor):
        cov = cov.clone().numpy()
    if isinstance(loadings,  torch.Tensor):
        loadings = loadings.clone().numpy()
    if isinstance(intercepts,  torch.Tensor):
        intercepts = intercepts.clone().numpy()

    # Obtain intercepts.
    intercepts = np.hstack([intercepts.copy() for _ in range(item_mul * factor_mul)])
    n_items = item_mul * factor_mul * loadings.shape[0]
    idxs = np.cumsum([n_cat - 1 for n_cat in ([1] + [orig_n_cats]*n_items)])
    sliced_ints = [intercepts[idxs[i]:idxs[i + 1]] for i in range(len(idxs) - 1)]
    intercepts = torch.Tensor(np.hstack([np.sort(np.random.choice(a, 
                                                                  size = new_n_cats - 1,
                                                                  replace = False),
                                                 axis = None) for a in sliced_ints]))
        
    # Obtain loadings matrix.
    loadings = torch.from_numpy(block_diag(*[loadings.copy().repeat(item_mul, axis = 0) for _ in range(factor_mul)]))

    # Obtain correlation matrix.
    if cov is not None:
        cov = torch.from_numpy(block_diag(*[cov.copy() for _ in range(factor_mul)]))
    else:
        cov = torch.eye(loadings.shape[1])
    
    return loadings, cov, intercepts

# Find converged runs based on congruences with a reference loadings matrix.
def find_converged_runs(loadings_list,
                        ref_loadings,
                        n_skip,
                        threshold = 0.98):
    """
    Args:
        loadings_list (string): List of loadings matrices.
        ref_loadings  (array): Reference loadings for assessing convergence.
        n_skip        (int): Number of folds to skip due to non-convergence.
        threshold     (float): Factor congruence threshold for determining convergence.
    """
    # Check for runs that did not converge.
    converged_runs = [all([c > threshold for c in permuted_congruence(ref_loadings, ldgs, mean = False)]) for
                      ldgs in loadings_list]
    
    # Account for runs we wish to skip.
    if n_skip > 0:
        if not all(converged_runs):
            for elt_idx, elt in enumerate(converged_runs): 
                if n_skip > 0 and not elt: 
                    converged_runs[elt_idx] = True  
                    n_skip -= 1
    
    return converged_runs

# Find reference loadings at each replication.
def find_reference_loadings(cv_res):
    """
    Args:
        cv_res (dict): Dictionary holding cross-validation results.
    """
    congruences = [[sum([permuted_congruence(ldgs1, ldgs2) for ldgs2 in ls]) for ldgs1 in ls] for
                   ls in cv_res["loadings"]]
    max_congruences = [max(ls) for ls in congruences]
    max_idxs = [ls.index(m) for ls, m in zip(congruences, max_congruences)]
    
    return [ls[idx] for ls, idx in zip(cv_res["loadings"], max_idxs)]

# Run simulation replications.
def run_replications(n_reps,
                     n_obs,
                     cov,
                     loadings,
                     intercepts,
                     batch_size,
                     input_dim,
                     inference_model_dims,
                     latent_dim,
                     n_cats,
                     learning_rate,
                     device,
                     log_interval,
                     steps_anneal,
                     kwargs,
                     max_iter = 3000,
                     mc_samples = 1,
                     iw_samples = 1,
                     correlated_factors = False,
                     rep_subset = None,
                     test_latent_dims = []):
    """
    Args:
        n_reps               (int): Number of replications of simulation.
        n_obs                (int): Number of observations to simulate.
        cov                  (Tensor): Factor covariance matrix.
        loadings             (Tensor): Factor loadings matrix.
        intercepts           (Tensor): Vector of intercepts.
        batch_size           (int): Batch size for stochastic gradient descent.
        input_dim            (int): Input vector dimension.
        inference_model_dims (list of int): Inference model neural network layer dimensions.
        latent_dim           (int): Latent vector dimension.
        n_cats               (list of int):  List containing number of categories for each observed variable.
        learning_rate        (float): Learning rate for stochastic gradient descent.
        device               (str): String specifying whether to run on CPU or GPU.
        log_interval         (int): Number of batches to wait before printing progress report.
        steps_anneal         (int): Number of batches over which to linearly anneal KL divergence.
        kwargs               (dict): Dictionary containing CUDA parameters, if used.
        max_iter             (float): Maximum possible number of fitting iterations; may be fractional.
        mc_samples           (int): Number of Monte Carlo samples to draw at each fitting iteration.
        iw_samples           (int): Number of importance-weighted samples to draw at each fitting iteration.
        correlated_factors   (Boolean): Whether or not factors should be correlated.
        rep_subset           (list): List containing subset of total replications to fit.
        test_latent_dims     (list of int): List containing settings for the latent dimension to compute test set likelihoods for.
    """
    # Define dictionary for holding simulation cell results.
    keys = ["BIC", "log_likelihood", "ELBO", "run_time", "loadings",
            "intercepts", "cov", "scores", "seed", "state_dict", "true_scores",
            "test_lls"]
    cell_res = {key : [] for key in keys}
    
    if rep_subset is None:
        rep_subset = [rep for rep in range(n_reps)]

    # Loop across simulation replications.
    for rep in range(n_reps):
        if rep in rep_subset:
            # Set random seeds.
            seed = rep + 1
            torch.manual_seed(seed)
            np.random.seed(seed)

            # Simulate data set.
            data, scores = sim_mirt(n_obs = n_obs,
                                    distribution = dist.MultivariateNormal(torch.zeros(cov.size(0)), cov),
                                    loadings = loadings,
                                    intercepts = intercepts,
                                    n_cats = n_cats)
            train_data, test_data = train_test_split(pd.DataFrame(data.numpy()),
                                                     train_size = 0.8,
                                                     test_size = 0.2,
                                                     random_state = 45)
            
            # Create data loaders.
            sim_loader = torch.utils.data.DataLoader(tensor_dataset(data),
                                                     batch_size = batch_size,
                                                     shuffle = True,
                                                     **kwargs)
            ordered_sim_loader = torch.utils.data.DataLoader(tensor_dataset(data),
                                                             batch_size = batch_size,
                                                             shuffle = False,
                                                             **kwargs)
            sim_train_loader = torch.utils.data.DataLoader(tensor_dataset(torch.from_numpy(train_data.to_numpy())),
                                                           batch_size = batch_size,
                                                           shuffle = True,
                                                           **kwargs)
            sim_test_loader = torch.utils.data.DataLoader(tensor_dataset(torch.from_numpy(test_data.to_numpy())),
                                                          batch_size = batch_size,
                                                          shuffle = True,
                                                          **kwargs)

            # Fit model while ensuring numerical stability.
            nan_output = True
            while nan_output:
                # Re-set random seeds.
                torch.manual_seed(seed)
                np.random.seed(seed)

                # Initialize model.
                vae = MIRTVAEClass(input_dim = input_dim,
                                   inference_model_dims = inference_model_dims,
                                   latent_dim = latent_dim,
                                   n_cats = n_cats,
                                   learning_rate = learning_rate,
                                   device = device,
                                   log_interval = log_interval,
                                   correlated_factors = correlated_factors,
                                   steps_anneal = steps_anneal)

                # Train model.
                start = timeit.default_timer()
                vae.run_training(sim_loader,
                                 sim_loader,
                                 mc_samples = mc_samples,
                                 iw_samples = iw_samples,
                                 max_epochs = max_iter)
                stop = timeit.default_timer()

                # Evaluate model.
                elbo = vae.test(sim_loader,
                                mc_samples = mc_samples,
                                iw_samples = iw_samples,
                                print_result = False)

                if np.isnan(elbo):
                    nan_output = True
                    seed += 1
                else:
                    nan_output = False
                    torch.manual_seed(1)
                    np.random.seed(1)
                    bic = vae.bic(sim_loader, 100)
                    
            # Get fitted values.
            rep_loadings = vae.model.loadings.weight.data.numpy()
            rep_ints = vae.model.intercepts.bias.data.numpy()
            rep_cov = vae.model.cholesky.cov().numpy()
            rep_scores = vae.scores(ordered_sim_loader, mc_samples = 50, iw_samples = iw_samples).numpy()

            # Store results.
            cell_res["BIC"].append(bic[0])
            cell_res["log_likelihood"].append(-bic[1])
            cell_res["ELBO"].append(elbo)
            cell_res["run_time"].append(stop - start)
            cell_res["loadings"].append(rep_loadings)
            cell_res["intercepts"].append(rep_ints)
            cell_res["cov"].append(rep_cov)
            cell_res["scores"].append(rep_scores)
            cell_res["seed"].append(seed)
            cell_res["state_dict"].append(vae.model.state_dict())
            cell_res["true_scores"].append(scores.numpy())
            
            print("Stored values for replication {:}\tLog-likelihood = {:.2f}".format(rep + 1, -bic[1]))
            
            # Conduct analyses for extra latent dimensions.
            extra_lls = []
            for dim in test_latent_dims:
                # Fit model while ensuring numerical stability.
                nan_output = True
                while nan_output:
                    # Re-set random seeds.
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    # Initialize model.
                    vae = MIRTVAEClass(input_dim = input_dim,
                                       inference_model_dims = [int(np.floor((input_dim + 2 * dim) / 2))],
                                       latent_dim = dim,
                                       n_cats = n_cats,
                                       learning_rate = learning_rate,
                                       device = device,
                                       log_interval = log_interval,
                                       correlated_factors = correlated_factors,
                                       steps_anneal = steps_anneal)

                    # Train model.
                    start = timeit.default_timer()
                    vae.run_training(sim_train_loader,
                                     sim_test_loader,
                                     mc_samples = mc_samples,
                                     iw_samples = iw_samples,
                                     max_epochs = max_iter)
                    stop = timeit.default_timer()

                    # Evaluate model.
                    elbo = vae.test(sim_test_loader,
                                    mc_samples = mc_samples,
                                    iw_samples = iw_samples,
                                    print_result = False)

                    if np.isnan(elbo):
                        nan_output = True
                        seed += 1
                    else:
                        nan_output = False
                        torch.manual_seed(1)
                        np.random.seed(1)
                        bic = vae.bic(sim_test_loader, 5000)

                # Store results.
                extra_lls.append(-bic[1])
            cell_res["test_lls"].append(extra_lls)
                
        else:
            {cell_res[k].append(None) for k, _ in cell_res.items()}
        
    return cell_res

# Compute factor score correlations across simulation replications.
def score_cors(cell_res,
               loadings):
    """
    Args:
        cell_res   (dict): Dictionary holding replication results.
        loadings   (Tensor): Factor loadings matrix.
    """
    n_obs  = cell_res["scores"][0].shape[0]
    n_reps = len(cell_res["scores"])
    cor_ls = []
    
    # Loop across simulation replications.
    for rep in range(n_reps):
        col_idxs = hungarian_algorithm(loadings.numpy(), cell_res["loadings"][rep])[0]
        fitted_scores = cell_res["scores"][rep][:, col_idxs].copy()
        true_scores = cell_res["true_scores"][rep].copy()
        
        X = (fitted_scores - fitted_scores.mean(axis = 0)) / fitted_scores.std(axis = 0)
        Y = (true_scores - true_scores.mean(axis = 0)) / true_scores.std(axis = 0)
        cor_ls.append((np.dot(X.T, Y) / X.shape[0]).diagonal())
        
    return cor_ls

# Compute factor score biases across simulation replications.
def score_biases(cell_res,
                 loadings,
                 power):
    """
    Args:
        cell_res   (dict): Dictionary holding replication results.
        loadings   (Tensor): Factor loadings matrix.
        power      (int): Power to raise biases to.
    """
    n_obs   = cell_res["scores"][0].shape[0]
    n_reps  = len(cell_res["scores"])
    bias_ls = []
    
    # Loop across simulation replications.
    cors = score_cors(cell_res, loadings)
    for rep in range(n_reps):
        col_idxs = hungarian_algorithm(loadings.numpy(), cell_res["loadings"][rep])[0]
        fitted_scores = cell_res["scores"][rep][:, col_idxs].copy()
        true_scores = cell_res["true_scores"][rep].copy()
        
        for col_idx, cor in enumerate(cors[rep]):
            if cor < 0:
                fitted_scores[:, col_idx] = -fitted_scores[:, col_idx]
        
        bias_ls.append((fitted_scores - true_scores)**power)
        
    return bias_ls

# Make boxplots of mean per-parameter biases across replications.
def bias_boxplots(cell_res_ls,
                  cov,
                  loadings,
                  intercepts,
                  n_cats,
                  power = 1,
                  ldgs_lims = [-.15, .15],
                  cov_lims = [-.15, .15],
                  int_lims = [-.35, .35]):
    """
    Args:
        cell_res_ls   (list of dict): List of dictionaries holding replication results.
        cov           (Tensor): Factor correlation matrix.
        loadings      (Tensor): Factor loadings matrix.
        intercepts    (Tensor): Vector of intercepts.
        n_cats        (list of int): List containing number of categories for each observed variable.
        power         (int): Power to raise biases to.
        ldgs_lims     (list): y-axis limits for loadings biases.
        cross_lims    (list): y-axis limits for factor correlation biases.
        int_lims      (list): y-axis limits for intercepts biases.
    """
    if power == 1:
        bias_type = "Bias"
    elif power == 2:
        bias_type = "MSE"
    x_ticks = np.arange(len(cell_res_ls) + 1).tolist()[1:]
    nums = [cell_res["scores"][0].shape[0] for cell_res in cell_res_ls]
    x_names = ["N = " + f"{num:,}".replace(',', ' ') for num in nums]
    
    # Lists for saving results.
    loadings_bias_ls = []
    int_bias_ls      = []
    cov_bias_ls      = []
    
    # Compute biases.
    for cell_res in cell_res_ls:
        n_reps = len(cell_res["loadings"])
        loadings_bias_ls.append(reduce(np.add, [permuted_biases(loadings.numpy(), ldgs)**power for
                                                ldgs in cell_res["loadings"]]).flatten() / n_reps)
        int_bias_ls.append(reduce(np.add, [(normalize_ints(ints, ldgs, n_cats) -
                                            normalize_ints(intercepts.numpy(), loadings.numpy(), n_cats))**power for 
                                           ints, ldgs in zip(cell_res["intercepts"], cell_res["loadings"])]).flatten() / n_reps)
        cov_bias_ls.append(reduce(np.add, [cov_biases(cov.numpy(), perm_cov, loadings.numpy(), ldgs)**power for
                                           perm_cov, ldgs in zip(cell_res["cov"], cell_res["loadings"])]).flatten() / n_reps)
        
    # Create boxplots.
    fig = plt.figure()
    fig.set_size_inches(8, 8, forward = True)
    if power == 1:
        fig.suptitle("Bias for Amortized IWVI")
    else:
        fig.suptitle("MSE for Amortized IWVI")
    gs = fig.add_gridspec(4, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:])
    ax3 = fig.add_subplot(gs[2:, 1:3])
    
    plt.subplots_adjust(wspace = 1.1, hspace = 1.1)
    
    flierprops = dict(marker = "o", markerfacecolor = "black", markersize = 2, linestyle = "none")

    # Main loadings boxplots.
    b01 = ax1.boxplot(loadings_bias_ls[0:4],
                         positions = [0.5, 2.5, 4.5, 6.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         showfliers = False,
                         patch_artist = True)
    for b in b01["medians"]:
        setp(b, color = "black")
    for b in b01["boxes"]:
        b.set(facecolor = "white")
    b02 = ax1.boxplot(loadings_bias_ls[4:8],
                         positions = [1, 3, 5, 7],
                         widths = 0.45,
                         flierprops = flierprops,
                         showfliers = False,
                         patch_artist = True)
    for b in b02["medians"]:
        setp(b, color = "black")
    for b in b02["boxes"]:
        b.set(facecolor = "gray")
#              hatch = "//////")
    b03 = ax1.boxplot(loadings_bias_ls[8:12],
                         positions = [1.5, 3.5, 5.5, 7.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         showfliers = False,
                         patch_artist = True)
    for b in b03["medians"]:
        setp(b, color = "white")
    for b in b03["boxes"]:
        b.set(facecolor = "black")
#              hatch = ".....")
    ax1.set_xticks([1, 3, 5, 7])
    ax1.set_xticklabels(x_names, rotation = 30)
    ax1.set_ylim(ldgs_lims)
    ax1.set_ylabel(bias_type)
    if power == 1:
        ax1.axhline(y = 0, color = "gray", linestyle = "--")
    ax1.set_title("Loadings")
    
    # Set legend.
    circ1 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "white", label = "1 IW Sample", edgecolor = "black")
    circ2 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "gray", label = "5 IW Samples", edgecolor = "black") #, hatch = "//////",
    circ3 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "black", label = "25 IW Samples", edgecolor = "black") # hatch = ".....",
    if power == 1:
        ax1.legend(handles = [circ1, circ2, circ3], loc = "upper right", frameon = False, prop = {"size" : 8})
    else:
        ax1.legend(handles = [circ1, circ2, circ3], loc = "upper right", frameon = False, prop = {"size" : 8})
    
    # Factor correlation boxplots.
    b21 = ax2.boxplot(cov_bias_ls[0:4],
                         positions = [0.5, 2.5, 4.5, 6.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         showfliers = False,
                         patch_artist = True)
    for b in b21["medians"]:
        setp(b, color = "black")
    for b in b21["boxes"]:
        b.set(facecolor = "white")
    b22 = ax2.boxplot(cov_bias_ls[4:8],
                         positions = [1, 3, 5, 7],
                         widths = 0.45,
                         flierprops = flierprops,
                         showfliers = False,
                         patch_artist = True)
    for b in b22["medians"]:
        setp(b, color = "black")
    for b in b22["boxes"]:
        b.set(facecolor = "gray")
#              hatch = "//////")
    b23 = ax2.boxplot(cov_bias_ls[8:12],
                         positions = [1.5, 3.5, 5.5, 7.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         showfliers = False,
                         patch_artist = True)
    for b in b23["medians"]:
        setp(b, color = "white")
    for b in b23["boxes"]:
        b.set(facecolor = "black")
#              hatch = ".....")
    ax2.set_xticks([1, 3, 5, 7])
    ax2.set_xticklabels(x_names, rotation = 30)
    ax2.set_ylim(cov_lims)
    ax2.set_ylabel(bias_type)
    if power == 1:
        ax2.axhline(y = 0, color = "gray", linestyle = "--")
    ax2.set_title("Factor Correlations")    

    # Intercepts boxplots.
    b31 = ax3.boxplot(int_bias_ls[0:4],
                         positions = [0.5, 2.5, 4.5, 6.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         showfliers = False,
                         patch_artist = True)
    for b in b31["medians"]:
        setp(b, color = "black")
    for b in b31["boxes"]:
        b.set(facecolor = "white")
    b32 = ax3.boxplot(int_bias_ls[4:8],
                         positions = [1, 3, 5, 7],
                         widths = 0.45,
                         flierprops = flierprops,
                         showfliers = False,
                         patch_artist = True)
    for b in b32["medians"]:
        setp(b, color = "black")
    for b in b32["boxes"]:
        b.set(facecolor = "gray")
#              hatch = "//////")
    b33 = ax3.boxplot(int_bias_ls[8:12],
                         positions = [1.5, 3.5, 5.5, 7.5],
                         widths = 0.45,
                         flierprops = flierprops,
                         showfliers = False,
                         patch_artist = True)
    for b in b33["medians"]:
        setp(b, color = "white")
    for b in b33["boxes"]:
        b.set(facecolor = "black")
#              hatch = ".....")
    ax3.set_xticks([1, 3, 5, 7])
    ax3.set_xticklabels(x_names, rotation = 30)
    ax3.set_ylim(int_lims)
    ax3.set_ylabel(bias_type)
    if power == 1:
        ax3.axhline(y = 0, color = "gray", linestyle = "--")
    ax3.set_title("Intercepts")
    
    return fig

# Make scree plots of BIC across replications.
def scree_plots(cell_res_ls):
    """
    Args:
        cell_res_ls (list of dict): List of dictionaries holding replication results.
    """
    n_reps = len(cell_res_ls[0]["test_lls"])
    x_names = [str(i) for i in np.arange(2, 9).tolist()]
    
    # Extract LLs for scree plots.
    res = [] 
    for i in [3, 7, 11]: 
        test_ll_ls_ls = cell_res_ls[i]["test_lls"].copy() 
        temp_test_ll_ls = [] 
        for j in range(len(test_ll_ls_ls[0])): 
            temp_test_ll_ls.append([test_ll_ls[j] for test_ll_ls in test_ll_ls_ls]) 
        res.append(temp_test_ll_ls)
        
    # Create scree plots.
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    
    trans0 = Affine2D().translate(-0.15, 0.0) + ax.transData
    trans1 = Affine2D().translate(+0.00, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.15, 0.0) + ax.transData

    # Main loadings boxplots.
    qnts = [np.quantile(a, [.25, .50, .75]) for a in res[0]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p1 = ax.errorbar(x_names, meds, yerr = [err1, err2], capsize = 5, fmt = "k-o", transform = trans0, markersize = 3, label = "1 IW Sample")
    qnts = [np.quantile(a, [.25, .50, .75]) for a in res[1]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p2 = ax.errorbar(x_names, meds, yerr = [err1, err2], capsize = 5, fmt = "k--o", transform = trans1, markersize = 3, label = "5 IW Samples")
    qnts = [np.quantile(a, [.25, .50, .75]) for a in res[2]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p3 = ax.errorbar(x_names, meds, yerr = [err1, err2], capsize = 5, fmt = "k:o", transform = trans2, markersize = 3, label = "25 IW Samples")
    ax.axvline(x = 3, color = "gray", linestyle = "--")
    
    ax.set_ylabel("Predicted Approximate Negative Log-Likelihood")
    ax.set_xlabel("Number of Factors")
    ax.set_title("Approximate Log-Likelihood Scree Plots for\nAmortized IWVI, N = 10 000")
    
    # Set legend.
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc = "upper right", frameon = False, prop = {"size" : 10}) 
    
    return fig

# Make boxplots of fitting times across replications.
def time_plot(cell_res_ls,
              y_lims = None):
    """
    Args:
        cell_res_ls   (list of dict): List of dictionaries holding replication results.
        y_lims        (list of float): y-axis limits.
    """
    n_reps = len(cell_res_ls[0]["loadings"])
    x_ticks = np.arange(len(cell_res_ls) + 1).tolist()[1:]
    nums = [500, 1000, 2000, 10000]
    x_names = ["N = " + f"{num:,}".replace(',', ' ') for num in nums]
        
    # Create subplots.
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    
    trans0 = Affine2D().translate(+0.00, 0.0) + ax.transData
    trans1 = Affine2D().translate(-0.1, 0.0) + ax.transData
    trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
    
    qnts = [np.quantile(a, [.25, .50, .75]) for a in [cell_res_ls[0]["run_time"], cell_res_ls[1]["run_time"], cell_res_ls[2]["run_time"], cell_res_ls[3]["run_time"]]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p1 = ax.errorbar(x_names, meds, yerr = [err1, err2], capsize = 5, fmt = "k-o", transform = trans0, markersize = 3, label = "1 IW Sample")
    qnts = [np.quantile(a, [.25, .50, .75]) for a in [cell_res_ls[4]["run_time"], cell_res_ls[5]["run_time"], cell_res_ls[6]["run_time"], cell_res_ls[7]["run_time"]]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p2 = ax.errorbar(x_names, meds, yerr = [err1, err2], capsize = 5, fmt = "k--o", transform = trans1, markersize = 3, label = "5 IW Samples")
    qnts = [np.quantile(a, [.25, .50, .75]) for a in [cell_res_ls[8]["run_time"], cell_res_ls[9]["run_time"], cell_res_ls[10]["run_time"], cell_res_ls[11]["run_time"]]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p3 = ax.errorbar(x_names, meds, yerr = [err1, err2], capsize = 5, fmt = "k:o", transform = trans2, markersize = 3, label = "25 IW Samples")

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc = "upper right", frameon = False, prop = {"size" : 12}) 
    
    ax.set_xticklabels(x_names, rotation = 30)
    if y_lims is not None:
        ax.set_ylim(y_lims)
    ax.set_ylabel("Fitting Time (Seconds)")
    ax.set_title("Fitting Times for Amortized IWVI")
    
    return fig

# Make MSE boxplots for MH-RM comparisons.
def mhrm_boxplots(aiwvi_cell_res_ls,
                  mhrm_cell_res_ls,
                  cov,
                  loadings,
                  intercepts,
                  n_cats,
                  ldgs_lims = [0, .005],
                  cov_lims = [0, .15],
                  int_lims = [0, .35]):
    """
    Args:
        cell_res_ls   (list of dict): List of dictionaries holding replication results.
        cov           (Tensor): Factor correlation matrix.
        loadings      (Tensor): Factor loadings matrix.
        intercepts    (Tensor): Vector of intercepts.
        n_cats        (list of int): List containing number of categories for each observed variable.
        ldgs_lims     (list): y-axis limits for loadings biases.
        cov_lims      (list): y-axis limits for factor correlation biases.
        int_lims      (list): y-axis limits for intercepts biases.
    """
    x_ticks = np.arange(len(aiwvi_cell_res_ls) + 1).tolist()[1:]
    nums = [1000, 2000, 5000, 10000]
    x_names = ["N = " + f"{num:,}".replace(',', ' ') for num in nums]
    
    # Lists for saving results.
    aiwvi_loadings_bias_ls = []
    aiwvi_int_bias_ls      = []
    aiwvi_cov_bias_ls      = []
    mhrm_loadings_bias_ls  = []
    mhrm_int_bias_ls       = []
    mhrm_cov_bias_ls       = []
    
    # Compute biases.
    for cell_res in aiwvi_cell_res_ls:
        n_reps = len(cell_res["loadings"])
        aiwvi_loadings_bias_ls.append(reduce(np.add, [permuted_biases(loadings.numpy(), ldgs)**2 for
                                                      ldgs in cell_res["loadings"]]).flatten() / n_reps)
        aiwvi_int_bias_ls.append(reduce(np.add, [(normalize_ints(ints, ldgs, n_cats) -
                                                  normalize_ints(intercepts.numpy(), loadings.numpy(), n_cats))**2 for 
                                                 ints, ldgs in zip(cell_res["intercepts"], cell_res["loadings"])]).flatten() / n_reps)
        aiwvi_cov_bias_ls.append(reduce(np.add, [cov_biases(cov.numpy(), perm_cov, loadings.numpy(), ldgs)**2 for
                                                 perm_cov, ldgs in zip(cell_res["cov"], cell_res["loadings"])]).flatten() / n_reps)
    for cell_res in mhrm_cell_res_ls:
        n_reps = len(cell_res["loadings"])
        mhrm_loadings_bias_ls.append(reduce(np.add, [permuted_biases(loadings.numpy(), ldgs)**2 for
                                                      ldgs in cell_res["loadings"]]).flatten() / n_reps)
        
        # Handle missing intercepts.
        ints_masks = [(np.isnan(ints - intercepts.numpy()) * 1 - 1) * -1 for ints in cell_res["intercepts"]]
        ints_reps = reduce(np.add, ints_masks)
        ints_ls = cell_res["intercepts"].copy()
        for ints in ints_ls:
            ints[np.isnan(ints)] = 0
        mhrm_int_bias_ls.append(reduce(np.add, [(normalize_ints(ints, ldgs, n_cats) -
                                                  normalize_ints(intercepts.numpy(), loadings.numpy(), n_cats))**2 for 
                                                 ints, ldgs in zip(ints_ls, cell_res["loadings"])]).flatten() / ints_reps)
        
        mhrm_cov_bias_ls.append(reduce(np.add, [cov_biases(cov.numpy(), perm_cov, loadings.numpy(), ldgs)**2 for
                                                perm_cov, ldgs in zip(cell_res["cov"], cell_res["loadings"])]).flatten() / n_reps)
        
    # Create boxplots.
    fig = plt.figure()
    fig.set_size_inches(8, 8, forward = True)
    fig.suptitle("MSE for Amortized IWVI vs. MH-RM")
    gs = fig.add_gridspec(4, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:])
    ax3 = fig.add_subplot(gs[2:, 1:3])
    
    plt.subplots_adjust(wspace = 1.1, hspace = 1.1)
    
    flierprops = dict(marker = "o", markerfacecolor = "black", markersize = 2, linestyle = "none")

    # Main loadings boxplots.
    b01 = ax1.boxplot(aiwvi_loadings_bias_ls,
                      positions = [0.5, 2.5, 4.5, 6.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      showfliers = False,
                      patch_artist = True)
    for b in b01["medians"]:
        setp(b, color = "black")
    for b in b01["boxes"]:
        b.set(facecolor = "white")
    b02 = ax1.boxplot(mhrm_loadings_bias_ls,
                      positions = [1, 3, 5, 7],
                      widths = 0.45,
                      flierprops = flierprops,
                      showfliers = False,
                      patch_artist = True)
    for b in b02["medians"]:
        setp(b, color = "white")
    for b in b02["boxes"]:
        b.set(facecolor = "black")
    ax1.set_xticks([0.75, 2.75, 4.75, 6.75])
    ax1.set_xticklabels(x_names, rotation = 30)
    ax1.set_ylim(ldgs_lims)
    ax1.set_ylabel("MSE")
    ax1.set_title("Loadings")
    
    # Set legend.
    circ1 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "white", label = "Amortized IWVI", edgecolor = "black")
    circ2 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "black", label = "MH-RM", edgecolor = "black")
    ax1.legend(handles = [circ1, circ2], loc = "upper right", frameon = False, prop = {"size" : 10})
    
    # Factor correlation boxplots.
    b21 = ax2.boxplot(aiwvi_cov_bias_ls,
                      positions = [0.5, 2.5, 4.5, 6.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      showfliers = False,
                      patch_artist = True)
    for b in b21["medians"]:
        setp(b, color = "black")
    for b in b21["boxes"]:
        b.set(facecolor = "white")
    b22 = ax2.boxplot(mhrm_cov_bias_ls,
                      positions = [1, 3, 5, 7],
                      widths = 0.45,
                      flierprops = flierprops,
                      showfliers = False,
                      patch_artist = True)
    for b in b22["medians"]:
        setp(b, color = "white")
    for b in b22["boxes"]:
        b.set(facecolor = "black")
    ax2.set_xticks([0.75, 2.75, 4.75, 6.75])
    ax2.set_xticklabels(x_names, rotation = 30)
    ax2.set_ylim(cov_lims)
    ax2.set_ylabel("MSE")
    ax2.set_title("Factor Correlations")    

    # Intercepts boxplots.
    b31 = ax3.boxplot(aiwvi_int_bias_ls,
                      positions = [0.5, 2.5, 4.5, 6.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      showfliers = False,
                      patch_artist = True)
    for b in b31["medians"]:
        setp(b, color = "black")
    for b in b31["boxes"]:
        b.set(facecolor = "white")
    b32 = ax3.boxplot(mhrm_int_bias_ls,
                      positions = [1, 3, 5, 7],
                      widths = 0.45,
                      flierprops = flierprops,
                      showfliers = False,
                      patch_artist = True)
    for b in b32["medians"]:
        setp(b, color = "white")
    for b in b32["boxes"]:
        b.set(facecolor = "black")
    ax3.set_xticks([0.75, 2.75, 4.75, 6.75])
    ax3.set_xticklabels(x_names, rotation = 30)
    ax3.set_ylim(int_lims)
    ax3.set_ylabel("MSE")
    ax3.set_title("Intercepts")
    
    return fig

# Make MSE boxplots for CJMLE comparisons.
def cjmle_boxplots(aiwvi_cell_res_ls,
                   cjmle_cell_res_ls,
                   loadings_ls,
                   intercepts_ls,
                   ldgs_lims = [0, 0.001],
                   int_lims = [0, 0.02]):
    """
    Args:
        cell_res_ls   (list of dict): List of dictionaries holding replication results.
        cov           (Tensor): Factor correlation matrix.
        loadings_ls   (Tensor): List of factor loadings matrices.
        intercepts_ls (Tensor): List of intercept vectors.
        n_cats        (list of int): List containing number of categories for each observed variable.
        ldgs_lims     (list): y-axis limits for loadings biases.
        int_lims      (list): y-axis limits for intercepts biases.
    """
    x_ticks = np.arange(len(aiwvi_cell_res_ls) + 1).tolist()[1:]
    n_obs_ls = [2000, 10000, 50000, 100000]
    n_items_ls = [100, 200, 300, 400]
    x_names = ["N = " + f"{n_obs:,}".replace(',', ' ') + ",\nJ = " + f"{n_items:,}".replace(',', ' ') for 
               n_obs, n_items in zip(n_obs_ls, n_items_ls)]
    item_mul_ls = [1, 2, 3, 4]
    
    # Lists for saving results.
    aiwvi_loadings_bias_ls = []
    aiwvi_int_bias_ls      = []
    cjmle_loadings_bias_ls = []
    cjmle_int_bias_ls      = []
    
    # Compute biases.
    for idx, cell_res in enumerate(aiwvi_cell_res_ls):
        n_reps = len(cell_res["loadings"])
        temp_loadings = loadings_ls[idx]; temp_intercepts = intercepts_ls[idx]
        aiwvi_loadings_bias_ls.append(np.log10(reduce(np.add, [permuted_biases(temp_loadings.numpy(), ldgs)**2 for
                                                             ldgs in cell_res["loadings"]]).flatten() / n_reps))
        aiwvi_int_bias_ls.append(np.log10(reduce(np.add, [(normalize_ints(ints, ldgs, [2]*temp_loadings.shape[0]) -
                                                         normalize_ints(temp_intercepts.numpy(), temp_loadings.numpy(), [2]*temp_loadings.shape[0]))**2 for 
                                                        ints, ldgs in zip(cell_res["intercepts"], cell_res["loadings"])]).flatten() / n_reps))
    for idx, cell_res in enumerate(cjmle_cell_res_ls):
        n_reps = len(cell_res["loadings"])
        temp_loadings = loadings_ls[idx]; temp_intercepts = intercepts_ls[idx]
        cjmle_loadings_bias_ls.append(np.log10(reduce(np.add, [permuted_biases(temp_loadings.numpy(), ldgs)**2 for
                                                             ldgs in cell_res["loadings"]]).flatten() / n_reps))
        cjmle_int_bias_ls.append(np.log10(reduce(np.add, [(normalize_ints(ints, ldgs, [2]*temp_loadings.shape[0]) -
                                                         normalize_ints(temp_intercepts.numpy(), temp_loadings.numpy(), [2]*temp_loadings.shape[0]))**2 for 
                                                        ints, ldgs in zip(cell_res["intercepts"], cell_res["loadings"])]).flatten() / n_reps))
        
    # Create boxplots.
    fig = plt.figure()
    fig.set_size_inches(10, 5, forward = True)
    fig.suptitle("MSE for Amortized IWVI vs. CJMLE")
    gs = fig.add_gridspec(2, 4)
    gs.tight_layout(fig, rect = [0, 0.03, 1, 0.98])
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:])
    
    plt.subplots_adjust(wspace = 1.1, hspace = 0.05)
    
    flierprops = dict(marker = "o", markerfacecolor = "black", markersize = 2, linestyle = "none")

    # Main loadings boxplots.
    b11 = ax1.boxplot(aiwvi_loadings_bias_ls,
                      positions = [0.5, 2.5, 4.5, 6.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      showfliers = False,
                      patch_artist = True)
    for b in b11["medians"]:
        setp(b, color = "black")
    for b in b11["boxes"]:
        b.set(facecolor = "white")
    b12 = ax1.boxplot(cjmle_loadings_bias_ls,
                      positions = [1, 3, 5, 7],
                      widths = 0.45,
                      flierprops = flierprops,
                      showfliers = False,
                      patch_artist = True)
    for b in b12["medians"]:
        setp(b, color = "white")
    for b in b12["boxes"]:
        b.set(facecolor = "black")
    ax1.set_xticks([0.75, 2.75, 4.75, 6.75])
    ax1.set_xticklabels(x_names, rotation = 30)
    ax1.set_yticks(np.arange(-5, 0).tolist())
    ax1.set_yticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$"])
#    ax1.set_ylim(ldgs_lims)
    ax1.set_ylabel("MSE")
    ax1.set_title("Loadings")
    
    # Set legend.
    circ1 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "white", label = "Amortized IWVI", edgecolor = "black")
    circ2 = mpatches.Rectangle((0.5, 0.5), 1, 0.5, facecolor = "black", label = "CJMLE", edgecolor = "black")
    ax1.legend(handles = [circ1, circ2], loc = "upper right", frameon = False, prop = {"size" : 12}) 

    # Intercepts boxplots.
    b21 = ax2.boxplot(aiwvi_int_bias_ls,
                      positions = [0.5, 2.5, 4.5, 6.5],
                      widths = 0.45,
                      flierprops = flierprops,
                      showfliers = False,
                      patch_artist = True)
    for b in b21["medians"]:
        setp(b, color = "black")
    for b in b21["boxes"]:
        b.set(facecolor = "white")
    b22 = ax2.boxplot(cjmle_int_bias_ls,
                      positions = [1, 3, 5, 7],
                      widths = 0.45,
                      flierprops = flierprops,
                      showfliers = False,
                      patch_artist = True)
    for b in b22["medians"]:
        setp(b, color = "white")
    for b in b22["boxes"]:
        b.set(facecolor = "black")
    ax2.set_xticks([0.75, 2.75, 4.75, 6.75])
    ax2.set_xticklabels(x_names, rotation = 30)
    ax2.set_yticks(np.arange(-5, 1).tolist())
    ax2.set_yticklabels(["$10^{-5}$", "$10^{-4}$", "$10^{-3}$", "$10^{-2}$", "$10^{-1}$", "$1$"])
#    ax2.set_ylim(int_lims)
    ax2.set_ylabel("MSE")
    ax2.set_title("Intercepts")
    
    return fig

# Make line plots of fitting times for comparison studies.
def comparison_time_plots(aiwvi_cell_res_ls,
                          comp_cell_res_ls,
                          x_names,
                          lab1,
                          lab2,
                          title,
                          y_lims = None):
    """
    Args:
    """
    x_ticks = np.arange(len(aiwvi_cell_res_ls) + 1).tolist()[1:]
        
    # Create subplots.
    fig, ax = plt.subplots(constrained_layout = True)
    fig.set_size_inches(5, 5, forward = True)
    
    trans0 = Affine2D().translate(+0.1, 0.0) + ax.transData
    trans1 = Affine2D().translate(+0.00, 0.0) + ax.transData
    
    qnts = [np.quantile(a, [.25, .50, .75]) for a in [cell_res["run_time"] for cell_res in aiwvi_cell_res_ls]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p1 = ax.errorbar(x_names, meds, yerr = [err1, err2], capsize = 5, fmt = "k-o", transform = trans0, markersize = 3, label = lab1)
    qnts = [np.quantile(a, [.25, .50, .75]) for a in [cell_res["run_time"] for cell_res in comp_cell_res_ls]]
    meds = [p[1] for p in qnts]
    err1 = [abs(p[0] - p[1]) for p in qnts]
    err2 = [abs(p[1] - p[2]) for p in qnts]
    p2 = ax.errorbar(x_names, meds, yerr = [err1, err2], capsize = 5, fmt = "k--o", transform = trans1, markersize = 3, label = lab2)

    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] for h in handles]
    ax.legend(handles, labels, loc = "upper left", frameon = False, prop = {"size" : 12}) 
    
    ax.set_xticklabels(x_names, rotation = 30)
    ax.set_ylabel("Fitting Time (Seconds)")
    if y_lims is not None:
        ax.set_ylim(y_lims)
    ax.set_title(title)
    
    return fig