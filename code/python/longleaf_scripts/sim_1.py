#!/usr/bin/env python
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Fit simulated data sets.
#
###############################################################################

# Import packages.
from __future__ import print_function
import torch
import numpy as np
import sys
import os
from pathlib import Path
from utils import *
from simulations import *

# If CUDA is available, use it.
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
kwargs = {"num_workers" : 1, "pin_memory" : True} if cuda else {}

# Set log intervals.
sim_log_interval = 75

# Set hyperparameters.
hyperparameters = {
    "batch_size" : 32,
    "lr" : 5e-3,
    "steps_anneal" : 1000
}

base_path = "" # Base directory goes here.

###############################################################################
#
# Simulations
#
###############################################################################

# Load data generating parameters.
ref_loadings   = torch.from_numpy(np.loadtxt(base_path + "data/simulations/loadings.txt",
                                  dtype = float,
                                  delimiter = ",")).float()
ref_intercepts = torch.from_numpy(np.loadtxt(base_path + "data/simulations/intercepts.txt",
                                  dtype = float,
                                  delimiter = ",")).float()
ref_cov        = torch.from_numpy(np.loadtxt(base_path + "data/simulations/lv_cov.txt",
                                  dtype = float,
                                  delimiter = ",")).float()

# Set simulation cell parameters.
n_obs_ls         = [500, 1000, 2000, 10000]
iw_samples_ls    = [1, 5, 25]

# Loop across simulation cells.
i = int(os.environ["SLURM_ARRAY_TASK_ID"])
for iw_idx, iw_samples in enumerate(iw_samples_ls):
    for obs_idx, n_obs in enumerate(n_obs_ls):
        if iw_idx == int(np.floor((i % (len(n_obs_ls) * len(iw_samples_ls))) / len(n_obs_ls))) and obs_idx == i % len(n_obs_ls):
            # Print simulation cell.
            print("Starting replications for simulation cell " + str(int(iw_idx + 1)) + ", " + str(int(obs_idx + 1)))
            
            # Get replication.
            rep = int(np.floor(i / (len(n_obs_ls) * len(iw_samples_ls))))

            # Try to load replication results.
            sim_cell = str(int(iw_idx + 1)) + "_" + str(int(obs_idx + 1))
            cell_path = base_path + "results/sim_1/cell_" + sim_cell
            cell_res_path = cell_path + "/cell_results"
            Path(cell_res_path).mkdir(parents = True, exist_ok = True)
            cell_filename = cell_res_path + "/cell_res_" + str(rep)
            try:
                cell_res = load_obj(cell_filename)
            # If the replication results don't exist, conduct replications.
            except FileNotFoundError:
                cell_res = run_replications(n_reps = 100,
                                            n_obs = n_obs,
                                            cov = ref_cov,
                                            loadings = ref_loadings,
                                            intercepts = ref_intercepts,
                                            batch_size = hyperparameters["batch_size"],
                                            input_dim = 250,
                                            inference_model_dims = [130],
                                            latent_dim = 5,
                                            n_cats = [5]*50,
                                            learning_rate = hyperparameters["lr"],
                                            device = device,
                                            log_interval = sim_log_interval,
                                            steps_anneal = hyperparameters["steps_anneal"],
                                            kwargs = kwargs,
                                            mc_samples = 1,
                                            iw_samples = iw_samples,
                                            rep_subset = [rep],
                                            test_latent_dims = np.arange(2, 9).tolist())

                # Save cell results.
                save_obj(cell_res, cell_filename)