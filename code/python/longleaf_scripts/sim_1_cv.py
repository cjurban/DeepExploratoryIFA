#!/usr/bin/env python
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Conduct cross-validation for simulated data sets.
#
###############################################################################

# Import packages.
from __future__ import print_function
import torch
import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
from utils import *
from simulations import *
from cross_validation import *

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
# Cross-Validation
#
###############################################################################

# Get environment loop variable.
i = int(os.environ["SLURM_ARRAY_TASK_ID"])

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

# Set simulation cell parameters and regularization hyperparameter grid.
n_obs_ls       = [500, 1000, 2000, 10000]
iw_samples_ls  = [1, 5, 25]
n_factors_grid = [4, 5, 6]
n_reps         = 100

# Loop across simulation cells.
for rep in range(n_reps):
    for iw_idx, iw_samples in enumerate(iw_samples_ls):
        for obs_idx, n_obs in enumerate(n_obs_ls):
            if (iw_idx == int(np.floor((i % (len(n_obs_ls) * len(iw_samples_ls) * n_reps)) / (len(n_obs_ls) * n_reps))) and
                obs_idx == int(np.floor((i % (n_reps * len(n_obs_ls))) / n_reps)) and
                rep == i % n_reps):
                # Print simulation cell.
                print("Starting 5-fold CV. IW Samples = " + str(int(iw_idx + 1)) + ", N = " + str(int(obs_idx + 1)), ", Rep = " + str(int(rep)))

                # Obtain some simulation cell parameters.
                sim_cell = str(int(iw_idx + 1)) + "_" + str(int(obs_idx + 1))
                n_obs    = n_obs_ls[obs_idx]

                # Make simulation cell directory.
                Path(base_path + "results/sim_1").mkdir(parents = True, exist_ok = True)
                cell_path = base_path + "results/sim_1/cell_" + sim_cell
                Path(cell_path).mkdir(parents = True, exist_ok = True)

                # Set random seeds.
                torch.manual_seed(rep + 1)
                np.random.seed(rep + 1)

                # Simulate data set.
                data = pd.DataFrame(sim_mirt(n_obs = n_obs,
                                    mean = torch.zeros(ref_cov.size(0)),
                                    cov = ref_cov,
                                    loadings = ref_loadings,
                                    intercepts = ref_intercepts,
                                    n_cats = [5]*50)[0].numpy())

                # Try to load the CV results.
                cv_path = cell_path + "/cross_validation"
                Path(cv_path).mkdir(parents = True, exist_ok = True)
                cv_filename = cv_path + "/cv_res_" + str(int(rep))

                try:
                    cv_res = load_obj(cv_filename)
                # If the CV results don't exist, conduct CV.
                except FileNotFoundError:
                    cv_res = cross_validation(n_folds = 5,
                                              n_factors_grid = n_factors_grid,
                                              df = data,
                                              batch_size = hyperparameters["batch_size"],
                                              input_dim = 250,
                                              n_cats = [5]*50,
                                              learning_rate = hyperparameters["lr"],
                                              device = device,
                                              log_interval = sim_log_interval,
                                              steps_anneal = hyperparameters["steps_anneal"],
                                              kwargs = kwargs,
                                              iw_samples = iw_samples)

                    # Save results.
                    save_obj(cv_res, cv_filename)