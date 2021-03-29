#!/usr/bin/env python
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Gather Simulation 1 cell results into separate dictionaries.
#
###############################################################################

# Import packages.
import os
from pathlib import Path
from utils import *

base_path = "" # Base directory goes here.

# Set simulation cell parameters.
n_obs_ls         = [500, 1000, 2000, 10000]
iw_samples_ls    = [1, 5, 25]

for iw_idx, iw_samples in enumerate(iw_samples_ls):
    for obs_idx, n_obs in enumerate(n_obs_ls):
        # Try to load cell results.
        sim_cell = str(int(iw_idx + 1)) + "_" + str(int(obs_idx + 1))
        cell_path = base_path + "results/sim_1/cell_" + sim_cell
        cell_filename = cell_path + "/cell_results/cell_res"
        try:
            cell_res = load_obj(cell_filename)
        # If the cell results aren't already gathered, do so.
        except FileNotFoundError:
            cell_res_ls = []
            for i in range(100):
                rep_filename = cell_path + "/cell_results/cell_res_" + str(i)
                if os.path.isfile(rep_filename + ".pkl"):
                    cell_res_ls.append({k : [res for res in v if res is not None] for k, v in load_obj(rep_filename).items()})

            # Append results into one dictionary.
            cell_res = {} 
            for k in cell_res_ls[0].keys(): 
                cell_res[k] = [d[k][0] for d in cell_res_ls]

            # Save results.
            save_obj(cell_res, cell_filename)