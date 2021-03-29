#!/usr/bin/env Rscript
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Conduct high-dimensional asymptotic simulations.
#
###############################################################################

rm(list = ls())

if(!require(vroom)) {
  install.packages("vroom", repos = "http://cran.us.r-project.org")
  library(vroom)
}

source("cjmle_helper.R")

# Convert tibbles to matrices.
as_matrix = function(x, n = 1){
  if(!tibble::is_tibble(x) ) stop("x must be a tibble")
  y = as.matrix.data.frame(x)
  if (n == 1) {y} else {y[,-seq(from = 1, to = ncol(y), by = n)]}
}

# Simulated data directory.
path = dirname(dirname(getwd()))

# Create directory for saving results.
dir.create(file.path(path, "results/sim_3"), showWarnings = FALSE)

conds     = 1:4
n_factors = 10
n_reps    = 100

for (cond in conds) {
  # Create directories for saving results.
  dir.create(file.path(path, paste0("results/sim_3/cjmle/cond_", toString(cond - 1))), showWarnings = FALSE)
  dir.create(file.path(path, paste0("results/sim_3/cjmle/cond_", toString(cond - 1), "/loadings")), showWarnings = FALSE)
  dir.create(file.path(path, paste0("results/sim_3/cjmle/cond_", toString(cond - 1), "/ints")), showWarnings = FALSE)
  dir.create(file.path(path, paste0("results/sim_3/cjmle/cond_", toString(cond - 1), "/run_times")), showWarnings = FALSE)
  dir.create(file.path(path, paste0("results/sim_3/cjmle/cond_", toString(cond - 1), "/scores")), showWarnings = FALSE)
  
  for (rep in 1:n_reps) {
    # Read and format data.
    dat = as_matrix(vroom(file.path(path, paste0("/data/simulations/sim_3/cond_", toString(cond - 1), "/rep_", toString(rep - 1), ".gz")),
                          delim = ",",
                          col_names = FALSE),
                    n = 2)
  
    # Fit using CJMLE.
    setMIRTthreads(1)
    cjmle_res = mirtjml_expr(dat, n_factors)
    
    # Save results.
    write.table(cjmle_res$A_hat,
                file.path(path, paste0("results/sim_3/cjmle/cond_", toString(cond - 1), "/loadings/loadings_", toString(rep - 1))),
                sep = ",", row.names = FALSE, col.names = FALSE)
    write.table(cjmle_res$d_hat,
                file.path(path, paste0("results/sim_3/cjmle/cond_", toString(cond - 1), "/ints/ints_", toString(rep - 1))),
                sep = ",", row.names = FALSE, col.names = FALSE)
    write.table(cjmle_res$time,
                file.path(path, paste0("results/sim_3/cjmle/cond_", toString(cond - 1), "/run_times/run_time_", toString(rep - 1))),
                sep = ",", row.names = FALSE)
    write.table(cjmle_res$theta_hat,
                file.path(path, paste0("results/sim_3/cjmle/cond_", toString(cond - 1), "/scores/scores_", toString(rep - 1))),
                sep = ",", row.names = FALSE, col.names = FALSE)
    
    gc()
  }
}