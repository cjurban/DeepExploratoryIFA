#!/usr/bin/env Rscript
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Conduct classical asymptotic simulations.
#
###############################################################################

rm(list = ls())

if(!require(vroom)) {
  install.packages("vroom", repos = "http://cran.us.r-project.org")
  library(vroom)
}

if(!require(mirt)) {
  install.packages("mirt", repos = "http://cran.us.r-project.org")
  library(mirt)
}

if(!require(GPArotation)) {
  install.packages("GPArotation", repos = "http://cran.us.r-project.org")
  library(GPArotation)
}

# Convert tibbles to matrices.
as_matrix = function(x, n = 1){
  if(!tibble::is_tibble(x) ) stop("x must be a tibble")
  y = as.matrix.data.frame(x)
  if (n == 1) {y} else {y[,-seq(from = 1, to = ncol(y), by = n)]}
}

# Simulated data directory.
path = dirname(dirname(getwd()))

# Create directory for saving results.
dir.create(file.path(path, "results/sim_2"), showWarnings = FALSE)

conds     = 1:4
n_reps    = 100
n_factors = 10

for (cond in conds) {
  
  # Create directories for saving results.
  dir.create(file.path(path, "results/sim_2/"), showWarnings = FALSE)
  dir.create(file.path(path, "results/sim_2/mhrm"), showWarnings = FALSE)
  dir.create(file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1))), showWarnings = FALSE)
  dir.create(file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1), "/loadings")), showWarnings = FALSE)
  dir.create(file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1), "/ints")), showWarnings = FALSE)
  dir.create(file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1), "/run_times")), showWarnings = FALSE)
  dir.create(file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1), "/covs")), showWarnings = FALSE)
  dir.create(file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1), "/lls")), showWarnings = FALSE)
  
  # cond 0 rep 79 didn't compress correctly; needs to be loaded from CSV.
  if (cond == 1){n = 4} else {n = 1}
  
  for (rep in n:n_reps) {
    if (cond == 1 & rep == 79){next}
    # Read and format data.
    dat = as_matrix(vroom(file.path(path, paste0("/data/simulations/sim_2/cond_", toString(cond - 1), "/rep_", toString(rep - 1), ".gz")),
                          delim = ",",
                          col_names = FALSE),
                    n = 1)
   dat = cbind.data.frame(lapply(seq(from = 5, to = ncol(dat), by = 5),
                                 function(i) {rowSums(sweep(dat[, (i - 4):i], MARGIN = 2, c(1, 2, 3, 4, 5), `*`))})); colnames(dat) = 1:ncol(dat)
    
    ###############################################################################
    #
    # Fit using MH-RM.
    #
    ###############################################################################
    mhrm_res = mirt(dat, model = n_factors, itemtype = "graded", method = "MHRM")
    
    # Extract MH-RM results.
    unrot_ldgs = as.matrix(do.call(rbind, lapply(coef(mhrm_res)[1:length(coef(mhrm_res)) - 1],
                                                 function(parvec) {data.frame(parvec)[grepl("a", names(data.frame(parvec)))]})))
    t1 = Sys.time()
    rotated = GPArotation::geominQ(unrot_ldgs, maxit = 100000)
    t2 = Sys.time()
    loadings = rotated[[1]]
    cov = rotated[[2]]
    ints = unlist(lapply(coef(mhrm_res)[1:length(coef(mhrm_res)) - 1],
                         function(parvec) {data.frame(parvec)[grepl("d", names(data.frame(parvec)))]}))
    time = t(data.frame(extract.mirt(mhrm_res, "time"))); time = cbind(time, rot = as.numeric(difftime(t2, t1, units = "secs")))
    ll = extract.mirt(mhrm_res, "logLik")
    
    # Save results.
    write.table(loadings,
                file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1), "/loadings/loadings_", toString(rep - 1))),
                sep = ",", row.names = FALSE, col.names = FALSE)
    write.table(cov,
                file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1), "/covs/cov_", toString(rep - 1))),
                sep = ",", row.names = FALSE, col.names = FALSE)
    write.table(t(data.frame(ints)),
                file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1), "/ints/ints_", toString(rep - 1))),
                sep = ",", row.names = FALSE, col.names = FALSE)
    write.table(time,
                file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1), "/run_times/run_time_", toString(rep - 1))),
                sep = ",", row.names = FALSE)
    write.table(ll,
                file.path(path, paste0("results/sim_2/mhrm/cond_", toString(cond - 1), "/lls/ll_", toString(rep - 1))),
                sep = ",", row.names = FALSE, col.names = FALSE)

  }
}