#!/usr/bin/env Rscript
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Rotate CJMLE comparison factor loadings.
#
###############################################################################

rm(list = ls())

if(!require(GPArotation)) {
  install.packages("GPArotation", repos = "http://cran.us.r-project.org")
  library(GPArotation)
}

base_dir = dirname(dirname(getwd()))

for (cond in 1:4) {
  # Create directories for saving results.
  dir.create(file.path(base_dir, paste0("results/sim_3/aiwvi/cond_", toString(cond - 1)), "rotated_loadings"), showWarnings = FALSE)
  dir.create(file.path(base_dir, paste0("results/sim_3/aiwvi/cond_", toString(cond - 1)), "covs"), showWarnings = FALSE)
  
  # Rotate factor loadings and save results.
  loadings_filenames = unlist(lapply(1:100, function(i) {file.path(base_dir,
                                                                   paste0("results/sim_3/aiwvi/cond_", toString(cond - 1)),
                                                                   "loadings",
                                                                   paste0("loadings_", i - 1, ".txt"))}))
  run_time_filenames = unlist(lapply(1:100, function(i) {file.path(base_dir,
                                                                   paste0("results/sim_3/aiwvi/cond_", toString(cond - 1)),
                                                                   "run_times",
                                                                   paste0("run_time_", i - 1, ".txt"))}))
  lapply(seq_along(loadings_filenames), function(i) {
    loadings = as.matrix(read.table(loadings_filenames[i], header = FALSE))
    run_time = read.table(run_time_filenames[i], header = TRUE)
    if (is.na(run_time[1, 1])) {run_time = read.table(run_time_filenames[i], header = FALSE)}
    t1 <- Sys.time()
    rotated = GPArotation::geominQ(loadings, maxit = 100000)
    t2 <- Sys.time()
    write.table(rotated[[1]], file.path(base_dir, paste0("results/sim_3/aiwvi/cond_", toString(cond - 1), "/rotated_loadings/rotated_loadings_", i - 1, ".txt")), sep = " ", quote = FALSE, row.names = FALSE, col.names = FALSE)
    write.table(rotated[[2]], file.path(base_dir, paste0("results/sim_3/aiwvi/cond_", toString(cond - 1), "/covs/cov_", i - 1, ".txt")), sep = " ", quote = FALSE, row.names = FALSE, col.names = FALSE)
    write.table(data.frame(opt = run_time[1, 1], rot = as.numeric(difftime(t2, t1, units = "secs"))), run_time_filenames[i], sep = " ", quote = FALSE, row.names = FALSE)
  })
}