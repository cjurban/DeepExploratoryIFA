#!/usr/bin/env Rscript
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Code for screeing IPIP FFM data set for careless responders.
#
###############################################################################

rm(list = ls())

if(!require(VIM)) {
  install.packages("VIM", repos = "http://cran.us.r-project.org")
  library(VIM)
}

if(!require(fastDummies)) {
  install.packages("fastDummies", repos = "http://cran.us.r-project.org")
  library(fastDummies)
}

if(!require(PerFit)) {
  install.packages("PerFit", repos = "http://cran.us.r-project.org")
  library(PerFit)
}

if(!require(Hmisc)) {
  install.packages("Hmisc", repos = "http://cran.us.r-project.org")
  library(Hmisc)
}

source("ipip_helper.R")

base_dir = dirname(dirname(getwd()))

set.seed(1)

# Read IPIP data.
# NOTE: "stringsAsFactors = FALSE" reads in numbers as characters so we can drop non-numeric characters.
ipip_data = read.csv(file = file.path(base_dir, "data/ipip_ffm/data-final.csv"),
                     header = TRUE,
                     sep = "\t",
                     stringsAsFactors = FALSE)

# Some fixed variables.
n_cats             = 5
n_factors          = 5
n_items_per_factor = 10
n_items            = 50

# Remove non-numeric characters, then convert all values from character to numeric.
ipip_data[] = lapply(ipip_data, function (col) as.numeric(gsub("[^0-9]", "", col)))

# Drop responses with multiple submissions from same IP address.
ipip_data = ipip_data[ipip_data$IPC == 1, ]

# Replace 0s with NAs.
ipip_data[, 1:n_items][ipip_data[, 1:n_items] == 0] = NA

# Drop observations with all missing survey responses.
ipip_data = ipip_data[-which(apply(ipip_data[, 1:n_items], 1, function(row) all(is.na(row)))), ]

# Make survey time (ST) risk variable.
# NOTE: High risk: < 3 seconds/quesion; low risk: > 1 hour for survey.
ipip_data$ST = ifelse(ipip_data$testelapse < 150, 2, ifelse(ipip_data$testelapse > 3600, 1, 0))

# Make long string (LS) risk variable.
# NOTE: High risk: >= 10 consecutive repeated 1 or 5 responses; low risk: >= 10 consecutive repeated 2 or 4 responses.
ipip_data$LS = apply(ipip_data, 1, function(row) {
                     rles = rle(row[1:n_items])
                     ifelse(sum(c(0, 1, 5) %in% rles$values[which(rles$lengths >= 10)]) > 0, 2, ifelse(sum(c(2, 4) %in% rles$values[which(rles$lengths >= 10)]) > 0, 1, 0))
                     })

# Fix reverse coded items.
reverse_coded_items = c("EXT2", "EXT4", "EXT6", "EXT8", "EXT10", "AGR1", "AGR3", "AGR5", "AGR7",
                        "CSN2", "CSN4", "CSN6", "CSN8", "EST2", "EST4", "OPN2", "OPN4", "OPN6")
ipip_data[, reverse_coded_items] = ((1 - ((ipip_data[, reverse_coded_items] - 1) / 4)) * 4) + 1

# Make Mahalanobis distance (MD) risk variable.
# NOTE: High risk: p < 0.05; low risk: 0.05 < p < 0.1.
cov_mat = cov(ipip_data[, 1:n_items], method = "pearson")
center = colMeans(ipip_data[, 1:n_items])
d2 = mahalanobis(x = ipip_data[, 1:n_items], center = center, cov = cov_mat)
p_vals = pchisq(d2, df = n_items, lower.tail = FALSE)
ipip_data$MD = ifelse(p_vals < 0.05, 2, ifelse((p_vals > 0.05) & (p_vals < 0.1), 1, 0))

# Make a data frame with thermometer-encoded item responses.
therm_enc_ipip_data = therm_enc(ipip_data[, 1:n_items])[, -c(1:n_items)]

# Compute multi-test Guttman error statistics.
gp = multi_test_guttman_stat(therm_enc_ipip_data)

# Bootstrap p-values for multi-test Guttman error statistics.
n_reps      = nrow(ipip_data)
b_reps      = 1000
ip_list     = list() # Item parameter list.
ab_list     = list() # Ability parameter list.
ab_gen_list = list() # Generated ability parameter list.
mat_list    = list() # List of bootstrapped matrices.
for (i in 1:n_factors) {
  df = as.matrix(ipip_data[, seq(from = ((i - 1) * n_items_per_factor) + 1, to = i * n_items_per_factor)])
  
  # Read or compute item and ability parameters.
  if (file.exists(file.path(base_dir, "results/item_parameter_list.RData")) & i == 1) {
    ip_list = readRDS(file.path(base_dir, "results/item_parameter_list.RData"))
  } else if (!file.exists(file.path(base_dir, "results/item_parameter_list.RData"))) {
    ip_list[[i]] = estIP.poly(df, n_cats, NULL)
    if (i == n_factors) {
      saveRDS(ip_list, file = file.path(base_dir, "results/item_parameter_list.RData"))
    }
  }
  if (file.exists(file.path(base_dir, "results/ability_parameter_list.RData")) & i == 1) {
    ab_list = readRDS(file.path(base_dir, "results/ability_parameter_list.RData"))
  } else if (!file.exists(file.path(base_dir, "results/ability_parameter_list.RData"))) {
    ab_list[[i]] = estAb.poly(df, ip_list[[i]][[2]], NULL, "EAP")
    if (i == n_factors) {
      saveRDS(ab_list, file = file.path(base_dir, "results/ability_parameter_list.RData"))
    }
  }

  # Randomly select ability parameters.
  ab_gen_list[[i]]    = sample(ab_list[[i]], size = n_reps, replace = TRUE)
  
  # Compute probabilities of simulees endorsing each answer option.
  P.CRF                   = estP.CRF(n_items_per_factor, n_cats, "GRM", ip_list[[i]][[1]], ab_gen_list[[i]])
  P.CRF.ind               = matrix(1:(n_cats * n_items_per_factor), nrow = n_cats)
  mat_list[[i]]           = matrix(NA, nrow = n_reps, ncol = n_items_per_factor)
  colnames(mat_list[[i]]) = colnames(df)
  
  # Randomly generate item scores.
  for (j in 1:n_items_per_factor) {
    mat_list[[i]][, j] = rMultinom(P.CRF[, ((j - 1) * n_cats + 1) : (j * n_cats)], 1)
  }
}

# Compute bootstrapped multi-test Guttman statistics.
bootstrapped_matrix           = do.call(cbind, mat_list)
therm_enc_bootstrapped_matrix = therm_enc(bootstrapped_matrix)[, -c(1:n_items)]
bootstrapped_gp               = multi_test_guttman_stat(therm_enc_bootstrapped_matrix)

# Compute cutoff values.
b_vec_95 = c()
b_vec_90 = c()
for (i in 1:b_reps)
{
  b_vec_95 = c(b_vec_95, quantile(sample(bootstrapped_gp, size = length(bootstrapped_gp), replace = TRUE), probs = 0.95))
  b_vec_90 = c(b_vec_90, quantile(sample(bootstrapped_gp, size = length(bootstrapped_gp), replace = TRUE), probs = 0.9))
}
cutoff_95 = round(median(b_vec_95), 4)
cutoff_90 = round(median(b_vec_90), 4)

# Make Guttman errors (GP) risk variable.
# NOTE: High risk: gp > cutoff_95; low risk: gp > cutoff_90
ipip_data$GP = ifelse(gp > cutoff_95, 2, ifelse(gp > cutoff_90, 1, 0))

# Drop possible careless responders.
screened_drops = unique(c(which(apply(ipip_data, 1, function(row) sum(row[111:114])) >= 4), # Two or more high risk, four low risk, one high risk two low risk cases covered.
                          which(apply(ipip_data, 1, function(row) length(which(row[111:114] == 1))) == 3),  # Three low risk cases covered.
                          which(apply(ipip_data, 1, function(row) length(which(row[111:114] == 1)) + length(which(row[111:114] == 2))) >= 3))) # One high and two low risk cases covered.
ipip_data = ipip_data[-screened_drops, 1:n_items]

# Convert item responses to dummies.
ipip_data[1:50] = lapply(ipip_data[1:50], function(col) factor(col, levels = 1:5))
ipip_data = dummy_cols(ipip_data[, 1:50])[, -c(1:50)]

write.csv(ipip_data,
          file = file.path(base_dir, "data/ipip_ffm/ipip_ffm_recoded.csv"),
          row.names = FALSE)