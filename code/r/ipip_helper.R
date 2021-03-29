#!/usr/bin/env Rscript
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Helper functions for screening IPIP FFM data.
#
###############################################################################

if(!require(data.table)) {
  install.packages("data.table", repos = "http://cran.us.r-project.org")
  library(data.table)
}

if(!require(ltm)) {
  install.packages("ltm", repos = "http://cran.us.r-project.org")
  library(ltm)
}

# A function to thermometer-encode a numeric data frame.
therm_enc = function(df) {
  .data = as.data.table(df)
  char_cols = names(.data)
  for (col_name in char_cols){
    unique_vals = as.character(1:5)
    alloc.col(.data, ncol(.data) + length(unique_vals))
    set(.data, j = paste0(col_name, "_", unique_vals), value = 0L)
    for (i in 1:5) {
      set(.data, i = which(.data[[col_name]] >= i), j = paste0(col_name, "_", unique_vals[i]), value = 1L)
    }
  }
  as.data.frame(.data)
}

# Compute multi-test Guttman errors statistic.
multi_test_guttman_stat = function(df) {
  gp = rep(0, nrow(df))
  for (i in 1:5) {
    sub_df = df[, seq(from = ((i - 1) * 50) + 1, to = i * 50)][c(FALSE, TRUE, TRUE, TRUE, TRUE)]
    pi = unlist(lapply(sub_df, function(col) length(which(col == 1)) / length(col)))
    gp = gp + apply(as.matrix(sub_df[, rev(order(pi))]), 1, function(x){sum(x * cumsum(abs(x - 1)))})
  }
  gp
}

# Estimate polytomous item parameters.
# https://github.com/cran/PerFit/blob/master/R/Accessory.R
estIP.poly = function(matrix, Ncat, ip)
{
  I       = dim(matrix)[2]
  matrix2 = data.frame(apply(matrix + 1, 2, as.factor)) # eliminates item levels with no answers
  if (is.null(ip)) 
  {
    ip = grm(matrix2, constrained = FALSE , IRT.param = TRUE)
    ip.coef = coef(ip)
  } else 
  {
    ip     = as.matrix(ip)
    ip.ltm = cbind(ip[, -ncol(ip)] * ip[, ncol(ip)], ip[, ncol(ip)])
    ip     = grm(matrix2, constrained = FALSE , IRT.param = TRUE, 
                 start.val = unlist(apply(ip.ltm, 1, list), recursive = FALSE), 
                 control = list(iter.qN = 0))
    ip.coef = coef(ip)
  }
  
  # In case NOT all answer categories of all items were used:
  if (is.list(ip.coef)) 
  {
    abs.freqs = apply(matrix, 2, table)
    abs.freqs = lapply(abs.freqs, function(vect) as.numeric(names(vect)))
    tmp       = matrix(NA, nrow = I, ncol = Ncat)
    for (i in 1:I) 
    {
      tmp[i,abs.freqs[[i]][-length(abs.freqs[[i]])]+1] = ip.coef[[i]][-length(abs.freqs[[i]])]
      tmp[i,Ncat] = ip.coef[[i]][length(ip.coef[[i]])]
    }
    ip.coef = tmp
  }
  # 
  list(ip.coef, ip)
}

# Estimate polytomous ability parameters.
# https://github.com/cran/PerFit/blob/master/R/Accessory.R
estAb.poly <- function(matrix, ip.ltm, ability, method)
{
  N       <- dim(matrix)[1]
  matrix2 <- data.frame(apply(matrix + 1, 2, as.factor)) # eliminates item levels with no answers
  if (is.null(ability)) 
  {
    ability <- factor.scores(ip.ltm, resp.patterns = matrix2, method = method)
    ability <- ability$score.dat[, ncol(ability$score.dat) - 1]
  } else
  {
    ability <- as.vector(ability)
  }
  ability
}

# estP.CRF(): Compute P.CRF (polytomous) ----
# https://github.com/cran/PerFit/blob/master/R/Accessory.R
estP.CRF <- function(I, Ncat, model, ip.coef, ability)
{
  N <- length(ability)
  M <- Ncat - 1
  #  Based on GRM:
  if (model == "GRM") 
  {
    # P.ISRF is N x I*M:
    P.ISRF <- t(
      sapply(ability, function(x)
      {
        as.vector(t(1 / (1 + exp(-ip.coef[, ncol(ip.coef)]*(x - ip.coef[, -ncol(ip.coef)])))))
      })
    )
    # Fix for datasets with non-chosen answer options (the NA entries):
    #   1s for NAs in the first item steps
    #   0s for NAs in the last item steps
    #   entry (x+1) for NAs in entry x
    if (sum(is.na(ip.coef)) > 0) 
    {
      first.cols  <- (which(is.na(ip.coef[, 1]))-1) * M + 1; P.ISRF[, first.cols][is.na(P.ISRF[, first.cols])] <- 1
      last.cols   <- which(is.na(ip.coef[, M]))*M; P.ISRF[, last.cols][is.na(P.ISRF[, last.cols])] <- 0
      middle.cols <- sort(which(is.na(t(cbind(rep(0, I), ip.coef[, -c(1, M, Ncat)], rep(0, I))))), decreasing = TRUE)
      for (i in 1:length(middle.cols)){P.ISRF[, middle.cols] <- P.ISRF[, middle.cols + 1]}
    }
    P.CRF <- matrix(NA, nrow = N, ncol = I * Ncat)
    for (i in 1:I) 
    {
      P.ISRF.item <- -cbind(rep(1, N), P.ISRF[, ((i-1)*M+1):(i*M)], rep(0, N))
      P.CRF[, ((i-1)*Ncat+1):(i*Ncat)] <- P.ISRF.item[, (2:(M+2))] - P.ISRF.item[, (1:(M+1))]
    }
  }
  
  # Based on PCM or GPCM:
  if (model == "PCM" | model == "GPCM") 
  {
    lin.it <- t(sapply(ability, function(x){as.vector(t(ip.coef[, ncol(ip.coef)] * (x - ip.coef[, -ncol(ip.coef)])))}))
    lin.it[, is.na(as.vector(t(ip.coef[, -ncol(ip.coef)])))] <- 0 # NAs -> 0 to eliminate these terms from the sums
    P.CRF  <- matrix(NA, nrow = N, ncol = I * Ncat)
    tri    <- matrix(1, nrow = M + 1, ncol = M + 1)
    tri    <- upper.tri(tri, diag = TRUE)
    for (i in 1:I) 
    {
      num <- exp(cbind(rep(0, N), lin.it[, ((i-1)*M+1):(i*M)]) %*% tri)
      P.CRF[, ((i-1) * Ncat+1):(i*Ncat)] <- num / rowSums(num)
    }  
  }
  return(P.CRF)
}