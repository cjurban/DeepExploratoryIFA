#!/usr/bin/env Rscript
#
# Journal: Psychometrika
# Authors: Christopher J. Urban and Daniel J. Bauer
# Affil.: L. L. Thurstone Psychometric Laboratory in the
#         Dept. of Psychology and Neuroscience, UNC-Chapel Hill
# E-mail: cjurban@live.unc.edu
#
# Purpose: Helper functions for CJMLE.
#
###############################################################################

if(!require(mirtjml)) {
  install.packages("mirtjml", repos = "http://cran.us.r-project.org")
  library(mirtjml)
}

if(!require(GPArotation)) {
  install.packages("GPArotation", repos = "http://cran.us.r-project.org")
  library(GPArotation)
}

cjmle_expr_cpp <- function(response, nonmis_ind, theta0, A0, cc, tol, print_proc) {
  .Call('_mirtjml_cjmle_expr_cpp', PACKAGE = 'mirtjml', response, nonmis_ind, theta0, A0, cc, tol, print_proc)
}
standardization_cjmle <- function(theta0, A0, d0){
  theta0 <- as.matrix(theta0)
  A0 <- as.matrix(A0)
  N <- nrow(theta0)
  d1 <- d0 + 1/N * A0 %*% t(theta0) %*% matrix(rep(1,N),N)
  svd_res = svd(theta0 - 1/N * matrix(rep(1,N),N) %*% matrix(rep(1,N),1) %*% theta0)
  theta1 = svd_res$u * sqrt(N)
  A1 = 1/sqrt(N) * A0 %*% svd_res$v %*% diag(svd_res$d, nrow = length(svd_res$d), ncol = length(svd_res$d))
  return(list("theta1"=theta1, "A1"=A1, "d1"=d1))
}
svd_start <- function(response, nonmis_ind, K, tol = 0.01){
  N <- nrow(response)
  J <- ncol(response)
  p_hat <- sum(nonmis_ind) / N / J
  X <- (2 * response - 1) * nonmis_ind
  X[is.na(X)] <- 0
  temp <- svd(X)
  eff_num <- max(1,sum(temp$d >= 2*sqrt(N*p_hat)))
  diagmat <- matrix(0, eff_num, eff_num)
  diag(diagmat) <- temp$d[1:eff_num]
  X <- as.matrix(temp$u[,1:eff_num]) %*% diagmat %*% t(as.matrix(temp$v[,1:eff_num]))
  X <- (X+1) / 2
  X[X<tol/2] <- tol
  X[X>(1-tol/2)] <- 1-tol
  M <- log(X / (1 - X))
  d0 <- colMeans(M)
  temp <- svd(t(t(M) - d0))
  theta0 <- sqrt(N) * as.matrix(temp$u[,1:K])
  A0 <- 1 / sqrt(N) * as.matrix(temp$v[,1:K]) %*% diag(temp$d[1:K], nrow = K, ncol = K)
  return(list("theta0"=theta0, "A0"=A0, "d0"=d0))
}
mirtjml_expr <- function(response, K, theta0 = NULL, A0 = NULL, d0 = NULL, cc = NULL, 
                         tol = 5, print_proc = TRUE){
  N <- nrow(response)
  J <- ncol(response)
  nonmis_ind <- 1 - is.na(response)
  response[is.na(response)] <- 0
  if(is.null(theta0) || is.null(A0) || is.null(d0)){
    t1 <- Sys.time()
    if(print_proc){
      cat("\n", "Initializing... finding good starting point.\n")
    }
    initial_value = svd_start(response, nonmis_ind, K)
    t2 <- Sys.time()
  }
  if(is.null(theta0)){
    theta0 <- initial_value$theta0
  }
  if(is.null(A0)){
    A0 <- initial_value$A0
  }
  if(is.null(d0)){
    d0 <- initial_value$d0
  }
  if(is.null(cc)){
    cc = 5*sqrt(K)
  }
  res <- cjmle_expr_cpp(response, nonmis_ind, cbind(rep(1,N),theta0),
                        cbind(d0,A0), cc, tol, print_proc)
  res_standard <- standardization_cjmle(res$theta[,2:(K+1)], res$A[,2:(K+1)], res$A[,1])
  if(K > 1){
    temp <- GPArotation::geominQ(res_standard$A1, maxit = 100000)
    A_rotated <- temp$loadings
    rotation_M <- temp$Th
    theta_rotated <- res_standard$theta1 %*% rotation_M
  } else{
    A_rotated <- res_standard$A1
    theta_rotated <- res_standard$theta1
  }
  t3 <- Sys.time()
  if(print_proc){
    cat("\n\n", "Precision reached!\n")
    cat("Time spent:\n")
    cat("Find start point: ", as.numeric(difftime(t2, t1, units = "secs"))," second(s) | ", "Optimization: ", as.numeric(difftime(t3, t2, units = "secs"))," second(s)\n")
  }
  return(list("theta_hat" = theta_rotated,
              "A_hat" = A_rotated,
              "d_hat" = res_standard$d1,
              "time" = data.frame(start = as.numeric(difftime(t2, t1, units = "secs")), opt = as.numeric(difftime(t3, t2, units = "secs")))))
}