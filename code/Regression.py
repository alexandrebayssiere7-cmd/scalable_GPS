import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from helpers import pairwise_distances,ic_pivot_fast,ic_fast,woodbury_solve,rbf_covariance, ic_pivot_rbf
import time

def GP(xtrain,xtest,ydata,l,tau,sigma,eta):
    # Usage:
    start=time.time()
    K11 = rbf_covariance(xtrain, xtrain, l, tau)
    K12 = rbf_covariance(xtrain, xtest,  l, tau)
    K21 = rbf_covariance(xtest,  xtrain, l, tau)
    K22 = rbf_covariance(xtest,  xtest,  l, tau)
    end=time.time()
    print("build K", end-start)
    # Mean of Y2 given Y1
    start=time.time()
    a=ic_pivot_fast(K11,eta)
    cov_train_inverse=woodbury_solve(a[0],sigma,ydata)
    B=np.dot(K21,cov_train_inverse)
    end=time.time()
    print("build mean", end-start)
    print("rank", a[1])
    # Covariance matrix of Y2 given Y1
    start=time.time()
    COV=K22-K21@woodbury_solve(a[0],sigma,K12)
    end=time.time()
    print("build cov", end-start)
    # Clip negative rounding errors + standard deviation
    diag=np.array([COV[i][i] for i in range (len(COV))])
    for i in range(len(diag)):
        if diag[i]<0:
            diag[i]=0
    std=np.sqrt(diag)

    return B,std


def GP_ic_matrixfree(xtrain, xtest, ydata, l, tau, sigma, eta):
    """
    GP with IC + Woodbury, without explicitly building K11.
    Only K12, K21, K22 are kept (they depend on n_test, assumed small).
    """
    start = time.time()

    # Matrix-free IC on K11 = K(xtrain, xtrain)
    L, k = ic_pivot_rbf(xtrain, l, tau, eta)
    print("IC rank:", k)

    # -- Mean: alpha ≈ (K11 + σ²I)^(-1) ydata --
    alpha = woodbury_solve(L, sigma, ydata)   # (n_train,)

    # K21 = K(xtest, xtrain) for the mean
    K21 = rbf_covariance(xtest, xtrain, l, tau)
    Y_mean = K21 @ alpha

    # -- Predictive covariance --
    # K12 = K(xtrain, xtest)
    K12 = rbf_covariance(xtrain, xtest, l, tau)
    # K22 = K(xtest, xtest)
    K22 = rbf_covariance(xtest, xtest, l, tau)

    # (K11 + σ²I)^(-1) K12 via Woodbury
    invK_K12 = woodbury_solve(L, sigma, K12)   # (n_train, n_test)
    COV = K22 - K21 @ invK_K12                     # (n_test, n_test)

    # Non-negative diagonal + standard deviations
    diag = np.diag(COV).copy()
    diag[diag < 0] = 0.0
    Y_std = np.sqrt(diag)

    end = time.time()
    print("Total GP_ic_matrixfree time:", end - start)

    return Y_mean, Y_std
