import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from fonctions_annexes import mat_soustractions,ic_pivot_fast,ic_fast,inv_woodburry2mult,rbf_covariance, ic_pivot_rbf
import time

def GP(xtrain,xtest,ydata,l,tau,sigma,eta):
    # Utilisation :
    deb=time.time()
    K11 = rbf_covariance(xtrain, xtrain, l, tau)
    K12 = rbf_covariance(xtrain, xtest,  l, tau)
    K21 = rbf_covariance(xtest,  xtrain, l, tau)
    K22 = rbf_covariance(xtest,  xtest,  l, tau)
    fin=time.time()
    print("constru K", fin-deb)
    #Moyenne Y2 sachant Y1
    deb=time.time()
    a=ic_pivot_fast(K11,eta)
    cov_train_inverse=inv_woodburry2mult(a[0],sigma,ydata)
    B=np.dot(K21,cov_train_inverse)
    fin=time.time()
    print("constru moy", fin-deb)
    print("rg", a[1])
    #Matrice de covariance Y2 sachant Y1
    deb=time.time()
    COV=K22-K21@inv_woodburry2mult(a[0],sigma,K12)
    fin=time.time()
    print("constru cov", fin-deb)
    #Suppression des valeurs négatives d'arrondi + écart-type
    diag=np.array([COV[i][i] for i in range (len(COV))])
    for i in range(len(diag)):
        if diag[i]<0:
            diag[i]=0
    Etype=np.sqrt(diag)
    
    return B,Etype


def GP_ic_matrixfree(xtrain, xtest, ydata, l, tau, sigma, eta):
    """
    GP avec IC + Woodbury, sans construire K11 explicitement.
    On garde seulement K12, K21, K22 (dépendent de n_test, supposé petit).
    """
    deb = time.time()

    # IC matrix-free sur K11 = K(xtrain, xtrain)
    L, k = ic_pivot_rbf(xtrain, l, tau, eta)
    print("IC rank:", k)

    # -- Moyenne : alpha ≈ (K11 + σ²I)^(-1) ydata --
    alpha = inv_woodburry2mult(L, sigma, ydata)   # (n_train,)

    # K21 = K(xtest, xtrain) pour la mean
    K21 = rbf_covariance(xtest, xtrain, l, tau)
    Y_mean = K21 @ alpha

    # -- Covariance prédictive --
    # K12 = K(xtrain, xtest)
    K12 = rbf_covariance(xtrain, xtest, l, tau)
    # K22 = K(xtest, xtest)
    K22 = rbf_covariance(xtest, xtest, l, tau)

    # (K11 + σ²I)^(-1) K12 via Woodbury
    invK_K12 = inv_woodburry2mult(L, sigma, K12)   # (n_train, n_test)
    COV = K22 - K21 @ invK_K12                     # (n_test, n_test)

    # Diagonale positive + écarts-types
    diag = np.diag(COV).copy()
    diag[diag < 0] = 0.0
    Y_std = np.sqrt(diag)

    fin = time.time()
    print("Total GP_ic_matrixfree time:", fin - deb)

    return Y_mean, Y_std
