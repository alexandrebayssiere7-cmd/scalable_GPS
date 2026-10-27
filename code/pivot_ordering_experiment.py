import numpy as np
from math import sqrt
import time
import pandas as pd
from helpers import pairwise_distances
import matplotlib.pyplot as plt

df_obs = pd.read_csv("data/borehole/borehole_obs.csv", header=None, skiprows=1).iloc[:1000]
df_tst = pd.read_csv("data/borehole/borehole_tst.csv", header=None, skiprows=1).iloc[:100]

# Convert all values to float
df_obs = df_obs.astype(float)



df_tst = df_tst.astype(float)

# min and max of the data
lower_bounds= np.array([0.05,100,63070,990,63.1,700,1120,9855])
upper_bounds=np.array([0.15,50000,115600,1110,116,820,1680,12045])

# normalization
X = df_obs.iloc[:, :-1].values
#X=[Y[len(Y)-1-i] for i in range(len(Y))]
X_t = df_tst.iloc[:, :-1].values
X_train=(X-lower_bounds)/(upper_bounds-lower_bounds)
X_test=(X_t-lower_bounds)/(upper_bounds-lower_bounds)

def order_by_distance_to_first(X):
    X = list(X)
    lst=[X.pop(0)]
    while len(X)>0:
        ind=0
        m=np.inf
        for i in range (len(X)):
            if np.linalg.norm(X[0]-X[i])<m:
                m=np.linalg.norm(X[0]-X[i])
                ind=i
        lst.append(X.pop(ind))
    return lst

def order_nearest_neighbor_chain(X):
    X = list(X)
    lst=[X.pop(0)]
    while len(X)>0:
        ind=0
        m=np.inf
        for i in range (len(X)):
            if np.linalg.norm(lst[-1]-X[i])<m:
                m=np.linalg.norm(lst[-1]-X[i])
                ind=i
        lst.append(X.pop(ind))
    return lst

def ic_pivot_fast(A, eta):
    """
    Pivoted incomplete Cholesky (optimized).

    Args:
        A (ndarray): Symmetric positive definite matrix (n x n).
        eta (float): Precision parameter in [0, 1].

    Returns:
        ndarray: Matrix L of size (n x k), such that A ≈ L @ L.T
    """
    n = A.shape[0]
    M = np.trace(A)

    L = np.zeros((n, n))
    D = np.diag(A).copy()
    #used = np.zeros(n, dtype=bool)
    S = 0
    k = 0
    J = np.array([i for i in range(n)])

    while S < (1 - eta) * M:
        # Pick the (unused) pivot with the largest diagonal value
        if len(J) == 0:
            break
        ik = J[np.argmax(D[J])]

        if D[ik] <= 0:
            break  # Safety in case the matrix loses positive definiteness

        L[ik, k] = sqrt(D[ik])

        # Update the remaining rows
        J = J[J != ik]
        if k >= 0:
            # Vectorized: L[J, k] = (A[J, ik] - dot(L[J, :k], L[ik, :k])) / L[ik, k]
            L[J, k] = (A[J, ik] - L[J, :k] @ L[ik, :k]) / L[ik, k]
        else:
            L[J, k] = A[J, ik] / L[ik, k]

        # Update D
        D[J] -= L[J, k]**2


        # Update the sum S
        S += L[ik, k]**2 + np.sum(L[J, k]**2)
        k += 1

    return L[:, :k],k




def trace_error(K,L):
    total=abs(np.trace(K-L@L.T))
    return total
def max_error(K,L):
    total=max([max((K-L@L.T)[i]) for i in range(len(K))])
    return total




def error_by_rank(K):
    L1=np.linalg.cholesky(K)
    ic=ic_pivot_fast(K,0)
    L2=ic[0]
    lst_err=[]
    lst_err2=[]
    n=len(L1)
    for k in range(n+1):
        mat=L1[:,:k]
        a=trace_error(K,mat)
        lst_err.append(a)
        mat2=L2[:,:k]
        a2=trace_error(K,mat2)
        lst_err2.append(a2)
    return lst_err,lst_err2

# Function values (observations and tests)

y_train = df_obs.iloc[:, -1].values
y_test = df_tst.iloc[:, -1].values

def covariance_function(x1,x2,l,tau):
        return tau**2*np.exp(-0.5*(pairwise_distances(x1,x2)**2)/l**2)

    # Covariance matrix (joint distribution)

X_train=order_nearest_neighbor_chain(X_train)

l=2.15
tau=1
K11=covariance_function(X_train,X_train,l,tau)
K12=covariance_function(X_train,X_test,l,tau)
K21=covariance_function(X_test,X_train,l,tau)
K22=covariance_function(X_test,X_test,l,tau)

x=[k for k in range(len(K11)+1)]

l1=error_by_rank(K11)[0]
l2=error_by_rank(K11)[1]

plt.plot(x,l1,color='r',label='Cholesky without pivoting')
plt.plot(x,l2,color='b',label='Cholesky with pivoting')
plt.yscale("log")
plt.xlabel("approximation low rank")
plt.ylabel("approximation error")
plt.grid()
plt.title("Order: (n+1)-th element = nearest neighbor of the n-th element")
plt.legend()
plt.show()

X_train=(X-lower_bounds)/(upper_bounds-lower_bounds)
X_train=order_by_distance_to_first(X_train)

l=2.15
tau=1
K11=covariance_function(X_train,X_train,l,tau)
K12=covariance_function(X_train,X_test,l,tau)
K21=covariance_function(X_test,X_train,l,tau)
K22=covariance_function(X_test,X_test,l,tau)

x=[k for k in range(len(K11)+1)]

l1=error_by_rank(K11)[0]
l2=error_by_rank(K11)[1]

plt.plot(x,l1,color='r',label='Cholesky without pivoting')
plt.plot(x,l2,color='b',label='Cholesky with pivoting')
plt.yscale("log")
plt.xlabel("approximation low rank")
plt.ylabel("approximation error")
plt.grid()
plt.title("Order: nearest neighbors of the 1st element")
plt.legend()
plt.show()

