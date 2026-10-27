import numpy as np
from math import sqrt, log
import math
import time
import heapq
import torch


def rbf_covariance(X, Y, l, tau):
        """
        X : (n1, d)
        Y : (n2, d)
        returns an (n1, n2) matrix
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)

        # Squared norms
        X_norm = np.sum(X**2, axis=1)[:, None]   # (n1, 1)
        Y_norm = np.sum(Y**2, axis=1)[None, :]   # (1, n2)

        # Squared distances: ||x_i||^2 + ||y_j||^2 - 2 x_i·y_j
        sq_dists = X_norm + Y_norm - 2.0 * X @ Y.T

        # RBF covariance matrix
        return tau**2 * np.exp(-0.5 * sq_dists / (l**2))
# The 3 nested loops 1, 2 and 3 give a worst-case complexity of O(n^3)





def rbf_kernel_col(X, i, l, tau):
    """
    Column i of the RBF covariance matrix K(X, X) without building K.

    Returns the vector K[:, i] (shape: (n,)) with
    K[j, i] = tau^2 * exp(-||x_j - x_i||^2 / (2 l^2)).
    """
    X = np.asarray(X, dtype=np.float64)
    x_i = X[i]                          # (d,)

    # Differences X_j - x_i for all j
    diff = X - x_i                      # (n, d)
    r2 = np.einsum('ij,ij->i', diff, diff)   # ||x_j - x_i||^2

    return tau**2 * np.exp(-0.5 * r2 / (l**2))



def ic_pivot_rbf(X, l, tau, eta):
    """
    Pivoted incomplete Cholesky for an RBF kernel,
    WITHOUT explicitly building K(X,X).

    X   : (n, d)
    l   : lengthscale
    tau : amplitude
    eta : trace tolerance (0 < eta < 1)

    Returns:
      L : (n, k), such that K ≈ L L^T
      k : effective rank
    """
    X = np.asarray(X, dtype=np.float64)
    n, d = X.shape

    # Diagonal of K for RBF: K_ii = tau^2
    diagK = np.full(n, tau**2, dtype=np.float64)
    M = np.sum(diagK)       # trace(K)

    L = np.zeros((n, n), dtype=np.float64)
    D = diagK.copy()        # residual diagonal
    S = 0.0                 # sum of column norms = Tr(L L^T)
    k = 0
    J = np.arange(n)        # indices still available as pivots

    while S < (1.0 - eta) * M:
        if J.size == 0:
            break

        # Pivot = index with the largest residual diagonal
        ik = J[np.argmax(D[J])]

        if D[ik] <= 0:
            # Stop if positive definiteness degrades numerically
            break

        # Diagonal entry of the new column
        L[ik, k] = np.sqrt(D[ik])

        # Update the set of available indices
        J = J[J != ik]

        # Column of K for pivot ik, computed on the fly
        col_ik = rbf_kernel_col(X, ik, l, tau)   # shape: (n,)

        if k > 0:
            # L[J, k] = (K[J, ik] - L[J, :k] @ L[ik, :k]) / L[ik, k]
            L[J, k] = (col_ik[J] - L[J, :k] @ L[ik, :k]) / L[ik, k]
        else:
            # First column: no rank k-1 yet
            L[J, k] = col_ik[J] / L[ik, k]

        # Update D on the remaining indices
        D[J] -= L[J, k] ** 2

        # Update the sum S = Tr(L L^T)
        S += L[ik, k]**2 + np.sum(L[J, k]**2)
        k += 1

    return L[:, :k], k


def cholesky(A):
    n=len(A)
    L=np.zeros((n,n))
    L[0,0]=sqrt(A[0,0])
    for j in range (1,n):
        L[j,0]=A[0,j]/L[0,0] # fill the 1st column

    for i in range (1,n):  # fill the i-th column                          #1
        s1=0
        for k in range(i):
            s1+=L[i,k]**2
        L[i,i]=sqrt(A[i,i]-s1) # diagonal coefficient
        for j in range(i+1,n):                         #2
            s2=0
            for k in range(i):  # rest of the column below the diagonal                 #3
                s2+=L[i,k]*L[j,k]
            L[j,i]=(A[i,j]-s2)/L[i,i]
    return(L)




def ic_fast(A, eta):
    """
    Incomplete Cholesky decomposition without pivoting (optimized).
    """
    n = A.shape[0]
    M = np.trace(A)

    L = np.zeros((n, n))
    L[0, 0] = sqrt(A[0, 0])
    L[1:, 0] = A[1:, 0] / L[0, 0]

    S = np.sum(L[:, 0]**2)
    i = 1

    while S < (1 - eta) * M and i < n:
        # L[i, i] = sqrt(A[i, i] - sum(L[i, :i]**2))
        s1 = np.dot(L[i, :i], L[i, :i])
        diag_val = A[i, i] - s1
        #if diag_val <= 0:
            #diag_val = 0
            #break  # safety in case A is not SPD
        L[i, i] = sqrt(diag_val)

        if i + 1 < n:
            # Vectorized computation of L[i+1:, i]
            lij = (A[i+1:, i] - L[i+1:, :i] @ L[i, :i].T) / L[i, i]
            L[i+1:, i] = lij
            S += L[i, i]**2 + np.sum(lij**2)
        else:
            S += L[i, i]**2

        i += 1

    return L[:, :i],i


def max_first_column(tab):
  maxval = tab[0][0]
  maxind = 0
  for i in range(1,len(tab)):
    if tab[i][0] > maxval:
      maxval = tab[i][0]
      maxind = i
  return tab[maxind],maxind




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
        if k > 0:
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


def woodbury_solve(L,sigma,K):
    a=L.T@K
    M=np.eye(L.shape[1])+1/sigma**2*(L.T@L)
    b=np.linalg.solve(M,a)
    c=1/sigma**4*L@b
    inv=1/sigma**2*K-c
    return inv




def woodbury_inverse(L,sigma):
    """
    Computes (LL^T + σ²I)^(-1) with the optimized Woodbury formula.

    Uses: (LL^T + σ²I)^(-1) = (1/σ²)(I - L(I + L^T L/σ²)^(-1)L^T/σ²)

    Args:
        L: (n x k) Cholesky approximation matrix
        sigma: Noise parameter

    Returns:
        Inverse matrix (n x n)

    Complexity: O(nk² + k³) instead of O(n²k + k³)
    """
    n = L.shape[0]
    k = L.shape[1]
    sigma_sq = sigma ** 2
    inv_sigma_sq = 1.0 / sigma_sq

    # Compute L^T @ L: O(nk²) instead of O(n²k)
    LTL = L.T @ L

    # I_k + L^T L / σ²: O(k²)
    M = np.eye(k) + LTL * inv_sigma_sq

    # Inverse of M (k x k): O(k³)
    M_inv = np.linalg.inv(M)

    # Final computation: (1/σ²)(I - L @ M_inv @ L^T / σ²)
    # L @ M_inv: O(nk²)
    # (L @ M_inv) @ L^T: O(nk²)
    temp = L @ np.linalg.solve(M,L.T)

    # Result: (1/σ²)(I - temp/σ²)
    result = inv_sigma_sq * (np.eye(n) - temp * inv_sigma_sq)

    return result

'''A = np.random.rand(5,5)
# Build a symmetric positive definite matrix
M = np.dot(A.T, A)
print(M)
print(ic(M,0.99))
print(np.trace(M))
print(woodbury_inverse(ic(M,0.99),0.1))
print(np.linalg.inv(M+0.1**2*np.eye(5)))'''


def inv_lower_triangular(A):
    n=len(A)
    M=np.zeros([n,n])
    for i in range(n):
        M[i][i]=1/A[i][i]
        for j in range(i):
            s=0
            for k in range(i):
                s+=A[i][k]*M[k][j]
            M[i][j]=-1/A[i][i] * s
    return M
'''test_matrix=np.array([[1,0,0,0],[3,60,0,0],[27,69,102,0],[15,11,345,25]])
test_inverse=inv_lower_triangular(test_matrix)
print(np.dot(test_matrix,test_inverse))'''

def inv_cholesky(M):
    L=cholesky(M)
    inv_L=inv_lower_triangular(L)
    res= np.dot(inv_L.T,inv_L)
    return res
'''M=np.array([[2,1,0],[1,1,0],[0,0,3]])
Y=inv_cholesky(M)
print(Y)
print(np.dot(Y,M))'''


### RUNTIME MEASUREMENT ####

def random_spd_matrix(n):
    Q, _ = np.linalg.qr(np.random.rand(n, n))
    D = np.diag(np.random.uniform(1, 10, n))
    A = Q.T @ D @ Q
    return A

def measure_runtime(function, *args):
    start = time.perf_counter()
    result = function(*args)
    end = time.perf_counter()
    return end - start


def distance(a,b):
    if type(a)==float or type(a)==np.int32 or type(a)==np.float64:
        s=(a-b)**2
    else:
        s=0
        for i in range(len(a)):
            s+=(a[i]-b[i])**2
    return sqrt(s)

def pairwise_distances(x1,x2):
    n1=len(x1)
    n2=len(x2)
    M=np.zeros([n1,n2])
    for i in range(n1):
        for j in range(n2):
            M[i][j]=distance(x1[i],x2[j])
    return M


### TRIANGULAR SYSTEM SOLVING ###

def forward_substitution(L, b):
    """Solves Ly = b by forward substitution."""
    n = L.shape[0]
    y = np.zeros(n)

    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    return y

def backward_substitution(LT, y):
    """Solves L^T x = y by backward substitution."""
    n = LT.shape[0]
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(LT[i, i + 1:], x[i + 1:])) / LT[i, i]

    return x


def eta_curve(X_train, y_train, X_test,y_test, l, tau, sigma, etas):
    """
    For each eta, computes:
      - the rank k of the incomplete Cholesky,
      - the MSE of the prediction vs the full (dense) GP.
    """

    n_train = X_train.shape[0]

    # Covariance matrices
    K11 = rbf_covariance(X_train, X_train, l, tau)
    K21 = rbf_covariance(X_test,  X_train, l, tau)

    # --- Reference: "full" GP ---
    start_full=time.time()
    K_full = K11 + sigma**2 * np.eye(n_train)
    alpha_full = np.linalg.solve(K_full, y_train)
    y_pred_full = K21 @ alpha_full
    end_full=time.time()
    t_full=end_full-start_full
    #print(t_full)
    mse_full=np.mean((y_pred_full-y_test)**2)
    k_list = []
    mse_list = []
    time_list = []
    # --- Loop over the eta values ---
    for eta in etas:
        # Pivoted IC
        a=time.time()
        L, k = ic_pivot_fast(K11, eta)

        # (K11 + sigma^2 I)^(-1) y ≈ via Woodbury
        alpha_eta = woodbury_solve(L, sigma, y_train)
        y_pred_eta = K21 @ alpha_eta
        b=time.time()

        t=b-a
        # MSE vs the reference prediction
        mse = np.mean((y_pred_eta - y_pred_full)**2)

        k_list.append(k)
        mse_list.append(mse)
        time_list.append(t)
    return np.array(k_list), np.array(mse_list), np.array(time_list),t_full,mse_full


def rbf_kernel(X1, X2, log_tau, log_ell):
    """
    Isotropic RBF kernel:
    k(x, x') = tau^2 * exp(-||x - x'||^2 / (2 ell^2))
    with tau = exp(log_tau), ell = exp(log_ell).
    """
    tau = torch.exp(log_tau)
    ell = torch.exp(log_ell)

    # X1 : (n1, d), X2 : (n2, d)
    # ||x_i||^2 and ||x_j||^2
    X1_norm = torch.sum(X1**2, dim=1, keepdim=True)  # (n1, 1)
    X2_norm = torch.sum(X2**2, dim=1, keepdim=True).T  # (1, n2)

    # Squared distance matrix
    R2 = X1_norm + X2_norm - 2.0 * X1 @ X2.T          # (n1, n2)
    R2 = torch.clamp(R2, min=0.0)                      # numerical safety

    arg = -0.5 * R2 / (ell**2)
    arg = torch.clamp(arg, min=-50.0, max=0.0)        # avoid exp overflow

    K = (tau**2) * torch.exp(arg)
    return K
'''


def nll(X_train,y_train, eta, theta):
    log_tau, log_sigma, log_l = theta
    tau   = np.exp(log_tau)
    sigma = np.exp(log_sigma)
    l     = np.exp(log_l)

    n = X_train.shape[0]

    K11 = rbf_covariance(X_train, X_train, tau,l)

    L, k = ic_pivot_fast(K11, eta)   # L: (n x k)

    alpha = woodbury_solve(L, sigma, y_train)
    quad = float(y_train.T @ alpha)

    LtL = L.T @ L            # (k x k)
    M = np.eye(k) + (1/sigma**2) * LtL


    R = np.linalg.cholesky(M)

    logdet_M = 2 * np.sum(np.log(np.diag(R)))
    logdet = 2 * n * np.log(sigma) + logdet_M

    logp = -0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
    return -logp

def numerical_gradient(X_train, y_train, eta, theta, eps=1e-3):
    grad = np.zeros_like(theta)
    f0 = nll(X_train, y_train, eta,theta)
    for i in range(len(theta)):
        theta_p = theta.copy()
        theta_p[i] += eps
        fp = nll( X_train, y_train, eta, theta_p)
        grad[i] = (fp - f0) / eps
    return grad

def gradient_descent_hyperparams2(X_train, y_train, eta,
                                 theta0_log,
                                 lr,
                                 max_iter=50,
                                 tol=1e-3,
                                 verbose=True):

    theta = theta0_log.copy()

    for it in range(max_iter):

        g = numerical_gradient( X_train, y_train, eta, theta)

        theta = theta - lr * g


        if verbose:
            tau, sigma, l = np.exp(theta)
            print(f"it {it:3d} |"
                  f"tau={tau:.3f}, sigma={sigma:.3f}, l={l:.3f} | "
                  f"||grad||={np.linalg.norm(g):.3e}")

    log_tau, log_sigma, log_l = theta
    tau, sigma, l = np.exp(log_tau), np.exp(log_sigma), np.exp(log_l)
    return tau, sigma, l

'''


def approx_log_marginal_likelihood2(theta, X_train, y_train, eta):
    """
    theta = [log_tau, log_sigma, log_l]
    """
    log_tau, log_sigma, log_l = theta
    tau   = np.exp(log_tau)
    sigma = np.exp(log_sigma)
    l     = np.exp(log_l)

    n = X_train.shape[0]

    # Covariance matrix K11
    t1=time.time()
    K11 = rbf_covariance(X_train, X_train, l, tau)
    t2=time.time()
    #print("Time to build K11:", t2-t1)

    t1=time.time()

    # Incomplete Cholesky: K ≈ L L^T
    L, k = ic_pivot_rbf(K11, eta)   # L: (n x k)
    tint=time.time()
    #print(tint-t1)
    #print("rank",k)

    # Quadratic term y^T (K + σ²I)^(-1) y
    alpha = woodbury_solve(L, sigma, y_train)
    quad = float(y_train.T @ alpha)
    t2=time.time()
    #print("Time to compute the quadratic term:", t2-t1)

    t1=time.time()
    # Log-det |K + σ²I| ≈ |L L^T + σ²I|
    LtL = L.T @ L            # (k x k)
    M = np.eye(k) + (1/sigma**2) * LtL
    try:
        R = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return 1e-30
    logdet_M = 2 * np.sum(np.log(np.diag(R)))
    logdet = 2 * n * np.log(sigma) + logdet_M
    t2=time.time()
    #print("Time to compute the log det:", t2-t1)
    logp = -0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
    return logp

def approx_log_marginal_likelihood(theta, X_train, y_train, eta):
    """
    theta = [log_tau, log_sigma, log_l]
    Approximates log p(y | theta) via IC + Woodbury, without building K.
    """
    log_tau, log_sigma, log_l = theta
    tau   = np.exp(log_tau)
    sigma = np.exp(log_sigma)
    l     = np.exp(log_l)

    n = X_train.shape[0]

    # --- Matrix-free incomplete Cholesky: K ≈ L L^T ---
    t1 = time.time()
    L, k = ic_pivot_rbf(X_train, l, tau, eta)   # <--- NO K11
    tint = time.time()
    #print("IC (matrix-free) time:", tint - t1, "| rank =", k)

    # --- Quadratic term y^T (K + σ²I)^(-1) y ---
    alpha = woodbury_solve(L, sigma, y_train)   # (n,)
    quad = float(y_train.T @ alpha)

    t2 = time.time()
    #print("Quadratic term time:", t2 - tint)

    # --- Log-det |K + σ²I| ≈ |L L^T + σ²I| ---
    LtL = L.T @ L           # (k x k)
    M = np.eye(k) + (1/sigma**2) * LtL

    try:
        R = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return -1e30  # i.e. a very small log-likelihood

    logdet_M = 2 * np.sum(np.log(np.diag(R)))
    logdet = 2 * n * np.log(sigma) + logdet_M

    t3 = time.time()
    #print("Logdet time:", t3 - t2)

    logp = -0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
    return logp


def loss(theta, X_train, y_train, eta):
    return -approx_log_marginal_likelihood(theta, X_train, y_train, eta)


def numerical_gradient(theta, X_train, y_train, eta, eps=1e-3):
    grad = np.zeros_like(theta)
    f0 = loss(theta, X_train, y_train, eta)
    for i in range(len(theta)):
        theta_p = theta.copy()
        theta_p[i] += eps
        fp = loss(theta_p, X_train, y_train, eta)
        grad[i] = (fp - f0) / eps
    return grad


def gradient_descent_hyperparams(X_train, y_train, eta,
                                 theta0,
                                 lr,
                                 max_iter=50,
                                 tol=1e-3,
                                 verbose=True):

    theta = np.array(theta0, dtype=float)

    history = []

    for it in range(max_iter):
        f = loss(theta, X_train, y_train, eta)

        g = numerical_gradient(theta, X_train, y_train, eta)

        # gradient clipping to avoid huge updates
        g_norm = np.linalg.norm(g)
        if g_norm > 1e3:
            g = g * (1e3 / g_norm)

        # update
        theta = theta - lr * g

        # CLIP the log parameters (hyperparameters in [1e-3, 1e3])
        theta = np.clip(theta, np.log(1e-3), np.log(1e3))

        history.append(f)

        if verbose:
            tau, sigma, l = np.exp(theta)
            print(f"it {it:3d} | loss={f:.4f} | "
                  f"tau={tau:.3f}, sigma={sigma:.3f}, l={l:.3f} | "
                  f"||grad||={g_norm:.3e}")

        if g_norm < tol:
            break

    log_tau, log_sigma, log_l = theta
    tau, sigma, l = np.exp(log_tau), np.exp(log_sigma), np.exp(log_l)
    return tau, sigma, l, history






























'''
def rbf_kernel_torch(X1, X2, log_tau, log_ell):
    """
    Isotropic RBF kernel in torch:
    k(x, x') = tau^2 * exp(-||x - x'||^2 / (2 ell^2))
    with tau = exp(log_tau), ell = exp(log_ell).
    """
    tau = torch.exp(log_tau)
    ell = torch.exp(log_ell)

    # X1 : (n1, d), X2 : (n2, d)
    X1_sq = torch.sum(X1**2, dim=1, keepdim=True)      # (n1, 1)
    X2_sq = torch.sum(X2**2, dim=1, keepdim=True).T    # (1, n2)
    R2 = X1_sq + X2_sq - 2.0 * (X1 @ X2.T)             # (n1, n2)
    R2 = torch.clamp(R2, min=0.0)                      # numerical safety

    arg = -0.5 * R2 / (ell**2)
    arg = torch.clamp(arg, min=-50.0, max=0.0)         # avoid exp overflow

    K = (tau**2) * torch.exp(arg)
    return K


def neg_log_marginal_likelihood_torch(params, X, y, jitter=1e-6):
    """
    Exact NLL of an RBF GP with Gaussian noise:
    params = (log_tau, log_sigma, log_ell)
    X : (n, d), y : (n,)
    """
    log_tau, log_sigma, log_ell = params
    sigma = torch.exp(log_sigma)
    sigma2 = sigma**2

    n = X.shape[0]

    K = rbf_kernel_torch(X, X, log_tau, log_ell)
    K = K + (sigma2 + jitter) * torch.eye(n, dtype=X.dtype, device=X.device)

    # Cholesky
    L = torch.linalg.cholesky(K)

    # alpha = K^{-1} y via Cholesky
    alpha = torch.cholesky_solve(y.unsqueeze(1), L).squeeze(1)

    # y^T K^{-1} y
    quad = torch.dot(y, alpha)

    # log |K|
    logdetK = 2.0 * torch.sum(torch.log(torch.diag(L)))

    # Constant term as a float (math.log, not torch.log), otherwise it breaks
    nll = 0.5 * quad + 0.5 * logdetK + 0.5 * n * math.log(2.0 * math.pi)
    return nll


def fit_gp_hypers_autodiff(X_np, y_np,
                           lr=1e-2,
                           max_iter=200,
                           verbose=True):
    """
    Learns (tau, sigma, ell) by minimizing the exact NLL
    with PyTorch autodiff, on a subsample.

    X_np : (n, d) numpy
    y_np : (n,)   numpy
    """

    # Convert to Torch
    X = torch.from_numpy(X_np)  # (n, d)
    y = torch.from_numpy(y_np)  # (n,)

    # Parameters in log space to stay in R
    init = torch.tensor(
        [0.0,                      # log_tau ~ log(1)
         1,            # log_sigma ~ log(0.1)
         math.log(3)],                     # log_ell ~ log(1)
        dtype=torch.float64
    )

    params = torch.nn.Parameter(init)
    optimizer = torch.optim.Adam([params], lr=lr)

    history = []

    for it in range(1, max_iter + 1):
        optimizer.zero_grad()
        loss = neg_log_marginal_likelihood_torch(params, X, y)
        loss.backward()
        optimizer.step()

        history.append(loss.item())

        if verbose and (it == 1 or it % 10 == 0):
            log_tau, log_sigma, log_ell = params.detach().numpy()
            tau   = math.exp(log_tau)
            sigma = math.exp(log_sigma)
            ell   = math.exp(log_ell)
            print(
                f"iter {it:03d} | nll={loss.item():.4f} | "
                f"tau={tau:.3g}, sigma={sigma:.3g}, ell={ell:.3g}"
            )

    # Back to numpy / float
    log_tau, log_sigma, log_ell = params.detach().numpy()
    tau   = float(math.exp(log_tau))
    sigma = float(math.exp(log_sigma))
    ell   = float(math.exp(log_ell))

    return tau, sigma, ell, np.array(history)
'''



from scipy.optimize import minimize

# Assumes rbf_covariance(X, Y, l, tau) is already defined in this file.




def approx_log_marginal_likelihood(theta, X_train, y_train, eta):
    """
    theta = [log_tau, log_sigma, log_l]
    Approximates log p(y | theta) via IC + Woodbury, without building K.
    """
    sigma=10
    log_tau, log_l = theta
    tau   = np.exp(log_tau)
    l     = np.exp(log_l)

    n = X_train.shape[0]

    # --- Matrix-free incomplete Cholesky: K ≈ L L^T ---
    t1 = time.time()
    L, k = ic_pivot_rbf(X_train, l, tau, eta)   # <--- NO K11
    tint = time.time()
    #print("IC (matrix-free) time:", tint - t1, "| rank =", k)

    # --- Quadratic term y^T (K + σ²I)^(-1) y ---
    alpha = woodbury_solve(L, sigma, y_train)   # (n,)
    quad = float(y_train.T @ alpha)

    t2 = time.time()
    #print("Quadratic term time:", t2 - tint)

    # --- Log-det |K + σ²I| ≈ |L L^T + σ²I| ---
    LtL = L.T @ L           # (k x k)
    M = np.eye(k) + (1/sigma**2) * LtL

    try:
        R = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        return -1e30  # i.e. a very small log-likelihood

    logdet_M = 2 * np.sum(np.log(np.diag(R)))
    logdet = 2 * n * np.log(sigma) + logdet_M

    t3 = time.time()
    #print("Logdet time:", t3 - t2)

    logp = -0.5 * quad - 0.5 * logdet - 0.5 * n * np.log(2 * np.pi)
    return logp

def nll(theta,X_train,y_train,eta):
    return -approx_log_marginal_likelihood(theta,X_train,y_train,eta)


def fit_hyperparams_bfgs(
    X_train,
    y_train,
    eta,
    sigma_fixed=10.0,
    init_log_tau=0.0,
    init_log_l=0.0,
    maxiter=50,
    random_state=0
):
    """
    Estimates (tau, l) by minimizing the exact NLL
    with sigma fixed, using (L-)BFGS-B on a subsample.

    X_train : (N, d)
    y_train : (N,)
    sigma_fixed : float
        Fixed observation noise.
    n_subsample : int
        Size of the subsample used for the optimization.
    init_log_tau, init_log_l : float
        Initial values in log space.
    maxiter : int
        Maximum number of optimizer iterations.
    """

    X_train = np.asarray(X_train, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64).ravel()

    X_sub = X_train
    y_sub = y_train
    history_nll = []

    # --------- 2) Objective function for scipy.minimize ---------
    def objective(log_params):
        val=nll(
            log_params, X_sub, y_sub,eta
        )
        history_nll.append(val)
        return val

    # Initial parameters in log space
    x0 = np.array([init_log_tau, init_log_l], dtype=np.float64)

    # Reasonable bounds help stabilize the optimization
    bounds = [
        (np.log(1e-3), np.log(1000)),   # tau roughly in [1e-3, 1e3]
        (np.log(0.9), np.log(10)),   # l   in [1e-3, 1e2]
    ]

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",    # or "BFGS" if preferred
        bounds=bounds,
        options={"maxiter": maxiter},
    )

    log_tau_opt, log_l_opt = res.x
    tau_opt = float(np.exp(log_tau_opt))
    l_opt = float(np.exp(log_l_opt))

    print("Optimization finished.")
    print("  tau* =", tau_opt)
    print("  l*   =", l_opt)
    print("  sigma (fixed) =", sigma_fixed)
    print("  NLL_min =", res.fun)

    return tau_opt, l_opt, res,np.array(history_nll)
