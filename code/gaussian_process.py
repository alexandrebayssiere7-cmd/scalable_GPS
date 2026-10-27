import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from regression import GP,GP_ic_matrixfree
import time
from helpers import ic_pivot_fast, woodbury_solve,rbf_covariance,eta_curve,gradient_descent_hyperparams,fit_hyperparams_bfgs,nll,ic_fast,ic_pivot_rbf


# Number of training and test points
n_train = 20000    # e.g. 10,000 observations
n_test  = 10000

# Load the observation (train) and test data
df_obs = pd.read_csv(
    "data/ibex/ibex.csv",
    header=None,
    skiprows=1
).iloc[:n_train]              # <-- several rows, stays a DataFrame

df_tst = pd.read_csv(
    "data/ibex/ibex.csv",
    header=None,
    skiprows=1
).iloc[(n_train+1):(n_train+1+n_test)]

# max and min of the data (8 input variables)
'''lower_bounds = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])
upper_bounds = np.array([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045])'''

# Convert all values to float
df_obs = df_obs.astype(float)
df_tst = df_tst.astype(float)


# ===================== Normalization =====================

# X = all columns except the last one (features)
X   = df_obs.iloc[:, :-1].to_numpy()   # shape (n_train, 8)
X_t = df_tst.iloc[:, :-1].to_numpy()   # shape (n_test, 8)

# Per-dimension normalization with lower_bounds/upper_bounds
'''X_train = (X   - lower_bounds) / (upper_bounds - lower_bounds)
X_test  = (X_t - lower_bounds) / (upper_bounds - lower_bounds)'''
X_train = X/100
X_test  = X_t/100


# y = last column (target)
y_train = df_obs.iloc[:, -1].to_numpy()   # shape (n_train,)
y_test  = df_tst.iloc[:, -1].to_numpy()   # shape (n_test,)




'''

eta = 1e-3  # IC criterion
theta0=np.log(np.array([10,10,10]))
tau_opt, sigma_opt, l_opt, hist = gradient_descent_hyperparams(
    X_train, y_train, eta, theta0,
    lr=0.001,      # start rather small
    max_iter=50,
    verbose=True
)

print("\nEstimated hyperparameters:")
print(f"  tau   = {tau_opt:.4g}")
print(f"  sigma = {sigma_opt:.4g}")
print(f"  ell   = {l_opt:.4g}")

eta= 10**(-7)

Y_mean, Y_std = GP_ic_matrixfree(X_train,X_test,y_train,l_opt,tau_opt,sigma_opt,eta)

plt.figure()
sc = plt.scatter(y_test, Y_mean, c=Y_std, cmap="viridis")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r', label="Ideal")
plt.axis('equal')
plt.colorbar(sc, label="Predictive std")
plt.xlabel("True values")
plt.ylabel("GP predictions")
plt.title("Prediction error vs predicted uncertainty")
plt.legend()
plt.show()


rmse = np.sqrt(np.mean((y_test[:100] - Y_mean[:100]) ** 2))          
print(f"Mean squared error (MSE):{rmse:.4f}")




eta = 1e-3  # IC criterion
theta0=[5,2,0.5]

tau_opt, sigma_opt, l_opt, hist = gradient_descent_hyperparams(
    X_train, y_train, eta, theta0,
    lr=0.001,      # start rather small
    max_iter=200,
    verbose=True
)

history = np.array(hist)   # just in case

plt.figure()
plt.plot(history, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Negative log-likelihood")
plt.title("Convergence of NLL during optimisation")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()




etas= np.logspace(-6,-1,20)

k_list, mse_list, time_list, t_full, mse_full = eta_curve(X_train, y_train, X_test, y_test, 3.02, 133.365, 10.5, etas)


plt.figure()

# MSE_IC(eta) curve, log-log
plt.loglog(etas, mse_list, marker='o', linestyle='-', label='IC MSE vs Full GP MSE',color='g')

# Horizontal line for the full GP MSE (optional)
plt.axhline(mse_full, linestyle='--', label='Full GP MSE',color='r')

plt.xlabel(r'$\eta$')
plt.ylabel('MSE')
plt.title(r'MSE as a function of $\eta$ (log-log)')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()



plt.figure()

# Option 1 : semilogx
plt.semilogx(etas, k_list, marker='o', linestyle='-', label='IC rank')

plt.xlabel(r'$\eta$')
plt.ylabel('Rank $k$')
plt.title(r'Rank of $L$ as a function of $\eta$')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()



plt.figure()

# IC runtime as a function of eta (log x axis)
plt.semilogx(etas, time_list, marker='o', linestyle='-', label='IC runtime')

# Horizontal line for the full Cholesky
plt.axhline(t_full, linestyle='--', label='Full Cholesky runtime')

plt.xlabel(r'$\eta$')
plt.ylabel('Time (s)')
plt.title(r'Runtime as a function of $\eta$')
plt.grid(True, which='both', linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()



plt.figure()
plt.xlabel(r'$rank$')
plt.ylabel('Time (s)')
plt.title(r'Runtime as a function of rank k$')
plt.plot(k_list,time_list)
plt.grid()
plt.show()


plt.figure()
sc = plt.scatter(y_test, Y_mean, c=Y_std, cmap="viridis")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r', label="Ideal")
plt.axis('equal')
plt.colorbar(sc, label="Predictive std")
plt.xlabel("True values")
plt.ylabel("GP predictions")
plt.title("Prediction error vs predicted uncertainty")
plt.legend()
plt.show()


rmse = np.sqrt(np.mean((y_test[:100] - Y_mean[:100]) ** 2))          
print(f"Mean squared error (MSE):{rmse:.4f}")


ci_95 = 1.96 * Y_std  # same size as Y_std

plt.figure()

# Scatter plot with vertical error bars
plt.errorbar(
    y_test,
    Y_mean,
    yerr=ci_95,
    fmt='o',            # 'o' = point marker
    markersize=3,
    ecolor='gray',      # bar color
    elinewidth=0.5,
    capsize=2,          # small caps at the ends
    alpha=0.7,
    label='GP prediction ± 1.96σ'
)

# Ideal line y = x
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    label="Ideal"
)

plt.axis('equal')
plt.xlabel("True values")
plt.ylabel("GP predictions")
plt.title("GP predictions with 95% predictive intervals")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()



total_std = np.sqrt(Y_std**2 + sigma_opt**2)

ci_95 = 1.96 * total_std

plt.figure()
plt.errorbar(
    y_test,
    Y_mean,
    yerr=ci_95,
    fmt='o',
    markersize=3,
    ecolor='gray',
    elinewidth=0.5,
    capsize=2,
    alpha=0.7,
    label='GP prediction ± 1.96 total std'
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--',
    label='Ideal'
)

plt.axis('equal')
plt.xlabel("True values")
plt.ylabel("GP predictions")
plt.title("Predictions with 95% intervals (including noise variance)")
plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


'''


eta_pred = 1e-6  



Y_mean, Y_std = GP_ic_matrixfree(X_train, X_test, y_train, 0.8145, 207.1272, 14.36, eta_pred)

plt.figure()
sc = plt.scatter(y_test, Y_mean, c=Y_std, cmap="viridis")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r', label="Ideal")
plt.axis('equal')
plt.colorbar(sc, label="Predictive std")
plt.xlabel("True values")
plt.ylabel("GP predictions")
plt.title("Prediction error vs predicted uncertainty")
plt.legend()

# --- RMSE ---
errors = Y_mean - y_test
rmse = np.sqrt(np.mean(errors**2))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

plt.show()





'''
def plot_rank_vs_n(
    X_train,
    l,
    tau,
    eta,
    n_list,
    seed=0,
    subsample_mode="first",  # "first" or "random"
):
    """
    Plot of the rank k reached by:
      - ic_pivot_rbf(X, l, tau, eta)  (matrix-free)
      - ic_fast(K, eta)              (needs K)

    Args:
        X_train: (N,d) dataset
        l, tau: hyperparams RBF
        eta: fixed IC tolerance
        n_list: list of sizes n to test
        seed: for reproducibility if subsample_mode="random"
        subsample_mode: "first" (takes the first n) or "random" (random sample)
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    N = X_train.shape[0]
    rng = np.random.default_rng(seed)

    k_pivot_rbf = []
    k_fast = []
    t_pivot_rbf = []
    t_fast = []

    for n in n_list:
        if n > N:
            raise ValueError(f"n={n} > N={N} (dataset size).")

        # ---- Subsample ----
        if subsample_mode == "random":
            idx = rng.choice(N, size=n, replace=False)
            Xn = X_train[idx]
        else:
            Xn = X_train[:n]

        # ---- 1) IC pivot RBF (matrix-free) ----
        t0 = time.perf_counter()
        _, k1 = ic_pivot_rbf(Xn, l, tau, eta)
        t1 = time.perf_counter()

        # ---- 2) IC fast (on explicit K) ----
        # Warning: expensive in memory/time when n is large
        K = rbf_covariance(Xn, Xn, l, tau)
        t2 = time.perf_counter()
        _, k2 = ic_fast(K, eta)
        t3 = time.perf_counter()

        k_pivot_rbf.append(k1)
        k_fast.append(k2)
        t_pivot_rbf.append(t1 - t0)
        t_fast.append(t3 - t2)

        print(f"n={n:6d} | k_pivot_rbf={k1:4d} (t={t1-t0:.3f}s)"
              f" | k_fast={k2:4d} (t={t3-t2:.3f}s)")

    # ---- Rank plot ----
    plt.figure()
    plt.plot(n_list, k_pivot_rbf, marker='o', label="ic_pivot_rbf (matrix-free)")
    plt.plot(n_list, k_fast, marker='s', label="ic_fast (on explicit K)")
    plt.xlabel("Size n (number of points / matrix size)")
    plt.ylabel("Stopping rank k")
    plt.title(f"Stopping rank as a function of n (eta={eta})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Runtime plot (optional but often useful) ----
    plt.figure()
    plt.plot(n_list, t_pivot_rbf, marker='o', label="ic_pivot_rbf runtime")
    plt.plot(n_list, t_fast, marker='s', label="ic_fast runtime (includes K)")
    plt.xlabel("Size n")
    plt.ylabel("Time (s)")
    plt.title(f"Runtime as a function of n (eta={eta})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return (np.array(k_pivot_rbf), np.array(k_fast),
            np.array(t_pivot_rbf), np.array(t_fast))


plot_rank_vs_n(X_train,3.54,301.7,1e-4, [500,1000,2000,3000,4000,5000,7500,10000,15000])

'''


'''
plt.figure()
plt.plot(nll_history, marker='o', linestyle='-')
plt.xlabel("Function evaluations")
plt.ylabel("Negative log-likelihood")
plt.title("Convergence of NLL (L-BFGS-B)")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()
'''






eta=0
def plot_nll_2d(X, y, eta,
                log_tau_range=(0, 4),
                log_l_range=(0, 4),
                n_tau=5,
                n_l=5):

    log_tau_grid = np.linspace(*log_tau_range, n_tau)
    log_l_grid   = np.linspace(*log_l_range, n_l)

    NLL = np.zeros((n_l, n_tau))

    for i, log_l in enumerate(log_l_grid):
        for j, log_tau in enumerate(log_tau_grid):
            theta = np.array([log_tau, log_l])
            NLL[i, j] = nll(theta, X, y, eta)

    Tau, L = np.meshgrid(np.exp(log_tau_grid), np.exp(log_l_grid))

    plt.figure()
    cs = plt.contourf(Tau, L, NLL,
                      levels=20,
                      cmap="viridis")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\ell$")
    plt.title("Negative log-likelihood surface")
    cbar = plt.colorbar(cs)
    cbar.set_label("NLL")
    plt.tight_layout()
    plt.show()



'''
# Example call (with reasonable values):
plot_nll_2d(X_train, y_train, eta,
            log_tau_range=(2,8),
            log_l_range=(-3,8),
            n_tau=10,
            n_l=10)

def plot_nll_like_example(X, y, eta,
                          log_tau_range=(2, 8),
                          log_l_range=(-3, 8),
                          n_tau=80,
                          n_l=80,
                          levels_fill=40,
                          levels_lines=30,
                          theta_star=None):
    """
    Reproduces a plot: contourf + contours + colorbar + red point.
    theta_star: tuple (log_tau*, log_l*) or None
    """

    # grid
    log_tau_grid = np.linspace(*log_tau_range, n_tau)
    log_l_grid   = np.linspace(*log_l_range, n_l)

    NLL = np.zeros((n_l, n_tau))
    for i, log_l in enumerate(log_l_grid):
        for j, log_tau in enumerate(log_tau_grid):
            theta = np.array([log_tau, log_l])
            NLL[i, j] = nll(theta, X, y, eta)   # <-- the NLL function

    # mesh (log axes if needed)
    Tau, L = np.meshgrid(np.exp(log_tau_grid), np.exp(log_l_grid))

    fig, ax = plt.subplots(figsize=(7, 5))

    # 1) colored background
    cf = ax.contourf(Tau, L, NLL, levels=levels_fill, cmap="viridis")

    # 2) contour lines
    cs = ax.contour(Tau, L, NLL, levels=levels_lines,
                    colors="deepskyblue", linewidths=0.8)

    # (optional) labels on some lines
    # ax.clabel(cs, inline=True, fontsize=7)

    # 3) red point (optional)
    if theta_star is not None:
        log_tau_star, log_l_star = theta_star
        ax.plot(np.exp(log_tau_star), np.exp(log_l_star),
                "o", color="red", markersize=6, zorder=5)

    # log axes, as for the contour over (tau, l)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title("Negative log-likelihood")
    ax.set_xlabel(r"$\tau$")      # or "theta1 = log(sigma^2)" for the same
    ax.set_ylabel(r"$\ell$")      # or "theta2 = log(range)"

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("NLL")

    plt.tight_layout()
    plt.show()


# Example call
# theta_star = (log_tau_opt, log_l_opt) if an optimum is known
plot_nll_like_example(X_train, y_train, eta,
                      log_tau_range=(2, 8),
                      log_l_range=(-3, 8),
                      n_tau=30, n_l=30,
                      theta_star=(4.7, -1.5))  # to adjust / or None

'''


def plot_nll_2d_optimal_point(X, y, eta,
                             log_tau_range=(2, 8),
                             log_l_range=(-3, 8),
                             n_tau=80,
                             n_l=80,
                             levels_fill=40,
                             levels_lines=30):

    log_tau_grid = np.linspace(*log_tau_range, n_tau)
    log_l_grid   = np.linspace(*log_l_range, n_l)

    NLL = np.zeros((n_l, n_tau))

    for i, log_l in enumerate(log_l_grid):
        for j, log_tau in enumerate(log_tau_grid):
            theta = np.array([log_tau, log_l])
            NLL[i, j] = nll(theta, X, y, eta)

    # Find the minimum on the grid
    idx_min = np.argmin(NLL)                       # "flattened" index
    i_min, j_min = np.unravel_index(idx_min, NLL.shape)

    log_l_star   = log_l_grid[i_min]
    log_tau_star = log_tau_grid[j_min]

    # Convert to (tau, l) for display (log axes)
    tau_star = np.exp(log_tau_star)
    l_star   = np.exp(log_l_star)

    # meshgrid for the contour
    Tau, L = np.meshgrid(np.exp(log_tau_grid), np.exp(log_l_grid))

    fig, ax = plt.subplots(figsize=(7, 5))

    cf = ax.contourf(Tau, L, NLL, levels=levels_fill, cmap="viridis")
    cs = ax.contour(Tau, L, NLL, levels=levels_lines,
                    colors="deepskyblue", linewidths=0.8)

    # Red point = optimum (min NLL)
    ax.plot(tau_star, l_star, "o", color="red", markersize=7, zorder=5)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\ell$")
    ax.set_title("Negative log-likelihood")

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("NLL")

    plt.tight_layout()
    plt.show()

    print("Optimum on the grid:")
    print(f"  log_tau* = {log_tau_star:.4f}, log_l* = {log_l_star:.4f}")
    print(f"  tau*     = {tau_star:.4f}, l*     = {l_star:.4f}")
    print(f"  NLL_min  = {NLL[i_min, j_min]:.4f}")



plot_nll_2d_optimal_point(X_train, y_train, eta,
                          log_tau_range=(2, 16),
                          log_l_range=(-5, 6),
                          n_tau=40,
                          n_l=40)
