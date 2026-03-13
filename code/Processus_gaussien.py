import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Regression import GP,GP_ic_matrixfree
import time
from fonctions_annexes import ic_pivot_fast, inv_woodburry2mult,rbf_covariance,courbe_eta,gradient_descent_hyperparams,fit_hyperparams_bfgs,nll,ic_fast,ic_pivot_rbf


# Nombre de points d'apprentissage et de test
n_train = 20000    # par ex. 10 000 observations
n_test  = 10000     # comme tu l'avais mis

# Chargement des données d'observation (train) et de test
df_obs = pd.read_csv(
    "data/bouquetins/bouquetins.csv",
    header=None,
    skiprows=1
).iloc[:n_train]              # <-- plusieurs lignes, reste un DataFrame

df_tst = pd.read_csv(
    "data/bouquetins/bouquetins.csv",
    header=None,
    skiprows=1
).iloc[(n_train+1):(n_train+1+n_test)]

# max et min des données (8 variables d'entrée)
'''minimaux = np.array([0.05, 100, 63070, 990, 63.1, 700, 1120, 9855])
maximaux = np.array([0.15, 50000, 115600, 1110, 116, 820, 1680, 12045])'''

# Convertir toutes les valeurs en float
df_obs = df_obs.astype(float)
df_tst = df_tst.astype(float)


# ===================== Normalisation =====================

# X = toutes les colonnes sauf la dernière (features)
X   = df_obs.iloc[:, :-1].to_numpy()   # shape (n_train, 8)
X_t = df_tst.iloc[:, :-1].to_numpy()   # shape (n_test, 8)

# Normalisation sur chaque dimension avec minimaux/maximaux
'''X_train = (X   - minimaux) / (maximaux - minimaux)
X_test  = (X_t - minimaux) / (maximaux - minimaux)'''
X_train = X/100
X_test  = X_t/100


# y = dernière colonne (cible)
y_train = df_obs.iloc[:, -1].to_numpy()   # shape (n_train,)
y_test  = df_tst.iloc[:, -1].to_numpy()   # shape (n_test,)




'''

eta = 1e-3  # ton critère d’IC
theta0=np.log(np.array([10,10,10]))
tau_opt, sigma_opt, l_opt, hist = gradient_descent_hyperparams(
    X_train, y_train, eta, theta0,
    lr=0.001,      # commence plutôt petit
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
print(f"Erreur quadratique moyenne (MSE):{rmse:.4f}")




eta = 1e-3  # ton critère d’IC
theta0=[5,2,0.5]

tau_opt, sigma_opt, l_opt, hist = gradient_descent_hyperparams(
    X_train, y_train, eta, theta0,
    lr=0.001,      # commence plutôt petit
    max_iter=200,
    verbose=True
)

history = np.array(hist)   # au cas où

plt.figure()
plt.plot(history, marker='o', linestyle='-')
plt.xlabel("Iteration")
plt.ylabel("Negative log-likelihood")
plt.title("Convergence of NLL during optimisation")
plt.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()
plt.show()




etas= np.logspace(-6,-1,20)

k_list, mse_list, time_list, t_full, mse_full = courbe_eta(X_train, y_train, X_test, y_test, 3.02, 133.365, 10.5, etas)


plt.figure()

# Courbe MSE_IC(eta) en log-log
plt.loglog(etas, mse_list, marker='o', linestyle='-', label='IC MSE vs Full GP MSE',color='g')

# Ligne horizontale pour la MSE du full GP (optionnel)
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

# Temps IC en fonction de eta (x en log)
plt.semilogx(etas, time_list, marker='o', linestyle='-', label='IC runtime')

# Ligne horizontale pour la Cholesky complète
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
print(f"Erreur quadratique moyenne (MSE):{rmse:.4f}")


ci_95 = 1.96 * Y_std  # même taille que Y_std

plt.figure()

# Nuage de points avec barres d'erreur verticales
plt.errorbar(
    y_test,
    Y_mean,
    yerr=ci_95,
    fmt='o',            # 'o' = marqueur pour les points
    markersize=3,
    ecolor='gray',      # couleur des barres
    elinewidth=0.5,
    capsize=2,          # petites barres au bout
    alpha=0.7,
    label='GP prediction ± 1.96σ'
)

# Ligne idéale y = x
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
    subsample_mode="first",  # "first" ou "random"
):
    """
    Plot du rang k atteint par:
      - ic_pivot_rbf(X, l, tau, eta)  (matrix-free)
      - ic_fast(K, eta)              (nécessite K)

    Args:
        X_train: (N,d) dataset
        l, tau: hyperparams RBF
        eta: tolérance IC fixée
        n_list: liste des tailles n à tester
        seed: pour reproductibilité si subsample_mode="random"
        subsample_mode: "first" (prend les n premiers) ou "random" (échantillon aléatoire)
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
            raise ValueError(f"n={n} > N={N} (taille du dataset).")

        # ---- Sous-échantillon ----
        if subsample_mode == "random":
            idx = rng.choice(N, size=n, replace=False)
            Xn = X_train[idx]
        else:
            Xn = X_train[:n]

        # ---- 1) IC pivot RBF (matrix-free) ----
        t0 = time.perf_counter()
        _, k1 = ic_pivot_rbf(Xn, l, tau, eta)
        t1 = time.perf_counter()

        # ---- 2) IC fast (sur K explicite) ----
        # Attention: coûteux en mémoire/temps quand n grand
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

    # ---- Plot rang ----
    plt.figure()
    plt.plot(n_list, k_pivot_rbf, marker='o', label="ic_pivot_rbf (matrix-free)")
    plt.plot(n_list, k_fast, marker='s', label="ic_fast (sur K explicite)")
    plt.xlabel("Taille n (nombre de points / taille de la matrice)")
    plt.ylabel("Rang k d'arrêt")
    plt.title(f"Rang d'arrêt en fonction de n (eta={eta})")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---- Plot temps (optionnel mais souvent utile) ----
    plt.figure()
    plt.plot(n_list, t_pivot_rbf, marker='o', label="Temps ic_pivot_rbf")
    plt.plot(n_list, t_fast, marker='s', label="Temps ic_fast (inclut K)")
    plt.xlabel("Taille n")
    plt.ylabel("Temps (s)")
    plt.title(f"Temps en fonction de n (eta={eta})")
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
# Exemple d'appel (prise de valeurs raisonnables):
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
    Reproduit un plot: contourf + contours + colorbar + point rouge.
    theta_star: tuple (log_tau*, log_l*) ou None
    """

    # grille
    log_tau_grid = np.linspace(*log_tau_range, n_tau)
    log_l_grid   = np.linspace(*log_l_range, n_l)

    NLL = np.zeros((n_l, n_tau))
    for i, log_l in enumerate(log_l_grid):
        for j, log_tau in enumerate(log_tau_grid):
            theta = np.array([log_tau, log_l])
            NLL[i, j] = nll(theta, X, y, eta)   # <-- ta fonction

    # maillage (axes en log si tu veux)
    Tau, L = np.meshgrid(np.exp(log_tau_grid), np.exp(log_l_grid))

    fig, ax = plt.subplots(figsize=(7, 5))

    # 1) fond coloré
    cf = ax.contourf(Tau, L, NLL, levels=levels_fill, cmap="viridis")

    # 2) lignes de niveau
    cs = ax.contour(Tau, L, NLL, levels=levels_lines,
                    colors="deepskyblue", linewidths=0.8)

    # (optionnel) étiquettes sur certaines lignes
    # ax.clabel(cs, inline=True, fontsize=7)

    # 3) point rouge (optionnel)
    if theta_star is not None:
        log_tau_star, log_l_star = theta_star
        ax.plot(np.exp(log_tau_star), np.exp(log_l_star),
                "o", color="red", markersize=6, zorder=5)

    # axes en log comme ton contour sur (tau, l)
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_title("Negative log-likelihood")
    ax.set_xlabel(r"$\tau$")      # ou "theta1 = log(sigma^2)" si tu veux pareil
    ax.set_ylabel(r"$\ell$")      # ou "theta2 = log(range)"

    cbar = fig.colorbar(cf, ax=ax)
    cbar.set_label("NLL")

    plt.tight_layout()
    plt.show()


# Exemple d'appel
# theta_star = (log_tau_opt, log_l_opt) si tu as un optimum
plot_nll_like_example(X_train, y_train, eta,
                      log_tau_range=(2, 8),
                      log_l_range=(-3, 8),
                      n_tau=30, n_l=30,
                      theta_star=(4.7, -1.5))  # à adapter / ou None

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

    # Trouver le minimum sur la grille
    idx_min = np.argmin(NLL)                       # index "aplati"
    i_min, j_min = np.unravel_index(idx_min, NLL.shape)

    log_l_star   = log_l_grid[i_min]
    log_tau_star = log_tau_grid[j_min]

    # Convertir en (tau, l) pour l'affichage (axes en log)
    tau_star = np.exp(log_tau_star)
    l_star   = np.exp(log_l_star)

    # meshgrid pour contour
    Tau, L = np.meshgrid(np.exp(log_tau_grid), np.exp(log_l_grid))

    fig, ax = plt.subplots(figsize=(7, 5))

    cf = ax.contourf(Tau, L, NLL, levels=levels_fill, cmap="viridis")
    cs = ax.contour(Tau, L, NLL, levels=levels_lines,
                    colors="deepskyblue", linewidths=0.8)

    # Point rouge = optimum (min NLL)
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

    print("Optimum sur la grille :")
    print(f"  log_tau* = {log_tau_star:.4f}, log_l* = {log_l_star:.4f}")
    print(f"  tau*     = {tau_star:.4f}, l*     = {l_star:.4f}")
    print(f"  NLL_min  = {NLL[i_min, j_min]:.4f}")



plot_nll_2d_optimal_point(X_train, y_train, eta,
                          log_tau_range=(2, 16),
                          log_l_range=(-5, 6),
                          n_tau=40,
                          n_l=40)
