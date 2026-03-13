import numpy as np
import gc
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from fonctions_annexes import cholesky, ic_pivot_fast, mat_soustractions, ic_pivot_rbf

#constantes importantes
range_min = 600
range_max = 2001
pas = 200
eta = 0.001
l=2
tau=200

# Listes pour stocker les tailles de matrices et les temps d'exécution
sizes = [k for k in range(range_min, range_max, pas)]
ic_times = []
ic_rbf_times = []
ic_ranks = []
ic_rbf_ranks = []


def fonction_covariance(x1, x2, l, tau):
    return tau**2 * np.exp(-0.5 * (mat_soustractions(x1, x2)**2) / l**2)


def timed_run(func, *args):
    gc.collect()
    start = time.perf_counter_ns()
    a = func(*args)
    t = time.perf_counter_ns() - start
    return t, a


# Chargement de la base une seule fois
df_all = pd.read_csv("data/borehole/borehole_obs.csv", header=None, skiprows=1).astype(float)

# min et max des données
minimaux = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
maximaux = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]


# Calcul des temps d'exécution pour chaque taille de matrice
for size in sizes:

    ic_times_rep, ic_rbf_times_rep = [], []
    ic_ranks_rep, ic_rbf_ranks_rep = [], []

    # Trois répétitions avec des sous-échantillons différents
    for _ in range(3):
        df_obs = df_all.sample(n=size, random_state=None)  # échantillonnage aléatoire

        # normalisation
        X = df_obs.iloc[:, :-1].values
        X_train = (X - minimaux) / maximaux

        # Génération de la matrice
        cst_t , K11 = timed_run(fonction_covariance,X_train, X_train, l, tau)

        # Chronométrage IC
        t, k = timed_run(ic_pivot_fast, K11, eta)
        ic_times_rep.append(t+cst_t)
        ic_ranks_rep.append(k[1])

        # Chronométrage IC rbf
        t, k = timed_run(ic_pivot_rbf, X_train, l, tau , eta) #l=1 tau=1
        ic_rbf_times_rep.append(t)
        ic_rbf_ranks_rep.append(k[1])

    # On stocke les moyennes des 3 répétitions
    ic_times.append(np.mean(ic_times_rep))
    ic_ranks.append(np.mean(ic_ranks_rep))
    ic_rbf_times.append(np.mean(ic_rbf_times_rep))
    ic_rbf_ranks.append(np.mean(ic_rbf_ranks_rep))


# Conversion en millisecondes
ic_times2 = np.array(ic_times) / 1_000_000
ic_rbf_times2 = np.array(ic_rbf_times) / 1_000_000


# Création du graphique des temps
plt.figure(figsize=(10, 6))
plt.plot(sizes, ic_times2, label="Cholesky Incomplet", color='g', marker='s')
plt.plot(sizes, ic_rbf_times2, label="Cholesky Incomplet avec rbf", color='r', marker='^')
plt.yscale('log')
plt.title("Temps d'exécution en fonction de la taille de la matrice (moyenne sur 3 runs, échelle logarithmique)")
plt.xlabel("Taille n de la matrice")
plt.ylabel("log(Temps d'exécution, millisecondes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Création du graphique des rangs
plt.figure(figsize=(10, 6))
plt.plot(sizes, ic_ranks, label="IC", color='g', marker='s')
plt.plot(sizes, ic_rbf_ranks, label="IC rbf", color='r', marker='^')
plt.title("Rang faible en fonction de la dimension n (moyenne sur 3 runs)")
plt.xlabel("Taille n de la matrice")
plt.ylabel("Rang faible (nombre de colonnes conservées)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
