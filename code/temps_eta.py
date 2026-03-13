import numpy as np
import matplotlib.pyplot as plt
import time
from fonctions_annexes import ic_pivot_fast,ic_fast,mat_soustractions
import gc
import pandas as pd

def fonction_covariance(x1,x2,l,tau):
        return tau**2*np.exp(-0.5*(mat_soustractions(x1,x2)**2)/l**2)

n=500
#A=np.random.randn(n,n)
#M=A@A.T

df_obs = pd.read_csv("data/borehole/borehole_obs.csv", header=None, skiprows=1).iloc[10000:10000+n]  


# Convertir toutes les valeurs en float
df_obs = df_obs.astype(float)


#max et min des données
minimaux= [0.05,100,63070,990,63.1,700,1120,9855]
maximaux=[0.15,50000,115600,1110,116,820,1680,12045]

#normalisation
X = df_obs.iloc[:, :-1].values

X_train=(X-minimaux)/maximaux

# Génération de la matrice
K11=fonction_covariance(X_train,X_train,1,1)

gc.collect()

def courbe2(A):
    # On crée des valeurs de 1 - eta allant de 1e-10 à environ 1 (log-équidistantes)
    etas = np.logspace(-6, -0.02, 50)  # 10^-0.02 ≈ 0.95
    times = []
    rangs = []

    for eta in etas:
        start = time.perf_counter_ns()
        _, r = ic_pivot_fast(A, eta)
        times.append(time.perf_counter_ns() - start)
        times[-1] = times[-1]/1_000_000
        rangs.append(r)
        gc.collect()


    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel(r'$\eta$')
    ax1.set_xscale('log')
    ax1.set_ylabel('Temps (ms)', color=color)
    ax1.plot(etas, times, label='Temps', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Rang de $L$', color=color)
    ax2.plot(etas, rangs, label='Rang', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Évolution du temps et du rang selon $\eta$ , avec pivot pour n=500")
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

courbe2(K11)