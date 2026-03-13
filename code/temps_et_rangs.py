import numpy as np
import gc
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from fonctions_annexes import ic_fast, cholesky, ic_pivot_fast, mat_soustractions



range_min = 100
range_max = 300
pas = 100

# Listes pour stocker les tailles de matrices et les temps d'exécution
sizes = [k for k in range(range_min,range_max,pas)]  # Tailles de matrices de 10 à 1000 avec un pas de 10
#cholesky_times = []
ic_times = []
ic_pivot_times = []
ic_ranks = []
ic_pivot_ranks = []

def fonction_covariance(x1,x2,l,tau):
        return tau**2*np.exp(-0.5*(mat_soustractions(x1,x2)**2)/l**2)


def timed_run(func, *args, repeats=3):
    times = []
    ranks=[]
    for _ in range(repeats):
        start = time.perf_counter_ns()
        a = func(*args)
        times.append(time.perf_counter_ns() - start)
        ranks.append(a[1])
        #print(a[0])
        gc.collect()
    return np.mean(times),np.mean(ranks)

# Calcul des temps d'exécution pour chaque taille de matrice
for size in sizes:
  
    df_obs = pd.read_csv("data/borehole/borehole_obs.csv", header=None, skiprows=1).iloc[10000:10000+size]  


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
    
    # Chronométrage de la fonction Cholesky
    '''
    start_time = time.perf_counter_ns()
    cholesky(K11)
    cholesky_times.append(time.perf_counter_ns() - start_time)
    gc.collect()
    '''
    
    # Chronométrage de la fonction Cholesky Incomplet
    t,k = timed_run(ic_fast,K11,0.001)
    ic_times.append(t)
    ic_ranks.append(k)
    
    # Chronométrage de la fonction Cholesky Incomplet avec Pivot
    t,k = timed_run(ic_pivot_fast,K11,0.001)
    ic_pivot_times.append(t)
    ic_pivot_ranks.append(k)

'''    
arr = np.array(cholesky_times)
cholesky_times2 = arr / 1_000_000
'''

arr = np.array(ic_times)
ic_times2 = arr / 1_000_000

arr = np.array(ic_pivot_times)
ic_pivot_times2 = arr / 1_000_000




# Création du graphique

plt.figure(figsize=(10, 6))
#plt.plot(sizes, cholesky_times, label="Cholesky", color='b', marker='o')
plt.plot(sizes, ic_times2, label="Cholesky Incomplet", color='g', marker='s')
plt.plot(sizes, ic_pivot_times2, label="Cholesky Incomplet avec Pivot", color='r', marker='^')
plt.title("Temps d'exécution en fonction de la taille de la matrice")
plt.xlabel("Taille n de la matrice")
plt.ylabel("Temps d'exécution (millisecondes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''
plt.figure(figsize=(10, 6))
plt.plot(sizes, cholesky_times2, label="Cholesky", color='b', marker='o')
plt.plot(sizes, ic_times2, label="Cholesky Incomplet", color='g', marker='s')
plt.plot(sizes, ic_pivot_times2, label="Cholesky Incomplet avec Pivot", color='r', marker='^')
plt.title("Temps d'exécution en fonction de la taille de la matrice")
plt.xlabel("Taille n de la matrice")
plt.ylabel("Temps d'exécution (millisecondes)")

plt.xscale('log')
plt.yscale('log') 

plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
'''


plt.figure(figsize=(10, 6))
plt.plot(sizes, ic_ranks, label="IC", color='g', marker='s')
plt.plot(sizes, ic_pivot_ranks, label="IC Pivot", color='r', marker='^')
plt.title("Rang faible en fonction de la dimension n")
plt.xlabel("Taille n de la matrice")
plt.ylabel("Rang faible (nombre de colonnes conservées)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Récupérer l'objet axes courant et forcer les ticks entiers
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()


