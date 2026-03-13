import numpy as np
from math import sqrt
import time
import pandas as pd
from fonctions_annexes import mat_soustractions
import matplotlib.pyplot as plt

df_obs = pd.read_csv("data/borehole/borehole_obs.csv", header=None, skiprows=1).iloc[:1000]  
df_tst = pd.read_csv("data/borehole/borehole_tst.csv", header=None, skiprows=1).iloc[:100]

# Convertir toutes les valeurs en float
df_obs = df_obs.astype(float)



df_tst = df_tst.astype(float)

#max et min des données
minimaux= np.array([0.05,100,63070,990,63.1,700,1120,9855])
maximaux=np.array([0.15,50000,115600,1110,116,820,1680,12045])

#normalisation
X = df_obs.iloc[:, :-1].values
#X=[Y[len(Y)-1-i] for i in range(len(Y))]
X_t = df_tst.iloc[:, :-1].values
X_train=(X-minimaux)/(maximaux-minimaux)
X_test=(X_t-minimaux)/(maximaux-minimaux)

def ordonner_tilde(X):
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
  
def ordonner(X):
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
    Cholesky incomplète avec pivot (optimisée).
    
    Args:
        A (ndarray): Matrice symétrique définie positive (n x n).
        eta (float): Paramètre de précision dans [0, 1].
    
    Returns:
        ndarray: Matrice L de taille (n x k), telle que A ≈ L @ L.T
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
        # Sélection du pivot (non utilisé) avec la plus grande valeur diagonale
        if len(J) == 0:
            break
        ik = J[np.argmax(D[J])]
        
        if D[ik] <= 0:
            break  # Sécurité si la matrice perd sa SPD-ité

        L[ik, k] = sqrt(D[ik])

        # Mise à jour des lignes restantes
        J = J[J != ik]
        if k >= 0:
            # Calcul vectorisé : L[J, k] = (A[J, ik] - dot(L[J, :k], L[ik, :k])) / L[ik, k]
            L[J, k] = (A[J, ik] - L[J, :k] @ L[ik, :k]) / L[ik, k]
        else:
            L[J, k] = A[J, ik] / L[ik, k]

        # Mise à jour de D
        D[J] -= L[J, k]**2


        # Mettre à jour la somme S
        S += L[ik, k]**2 + np.sum(L[J, k]**2)
        k += 1

    return L[:, :k],k




def erreur1(K,L):
    somme=abs(np.trace(K-L@L.T))
    return somme
def erreur(K,L):
    somme=max([max((K-L@L.T)[i]) for i in range(len(K))])
    return somme




def rang_erreur(K):
    L1=np.linalg.cholesky(K)
    ic=ic_pivot_fast(K,0)
    L2=ic[0]
    lst_err=[]
    lst_err2=[]
    n=len(L1)
    for k in range(n+1):
        mat=L1[:,:k]
        a=erreur1(K,mat)
        lst_err.append(a)
        mat2=L2[:,:k]
        a2=erreur1(K,mat2)
        lst_err2.append(a2)
    return lst_err,lst_err2

#Valeurs de la fonction (observation et tests)

y_train = df_obs.iloc[:, -1].values
y_test = df_tst.iloc[:, -1].values

def fonction_covariance(x1,x2,l,tau):
        return tau**2*np.exp(-0.5*(mat_soustractions(x1,x2)**2)/l**2)

    #Matrice de covariance (loi jointe)
    
X_train=ordonner(X_train)

l=2.15
tau=1
K11=fonction_covariance(X_train,X_train,l,tau)
K12=fonction_covariance(X_train,X_test,l,tau)
K21=fonction_covariance(X_test,X_train,l,tau)
K22=fonction_covariance(X_test,X_test,l,tau)

x=[k for k in range(len(K11)+1)]

l1=rang_erreur(K11)[0]
l2=rang_erreur(K11)[1]

plt.plot(x,l1,color='r',label='Cholesky sans pivot')
plt.plot(x,l2,color='b',label='Cholesky avec pivot')
plt.yscale("log")
plt.xlabel("rang faible d'approximation")
plt.ylabel("erreur d'approximation")
plt.grid()
plt.title("Ordre : (n+1)-ième élément = voisin le plus proche du n-ième élément")
plt.legend()
plt.show()

X_train=(X-minimaux)/(maximaux-minimaux)
X_train=ordonner_tilde(X_train)

l=2.15
tau=1
K11=fonction_covariance(X_train,X_train,l,tau)
K12=fonction_covariance(X_train,X_test,l,tau)
K21=fonction_covariance(X_test,X_train,l,tau)
K22=fonction_covariance(X_test,X_test,l,tau)

x=[k for k in range(len(K11)+1)]

l1=rang_erreur(K11)[0]
l2=rang_erreur(K11)[1]

plt.plot(x,l1,color='r',label='Cholesky sans pivot')
plt.plot(x,l2,color='b',label='Cholesky avec pivot')
plt.yscale("log")
plt.xlabel("rang faible d'approximation")
plt.ylabel("erreur d'approximation")
plt.grid()
plt.title("Ordre des voisins les plus proches du 1er élément")
plt.legend()
plt.show()

