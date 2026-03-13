import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fonctions_annexes import mat_soustractions


def plot_eigenvalue_histogram(matrix):
    """
    Plot a histogram of the eigenvalues of a given numpy matrix.
    
    Parameters:
    matrix (numpy.ndarray): Input square matrix
    
    Returns:
    None (displays the histogram plot)
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Input matrix must be square")
    
    eigenvalues = np.linalg.eigvals(matrix)
    

    plt.figure(figsize=(8, 6))
    plt.hist(eigenvalues.real, bins=30, edgecolor='black', alpha=0.7)
    

    plt.title("Histogram of Eigenvalues", fontsize=16)
    plt.xlabel("Eigenvalue", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.show()


def plot_eigenvalue_histogram_log(matrix, log_scale=True, log_bins=True):
    """
    Parameters:
    - matrix: Input square matrix
    - log_scale: Whether to use log scale on x-axis
    - log_bins: Whether to use logarithmically spaced bins
    """
    eigenvalues = np.linalg.eigvals(matrix)
    real_parts = eigenvalues.real
    
    plt.figure(figsize=(10, 6))
    
    if log_bins:
        # Filter out zeros/negative values if in log scale
        positive_vals = real_parts[real_parts > 0]
        if len(positive_vals) < len(real_parts):
            print(f"Warning: {len(real_parts)-len(positive_vals)} non-positive values omitted")
        
        bins = np.logspace(np.log10(min(positive_vals)), 
                          np.log10(max(positive_vals+200)), 
                          30)
        plt.hist(positive_vals, bins=bins, edgecolor='black', alpha=0.7)
    else:
        plt.hist(real_parts, bins=30, edgecolor='black', alpha=0.7)
    
    if log_scale:
        plt.xscale('log')
    
    plt.title("Histogramme des valeurs propres" + 
              (" (échelle logarithmique)" if log_scale else ""))
    plt.xlabel("Amplitude des valeurs propres")
    plt.ylabel("Compte")
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.show()
    
    
def fonction_covariance(x1,x2,l,tau):
        return tau**2*np.exp(-0.5*(mat_soustractions(x1,x2)**2)/l**2)
      
      
size = 5000

df_obs = pd.read_csv("data/borehole/borehole_obs.csv", header=None, skiprows=1).iloc[0:0+size]  


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

plot_eigenvalue_histogram_log(K11)