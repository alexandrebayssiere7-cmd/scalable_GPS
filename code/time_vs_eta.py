import numpy as np
import matplotlib.pyplot as plt
import time
from helpers import ic_pivot_fast,ic_fast,pairwise_distances
import gc
import pandas as pd

def covariance_function(x1,x2,l,tau):
        return tau**2*np.exp(-0.5*(pairwise_distances(x1,x2)**2)/l**2)

n=500
#A=np.random.randn(n,n)
#M=A@A.T

df_obs = pd.read_csv("data/borehole/borehole_obs.csv", header=None, skiprows=1).iloc[10000:10000+n]


# Convert all values to float
df_obs = df_obs.astype(float)


# min and max of the data
lower_bounds= [0.05,100,63070,990,63.1,700,1120,9855]
upper_bounds=[0.15,50000,115600,1110,116,820,1680,12045]

# normalization
X = df_obs.iloc[:, :-1].values

X_train=(X-lower_bounds)/upper_bounds

# Build the matrix
K11=covariance_function(X_train,X_train,1,1)

gc.collect()

def plot_time_and_rank_vs_eta(A):
    # Values of 1 - eta from 1e-10 to about 1 (log-spaced)
    etas = np.logspace(-6, -0.02, 50)  # 10^-0.02 ≈ 0.95
    times = []
    ranks = []

    for eta in etas:
        start = time.perf_counter_ns()
        _, r = ic_pivot_fast(A, eta)
        times.append(time.perf_counter_ns() - start)
        times[-1] = times[-1]/1_000_000
        ranks.append(r)
        gc.collect()


    fig, ax1 = plt.subplots()

    color = 'tab:blue'
    ax1.set_xlabel(r'$\eta$')
    ax1.set_xscale('log')
    ax1.set_ylabel('Time (ms)', color=color)
    ax1.plot(etas, times, label='Time', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Rank of $L$', color=color)
    ax2.plot(etas, ranks, label='Rank', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Time and rank as a function of $\eta$, with pivoting, for n=500")
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

plot_time_and_rank_vs_eta(K11)