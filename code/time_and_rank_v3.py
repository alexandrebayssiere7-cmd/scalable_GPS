import numpy as np
import gc
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from helpers import cholesky, ic_pivot_fast, pairwise_distances, ic_pivot_rbf

# key constants
range_min = 600
range_max = 2001
step = 200
eta = 0.001
l=2
tau=200

# Lists storing the matrix sizes and the runtimes
sizes = [k for k in range(range_min, range_max, step)]
ic_times = []
ic_rbf_times = []
ic_ranks = []
ic_rbf_ranks = []


def covariance_function(x1, x2, l, tau):
    return tau**2 * np.exp(-0.5 * (pairwise_distances(x1, x2)**2) / l**2)


def timed_run(func, *args):
    gc.collect()
    start = time.perf_counter_ns()
    a = func(*args)
    t = time.perf_counter_ns() - start
    return t, a


# Load the dataset only once
df_all = pd.read_csv("data/borehole/borehole_obs.csv", header=None, skiprows=1).astype(float)

# min and max of the data
lower_bounds = [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855]
upper_bounds = [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045]


# Compute the runtimes for each matrix size
for size in sizes:

    ic_times_rep, ic_rbf_times_rep = [], []
    ic_ranks_rep, ic_rbf_ranks_rep = [], []

    # Three repetitions with different subsamples
    for _ in range(3):
        df_obs = df_all.sample(n=size, random_state=None)  # random sampling

        # normalization
        X = df_obs.iloc[:, :-1].values
        X_train = (X - lower_bounds) / upper_bounds

        # Build the matrix
        cst_t , K11 = timed_run(covariance_function,X_train, X_train, l, tau)

        # Time IC
        t, k = timed_run(ic_pivot_fast, K11, eta)
        ic_times_rep.append(t+cst_t)
        ic_ranks_rep.append(k[1])

        # Time RBF IC
        t, k = timed_run(ic_pivot_rbf, X_train, l, tau , eta) #l=1 tau=1
        ic_rbf_times_rep.append(t)
        ic_rbf_ranks_rep.append(k[1])

    # Store the means over the 3 repetitions
    ic_times.append(np.mean(ic_times_rep))
    ic_ranks.append(np.mean(ic_ranks_rep))
    ic_rbf_times.append(np.mean(ic_rbf_times_rep))
    ic_rbf_ranks.append(np.mean(ic_rbf_ranks_rep))


# Convert to milliseconds
ic_times2 = np.array(ic_times) / 1_000_000
ic_rbf_times2 = np.array(ic_rbf_times) / 1_000_000


# Runtime plot
plt.figure(figsize=(10, 6))
plt.plot(sizes, ic_times2, label="Incomplete Cholesky", color='g', marker='s')
plt.plot(sizes, ic_rbf_times2, label="Incomplete Cholesky with RBF", color='r', marker='^')
plt.yscale('log')
plt.title("Runtime as a function of the matrix size (mean over 3 runs, log scale)")
plt.xlabel("Matrix size n")
plt.ylabel("log(Runtime, milliseconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Rank plot
plt.figure(figsize=(10, 6))
plt.plot(sizes, ic_ranks, label="IC", color='g', marker='s')
plt.plot(sizes, ic_rbf_ranks, label="RBF IC", color='r', marker='^')
plt.title("Low rank as a function of the dimension n (mean over 3 runs)")
plt.xlabel("Matrix size n")
plt.ylabel("Low rank (number of columns kept)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
