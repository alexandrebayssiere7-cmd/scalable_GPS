import numpy as np
import gc
import pandas as pd
import time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from helpers import ic_fast, cholesky, ic_pivot_fast, pairwise_distances



range_min = 100
range_max = 300
step = 100

# Lists storing the matrix sizes and the runtimes
sizes = [k for k in range(range_min,range_max,step)]  # Matrix sizes from 10 to 1000 with a step of 10
#cholesky_times = []
ic_times = []
ic_pivot_times = []
ic_ranks = []
ic_pivot_ranks = []

def covariance_function(x1,x2,l,tau):
        return tau**2*np.exp(-0.5*(pairwise_distances(x1,x2)**2)/l**2)


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

# Compute the runtimes for each matrix size
for size in sizes:

    df_obs = pd.read_csv("data/borehole/borehole_obs.csv", header=None, skiprows=1).iloc[10000:10000+size]


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

    # Time the Cholesky function
    '''
    start_time = time.perf_counter_ns()
    cholesky(K11)
    cholesky_times.append(time.perf_counter_ns() - start_time)
    gc.collect()
    '''

    # Time the Incomplete Cholesky function
    t,k = timed_run(ic_fast,K11,0.001)
    ic_times.append(t)
    ic_ranks.append(k)

    # Time the Pivoted Incomplete Cholesky function
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




# Build the plot

plt.figure(figsize=(10, 6))
#plt.plot(sizes, cholesky_times, label="Cholesky", color='b', marker='o')
plt.plot(sizes, ic_times2, label="Incomplete Cholesky", color='g', marker='s')
plt.plot(sizes, ic_pivot_times2, label="Pivoted Incomplete Cholesky", color='r', marker='^')
plt.title("Runtime as a function of the matrix size")
plt.xlabel("Matrix size n")
plt.ylabel("Runtime (milliseconds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

'''
plt.figure(figsize=(10, 6))
plt.plot(sizes, cholesky_times2, label="Cholesky", color='b', marker='o')
plt.plot(sizes, ic_times2, label="Incomplete Cholesky", color='g', marker='s')
plt.plot(sizes, ic_pivot_times2, label="Pivoted Incomplete Cholesky", color='r', marker='^')
plt.title("Runtime as a function of the matrix size")
plt.xlabel("Matrix size n")
plt.ylabel("Runtime (milliseconds)")

plt.xscale('log')
plt.yscale('log')

plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
plt.show()
'''


plt.figure(figsize=(10, 6))
plt.plot(sizes, ic_ranks, label="IC", color='g', marker='s')
plt.plot(sizes, ic_pivot_ranks, label="Pivoted IC", color='r', marker='^')
plt.title("Low rank as a function of the dimension n")
plt.xlabel("Matrix size n")
plt.ylabel("Low rank (number of columns kept)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Get the current axes and force integer ticks
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()

