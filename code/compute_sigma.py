import numpy as np
import pandas as pd
from math import sqrt

df = pd.read_csv("data/ibex/PopSize1_2029.csv",header=0,index_col=0)

mean = [0 for i in range(216)]
second_moment = [0 for i in range(216)]
for i in range(216):
  for j in range(1000):
    mean[i] += df.iloc[j,i]
    second_moment[i] += df.iloc[j,i]*df.iloc[j,i]
  mean[i] = mean[i]/1000
  second_moment[i] = second_moment[i]/1000

std = [0 for i in range(216)]
for i in range(216):
  std[i] = sqrt(second_moment[i] - mean[i]*mean[i])

sigma = sum(std)/216

print(sigma)