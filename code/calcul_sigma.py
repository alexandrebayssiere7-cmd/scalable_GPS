import numpy as np
import pandas as pd 
from math import sqrt

df = pd.read_csv("data/bouquetins/PopSize1_2029.csv",header=0,index_col=0)

esp = [0 for i in range(216)]
esp2 = [0 for i in range(216)]
for i in range(216):
  for j in range(1000):
    esp[i] += df.iloc[j,i]
    esp2[i] += df.iloc[j,i]*df.iloc[j,i]
  esp[i] = esp[i]/1000 
  esp2[i] = esp2[i]/1000 
    
ecart = [0 for i in range(216)]
for i in range(216):
  ecart[i] = sqrt(esp2[i] - esp[i]*esp[i])
  
sigma = sum(ecart)/216

print(sigma)