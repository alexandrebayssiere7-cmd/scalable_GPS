import pandas as pd
import numpy as np

scenario_df = pd.read_csv(
    "data/bouquetins/scenario.csv",
    header=0,
    index_col=0
)

values_df = pd.read_csv(
    "data/bouquetins/PopSize1_2029.csv",
    header=0,
    index_col=0
)

tab = np.zeros((216000, 4))
df_obs = pd.DataFrame(tab)

for i in range(216):
    for j in range(1000):
        idx = j + i * 1000
        df_obs.iloc[idx, 0] = scenario_df.iloc[i, 0]
        df_obs.iloc[idx, 1] = scenario_df.iloc[i, 1]
        df_obs.iloc[idx, 2] = scenario_df.iloc[i, 2]
        df_obs.iloc[idx, 3] = values_df.iloc[j, i]


#MELANGER LES LIGNES
df_obs = df_obs.sample(frac=1)


df_obs.to_csv("data/bouquetins/bouquetins.csv",index=False)
