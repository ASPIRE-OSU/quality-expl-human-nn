import os
import pandas as pd
import numpy as np
import scipy.stats

DATAPATH = '../data/iucosine.csv'

df = pd.read_csv(DATAPATH)
tmos = df['scaled_mos'].to_numpy()
pmos = df['mosnet_pred'].to_numpy()

# stats 
mse = np.mean((tmos-pmos)**2)
lcc = np.corrcoef(tmos, pmos)
srcc = scipy.stats.spearmanr(tmos.T, pmos.T)

print(f"[UTTERANCE] MSE = {mse}")
print(f"[UTTERANCE] LCC = {lcc[0][1]}")
print(f"[UTTERANCE] SRCC = {srcc[0]}")