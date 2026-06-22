import pandas as pd
from tqdm import tqdm
import ast
import re
import numpy as np

 
DATA = 'f0.csv'
PROB_THRESHOlD = 0.5


def clean_convert(str_array):
    str_array = str_array.strip()
    str_array = str_array.replace('nan', 'None')
    str_array = re.sub(r'\s+', ',', str_array)
    if str_array[0:2] == '[,':
        str_array = '[' + str_array[2:]
    return np.array(ast.literal_eval(str_array), dtype=float)


df = pd.read_csv(DATA)

# first get old stats
print('Before thresholding:')
print('Minimum f0: {:.3f} Hz'.format(np.nanmin(df['fmin'].tolist())))
print('Average f0: {:.3f} Hz'.format(np.nanmean(df['favg'].tolist())))
print('Maximum f0: {:.3f} Hz\n'.format(np.nanmax(df['fmax'].tolist())))

for i, row in tqdm(df.iterrows(), total=len(df)):
    # clean and read f0 values and voiced probabilities from csv
    f0 = clean_convert(row['f0'])
    probs = clean_convert(row['voiced_probs'])    
    
    # get indices of frames whose voiced probability is below threshold
    indices = np.argwhere(probs < PROB_THRESHOlD).flatten()
    
    # clear out those frames' f0 value (set to nan)
    f0[indices] = np.nan
    
    # recalculate f0 stats
    fmin = np.nanmin(f0)
    fmax = np.nanmax(f0)
    favg = np.nanmean(f0)
    
    # update df
    df.at[i, 'fmin'] = fmin
    df.at[i, 'fmax'] = fmax
    df.at[i, 'favg'] = favg
    df.at[i, 'f0'] = f0
    
    # checkpoint save
    if i % 100 == 0:
        df.to_csv(DATA.split('.csv')[0] + '_threshold.csv', index=False)
    
# final save 
df.to_csv(DATA.split('.csv')[0] + '_threshold.csv', index=False)

# display new stats
print('After thresholding:')
print('Minimum f0: {:.3f} Hz'.format(np.nanmin(df['fmin'].tolist())))
print('Average f0: {:.3f} Hz'.format(np.nanmean(df['favg'].tolist())))
print('Maximum f0: {:.3f} Hz'.format(np.nanmax(df['fmax'].tolist())))