import os
import pandas as pd
import librosa
import numpy as np
from tqdm import tqdm

BASE_DIR = '../mosnet/results/shap'
PDP_DIR = '../mosnet/results/pdp'
DATA_DIR = '../../MOS_Quality_INTERSPEECH_2020/16k_speech'
NUM_FREQBINS = 257
FS = 16000
FFT_SIZE = 512
SGRAM_DIM = FFT_SIZE // 2 + 1
HOP_LENGTH=256
WIN_LENGTH=512
OUTFILE = 'f0.csv'

files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.wav')]

if os.path.exists(OUTFILE):
    df = pd.read_csv(OUTFILE)
    
    # remove files that we've already estimated
    complete_files = df['file'].tolist()
    complete_files = [os.path.join(DATA_DIR, f) for f in complete_files]
    files = list(set(files) - set(complete_files))
else:
    df = pd.DataFrame(columns=['file', 'favg', 'fmin', 'fmax', 'f0', 'voiced_flag', 'voiced_probs'])

for fidx in tqdm(range(len(files))):
    y, _ = librosa.load(files[fidx], sr=FS)

    # f0: time series of fundamental frequencies in Hz
    # voiced_flag: time series containing boolean flags indicated whether a frame is voiced or not
    # voiced_probs: time series containing the probability that a frame is voiced
    f0, voiced_flag, voiced_probs = librosa.pyin(y, 
                                                sr=FS,
                                                hop_length=HOP_LENGTH,
                                                win_length=WIN_LENGTH,
                                                fmin=1, # C2~65 Hz, recommended
                                                fmax=FS/2, # C7~2093 Hz, recommended
                                                )
    
    fmin = np.nanmin(f0)
    fmax = np.nanmax(f0)
    favg = np.nanmean(f0)
    
    new_row = {'file': files[fidx].split('/')[-1], 'favg': favg, 'fmin': fmin, 'fmax': fmax, 'f0': f0, 'voiced_flag': voiced_flag, 'voiced_probs':voiced_probs}
    df = pd.concat([df if not df.empty else None, pd.DataFrame([new_row])], ignore_index=True)    
    
    if fidx % 100 == 0:
        df.to_csv(OUTFILE, index=False)     

df.to_csv(OUTFILE, index=False)
