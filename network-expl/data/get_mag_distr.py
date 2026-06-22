import os
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import pickle
import librosa
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
np.seterr(divide='ignore')
import matplotlib.colors as mcolors
import scipy.stats
from matplotlib.patches import Patch

DATA = 'iucosine.csv'
DATA_DIR = '../../data/MOS_Quality_INTERSPEECH_2020/16k_speech'
FS = 16000
FFT_SIZE = 320
SGRAM_DIM = FFT_SIZE // 2 + 1
NUM_FREQBINS = SGRAM_DIM
HOP_LENGTH = 160
WIN_LENGTH = 320
PLOT = [0, 1, 2, 3, 4, 5, 30, 40, 41]
DEL = False
CALC = False

if CALC:
    df = pd.read_csv(DATA)
    avg_mag_db_distr = defaultdict(list)

    def to_dB(ampl):
        return 20*np.log10(ampl)

    def from_dB(dB):
        return [10 ** (d / 20) for d in dB]

    def get_width():
        max_f = FS/2
        bandwidth = max_f/(NUM_FREQBINS-1)
        return bandwidth

    def get_range(i, width):
        return (i*width, i*width + width)

    def get_spectrograms(sound_file, fs=FS, fft_size=FFT_SIZE):         
        # Loading sound file
        y, _ = librosa.load(sound_file, sr=fs) # or set sr to hp.sr.

        # Preemphasis
        #y = np.append(y[0], y[1:] - PREEMPHASIS * y[:-1])

        # stft. D: (1+n_fft//2, T)
        linear = librosa.stft(y=y,
                        n_fft=fft_size, 
                        hop_length=HOP_LENGTH, 
                        win_length=WIN_LENGTH,
                        window=scipy.signal.windows.hamming,
                        )

        # magnitude spectrogram
        mag = np.abs(linear) #(1+n_fft/2, T)
        
        # shape in (T, 1+n_fft/2)
        return np.transpose(mag.astype(np.float32))

    def get_avg_mag(file):
        # rows = files, columns = dB magnitude by frequency bin
        avg_mags = {i: None for i in range(NUM_FREQBINS)}
        spec = get_spectrograms(os.path.join(DATA_DIR, file))
        avg_mag = np.mean(spec, axis=0)
        avg_mag_dB = to_dB(avg_mag)
        
        for i in range(NUM_FREQBINS):
            avg_mags[i] = avg_mag_dB[i]
            
        return avg_mags    

    tmos = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        avg_mags = get_avg_mag(row['filename'])
        tmos.append(row['scaled_mos'])
        for i in range(len(avg_mags)):
            avg_mag_db_distr[i].append(avg_mags[i])
        
    pickle.dump((avg_mag_db_distr, tmos), open('mag_stats.pkl', 'wb'))

else:
    avg_mag_db_distr, tmos = pickle.load(open('mag_stats.pkl', 'rb'))

def plot(data, fbin, tmos):
    # set up figure
    fig, ax = plt.subplots(dpi=600, layout='constrained')
    ax.set_xlabel('Magnitude (dB)')
    ax.set_ylabel('Count')
    start = int(fbin)*50
    end = int((fbin)+1)*50
    ax.set_title(f"Average Magnitude in dB\nof Frequency Bin {fbin} ({start}-{end} Hz)")
    plt.rcParams.update({'font.size': 16})
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif"]
    plt.rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm'
    
    _, _, patches = ax.hist(data, bins=100, rwidth=0.9)
    mos_val = {1: 'tab:red',
               2: 'tab:orange',
               3: 'tab:green', 
               4: 'tab:blue',
               5: 'tab:purple'}
        
    for file_idx, patch in zip(range(len(tmos)), patches):
        trunc_mos = int(tmos[file_idx])
        patch.set_facecolor(mos_val[trunc_mos])
        
    # create custom legend
    legend_handles = [Patch(facecolor=color, label=f'{mos}')
                    for mos, color in mos_val.items()]
    ax.legend(handles=legend_handles, title="True MOS", fontsize='x-small', loc='upper left')

    plt.savefig(f"avg_db_freqbin{fbin}.png")

if len(PLOT) > 0:
    print('Plotting...')
    for p in tqdm(PLOT):
        distr = avg_mag_db_distr[p]
        plot(distr, p, tmos)
    
    # convert to PDF
    pngs = [f"avg_db_freqbin{b}.png" for b in PLOT]
    print('Converting to PDF...')
    pngs.sort()
    imgs = [Image.open(i).convert('RGB') for i in pngs]   
    imgs[0].save("abg_mag_db_distr.pdf", save_all=True, append_images=imgs[1:], quality=100)
            
    if DEL:
        print('Deleting .png files...')
        for png in tqdm(pngs):
            if os.path.exists(png):
                os.remove(png)
        
        