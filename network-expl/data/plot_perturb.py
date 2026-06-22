import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
np.seterr(divide='ignore')


BASE_DIR = '../mosnet/results/shap'
PDP_DIR = '../mosnet/results/pdp'
PRED_DF = 'iucosine.csv'
PERTURB_DF = 'picked_perturbed.csv'
FS = 16000
FFT_SIZE = 320
SGRAM_DIM = FFT_SIZE // 2 + 1
NUM_FREQBINS = SGRAM_DIM
HOP_LENGTH = 160
WIN_LENGTH = 320
#INDICES = [0, 1, 2, 3, 4, 5, 31, 40, 41]
DEL = False
plt.style.use('tableau-colorblind10')
COLORS = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
MARKERS = ['o', 's', 'P', '^', 'd', 'X']
TRENDLINE = True
SCALES = [0.25, 0.5, 1, 10.0, 50.0, 100.0, 1000.0]
# BINS = [0, 1, 2, 3, 4, 5, 31, 40, 41]
DISPLAY_SCALES = [0.25, 0.5, 1, 10, 50, 100, 1000]

plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm'

def main():
    df = pd.read_csv(PRED_DF)
    perturb = pd.read_csv(PERTURB_DF)['filename'].tolist()
    size_unique = len(perturb)
    models = ['MOSNet', 'DNSMOS', 'ScoreQ']
    freqbins = [(3, 4, 5, 6, 12, 17, 31, 40, 41), 
                (3, 4, 5, 14, 15, 26, 31, 40, 41),
                (0, 2, 3, 4, 5, 20, 21, 31, 40, 41)]
    
    for m, bins, l, marker in zip(models, freqbins, ['-', '-', '-'], ['o', 'o', 'o']): #['-', '--', ':'], ['o', 's', '^']):
        print(f'{m}...')
        colors = COLORS[:9 ]
        
        perturb_files = ['{}_freqbin{}_scaled{}.wav'.format(f.split(".wav")[0], b, s) 
                                for s in SCALES 
                                    for b in bins 
                                        for f in perturb]
        perturb_df = df[df['filename'].isin(perturb_files)]
        size_total = len(perturb_df)
        print('Unique: {}\nTotal: {}'.format(size_unique, size_total))
        
        fig, ax = plt.subplots(figsize=(12, 7), dpi=600)     
        for freq, c in zip(bins, colors):
            scale_df = df[df['filename'].str.contains(f'freqbin{freq}_')]
            scale_files = scale_df['filename'].tolist()
            orig_files = list(set([f[:f.index('_freqbin')] + '.wav' for f in scale_files]))
            for f in orig_files:
                scale_df = pd.concat([scale_df, df[df['filename'] == f]])
    
            mos = {i: [] for i in SCALES}
    
            for i, row in scale_df.iterrows():
                f = row['filename']
                if 'scaled' in f:
                    scale = float(f[f.index('scaled')+6:f.index('.wav')])
                else:
                    scale = 1
            
                if scale not in SCALES:
                    continue
                
                if row[f'{m.lower()}_pred'] is not None:
                    mos[scale] = mos.get(scale, []) + [row[f'{m.lower()}_pred']]
        
            avgs = [np.mean(mos[s]) for s in SCALES]
            print(avgs)
                
            ax.plot(SCALES, 
                    avgs, 
                    linestyle=l, 
                    linewidth=2, 
                    marker=marker, 
                    markersize=4, 
                    label=f'Freq. Band {freq}', 
                    color=mcolors.to_rgb(c))

        ax.set_xscale('log')
        ax.set_xticks(DISPLAY_SCALES)
        ax.get_xaxis().set_major_formatter(plt.FormatStrFormatter('%g'))
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Average MOS Prediction')
        ax.set_title(f'Average Prediction by Scale Factor')
        ax.set_ylim(0.5, 6.5)
        ax.axhspan(1, 5, facecolor='gainsboro', alpha=0.5, label='Valid MOS Range')
        ax.grid(alpha=0.5)
        ax.legend(loc='upper left', fontsize=14) if m == 'MOSNet' else ax.legend(loc='upper right', fontsize=14) 
        plt.savefig(f'scale_pred_{m.lower()}.png', bbox_inches='tight')
        plt.clf()

        

if __name__ == '__main__':
    main()