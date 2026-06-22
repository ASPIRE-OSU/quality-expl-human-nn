import glob
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import librosa
import scipy.signal
import matplotlib.pyplot as plt

TO_PLOT = {'Prediction vs. Scale': False,
           'Average SHAP vs. Frequency Band': True}

MOSNET_SHAP = '../mosnet/results/shap'
DNSMOS_SHAP = '../dnsmos/results/shap'
SCOREQ_SHAP = '../scoreq/results/shap'
DATA_DIR = '../../data/MOS_Quality_INTERSPEECH_2020/16k_speech'
IUCOSINE_INFO = 'iucosine.csv'
FS = 16000
FFT_SIZE = 320
SGRAM_DIM = FFT_SIZE // 2 + 1
NUM_FREQBINS = SGRAM_DIM
HOP_LENGTH = 160
WIN_LENGTH = 320

def main():
    iucosine_df = pd.read_csv(IUCOSINE_INFO)
    
    if TO_PLOT['Prediction vs. Scale']:
        plot_pred_vs_scale(iucosine_df)
    elif TO_PLOT['Average SHAP vs. Frequency Band']:
        plot_avg_shap_freq_band(iucosine_df)

def plot_avg_shap_freq_band(iucosine_df, collect_data=False):
    print('Plotting Average SHAP vs. Frequency Band...')
    if collect_data:
        df = collect_avg_mag_shap(iucosine_df)
    else:
        df = pd.read_csv('avg_mag_shap.csv')
    
    #for name, rpath in tqdm({'mosnet': MOSNET_SHAP, 'dnsmos': DNSMOS_SHAP, 'scoreq': SCOREQ_SHAP}.items()):
    name = 'mosnet'
    display_name = 'MOSNet'
    rpath = MOSNET_SHAP
    
    pred_column = f"{name}_pred"
    df = df[df[pred_column].notna()]
    # plot avg shap vs. frequency band
    fig, ax = plt.subplots(dpi=600)
    avg_shap = df.groupby('freqbin')['shap'].mean().reset_index()
    print(avg_shap)
    x_data = avg_shap['freqbin'].tolist()
    y_data = avg_shap['shap'].tolist()
    ax.bar(x_data, y_data)
    ax.set_title(f'Average SHAP Value by Frequency Band for {display_name}')
    ax.set_ylabel('Average SHAP Value')
    ax.set_xlabel('Frequency Band (width = 50 Hz)')
    plt.savefig(f'avgshap_freqband_{name}.png', bbox_inches='tight')
    
        
def collect_avg_mag_shap(iucosine_df, outfile='avg_mag_shap.csv'):
    df = pd.DataFrame(columns=['file', 'freqbin', 'avg_mag_db', 'shap', 'tmos', 'dnsmos_pred', 'mosnet_pred', 'scoreq_pred'])
    
    #for name, rpath in tqdm({'mosnet': MOSNET_SHAP, 'dnsmos': DNSMOS_SHAP, 'scoreq': SCOREQ_SHAP}.items()):
    name = 'mosnet'
    rpath = MOSNET_SHAP
    
    files = [f for f in glob.glob(rpath + '/shap*.csv')]
    
    for shap_file in tqdm(files):
        shap_df = pd.read_csv(shap_file)
        columns = ['file'] + [f'freqbin_{i}' for i in range(NUM_FREQBINS)]
        if 'filename' in shap_df.columns.tolist():
            shap_df = shap_df.rename(columns={'filename': 'file'})
        #print(shap_df.columns.tolist())
        shap_df = shap_df[columns]
        filename = shap_df.iloc[0]['file']
        spec = get_spectrograms(os.path.join(DATA_DIR, filename))
        
        for freqbin_str in shap_df.columns.tolist()[1:]:
            new_row = {k: None for k in df.columns.tolist()}
            iucosine_rows = iucosine_df[iucosine_df['filename'] == filename]
            
            freqbin = int(freqbin_str.split('_')[-1])
            new_row['file'] = filename
            new_row['freqbin'] = freqbin
            new_row['shap'] = shap_df.iloc[0][freqbin_str]
            
            avg_mag = np.mean(spec[freqbin])
            new_row['avg_mag_db'] = to_dB(avg_mag)
            
            pred_column = f"{name}_pred"
            new_row[pred_column] = iucosine_rows[pred_column].values[0]
            
            new_row['tmos'] = iucosine_rows['scaled_mos'].values[0]
            
            df.loc[len(df)] = new_row
    
    df.to_csv(outfile, index=False)
    return df


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
    #return np.transpose(mag.astype(np.float32))
    
    return mag

def to_dB(ampl):
    return 20*np.log10(ampl)

if __name__ == '__main__':
    main()