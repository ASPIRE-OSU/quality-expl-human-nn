import numpy as np
import sys
import os
import argparse
import pandas as pd
import glob
from tqdm import tqdm
import scipy.stats
import librosa

FS = 16000
FFT_SIZE = 320
SGRAM_DIM = FFT_SIZE // 2 + 1
HOP_LENGTH = 160
WIN_LENGTH = 320

DATA_DIR = '../../MOS_Quality_INTERSPEECH_2020/16k_speech'
PERTURB_DIR = 'data/perturbed_iucosine'
AVG_AMP_FILE = 'avg_ampl_iucosine.csv'
PREDS = 'data/iucosine.csv'

SCALES = [0.25, 0.5, 1, 10, 50, 100, 1000]
BANDS = [0, 1, 2, 3, 4, 5, 31, 40, 41]

def main():
    args = _parse_args()
    preds = pd.read_csv(PREDS)
    
    files = []
    pfiles = [f.split('/')[-1] for f in glob.glob(PERTURB_DIR + "/*.wav")]
    for pf in pfiles:
        scale = pf[pf.index('scaled')+6:pf.index('.wav')]
        freqbin = pf[pf.index('freqbin')+7:pf.index('_scaled')]
        files.append((pf, int(freqbin)))

    # now add scale 1 files
    for filename, fbin in files:
        if 'scaled' in filename:
            orig_filename = filename[0:filename.index('_freqbin')] + '.wav'
            files.append((orig_filename, fbin))
    files = list(set(files))
        
    if os.path.exists(AVG_AMP_FILE):
        df = pd.read_csv(AVG_AMP_FILE)
    else:
        df = pd.DataFrame(columns=['orig_file', 'filename', 'scale', 'freqbin', 'avg_mag_raw', 'avg_mag_db', 'mosnet_predicted_mos', 'dnsmos_predicted_mos', 'scoreq_predicted_mos', 'true_mos_before_scaling', 'mosnet_orig_predicted_mos', 'dnsmos_orig_predicted_mos','scoreq_predicted_mos', 'mosnet_delta_mos', 'dnsmos_delta_mos', 
                                'scoreq_delta_mos'])

        print('Calculating...')
        for f in tqdm(files):
            if 'scaled' in f[0]:
                filepath = os.path.join(PERTURB_DIR, f[0])
            else:
                filepath = os.path.join(DATA_DIR, f[0])
            get_amp(filepath, f[1], df, preds)
        df.to_csv(AVG_AMP_FILE, index=False)
        
    create_table(df)
    print('Done.')
    

def create_table(df, band=1, outfile='perturb-table.tex'):
    print('Creating table...')
    band_range = (band*50, (band+1)*50)
    vrule_str = '!{\\vrule width 0.05em}'
    
    ####### FREQ. BANDS 0-4 ########
    # table environment
    print_str = '\\begin{table}[htbp!]\n\t\centering\n'
    print_str += '\t\\captionsetup{skip=15pt}\n'
    print_str += '\t\\caption{Effect over 100 signals on model prediction from perturbing frequency bands 0-4. Note that scale 1 is the original signal. Average predictions are for the entire signal when only the given frequency band has been altered.}\n'
    print_str += '\t\\label{tab:perturb0-4}\n' 
                       
    # table headers
    print_str += '\t\\begin{tabular}{c r ' + vrule_str + ' r ' + vrule_str + ' r r r ' + '}\n'
    print_str += '\t\t\\toprule\n'
    print_str += '\t\t\\multirow{2}{3cm}{\\centering\\textbf{Frequency Band}} & \\multirow{2}{*}{\\textbf{Scale}} & \\multirow{2}{3cm}{\\centering\\textbf{Average Magnitude}} & \\multicolumn{3}{c}{\\textbf{Change in Prediction}}\\\\\n'
    print_str += '\t\t & & & \\textbf{MOSNet} & \\textbf{DNSMOS} & \\textbf{ScoreQ} \\\\\n'
    print_str += '\t\t\\midrule\n'
    
    # calculate averages for each scale
    for band in BANDS[:5]:
        band_start = band*50
        band_end = (band+1)*50
        print_str += '\t\t\\multirow{7}{*}{\\makecell{$' + str(band) + '$\\\\(' + str(band_start) + '--' + str(band_end) + ' Hz)}} & '
        band_df = df[df['freqbin'] == band]
        for s in tqdm(SCALES):
            if s != 0.25: print_str += '\t\t& '
            band_scale_df = band_df[band_df['scale'].astype(float) == s]
            avg_mag_db = np.mean(band_scale_df['avg_mag_db'].tolist())
            avg_mosnet = np.mean(band_scale_df['mosnet_delta_mos'].tolist())
            avg_dnsmos = np.mean(band_scale_df['dnsmos_delta_mos'].tolist())
            avg_scoreq = np.mean(band_scale_df['scoreq_delta_mos'].tolist())
            print_str += f'${s}$ & ${avg_mag_db:.4f}$ dB & ${avg_mosnet:.4f}$ & ${avg_dnsmos:.4f}$ & ${avg_scoreq:.4f}$\\\\\n'
        
        if band != BANDS[4]: print_str += '\t\t\\midrule\n'
    
    # table footer
    print_str += '\t\t\\bottomrule\n\t\\end{tabular}\n\\end{table}\n\n'
    
    ####### FREQ. BANDS 5+ ########
    # table environment
    print_str += '\\begin{table}[htbp!]\n\t\centering\n'
    print_str += '\t\\caption{Effect over 100 signals on model prediction from perturbing frequency bands 5, 31, 40-41. Note that scale 1 is the original signal. Average predictions are for the entire signal when only the given frequency band has been altered.}\n'
    print_str += '\t\\label{tab:perturb5+}\n' 
                       
    # table headers
    print_str += '\t\\begin{tabular}{c r ' + vrule_str + ' r ' + vrule_str + ' r r r ' + '}\n'
    print_str += '\t\t\\toprule\n'
    print_str += '\t\t\\multirow{2}{3cm}{\\centering\\textbf{Frequency Band}} & \\multirow{2}{*}{\\textbf{Scale}} & \\multirow{2}{3cm}{\\centering\\textbf{Average Magnitude}} & \\multicolumn{3}{c}{\\textbf{Change in Prediction}}\\\\\n'
    print_str += '\t\t & & & \\textbf{MOSNet} & \\textbf{DNSMOS} & \\textbf{ScoreQ} \\\\\n'
    print_str += '\t\t\\midrule\n'
    
    # calculate averages for each scale
    for band in BANDS[5:]:
        band_start = band*50
        band_end = (band+1)*50
        print_str += '\t\t\\multirow{7}{*}{\\makecell{$' + str(band) + '$\\\\(' + str(band_start) + '--' + str(band_end) + ' Hz)}} & '
        band_df = df[df['freqbin'] == band]
        for s in tqdm(SCALES):
            if s != 0.25: print_str += '\t\t& '
            band_scale_df = band_df[band_df['scale'].astype(float) == s]
            avg_mag_db = np.mean(band_scale_df['avg_mag_db'].tolist())
            avg_mosnet = np.mean(band_scale_df['mosnet_delta_mos'].tolist())
            avg_dnsmos = np.mean(band_scale_df['dnsmos_delta_mos'].tolist())
            avg_scoreq = np.mean(band_scale_df['scoreq_delta_mos'].tolist())
            print_str += f'${s}$ & ${avg_mag_db:.4f}$ dB & ${avg_mosnet:.4f}$ & ${avg_dnsmos:.4f}$ & ${avg_scoreq:.4f}$\\\\\n'
        
        if band != BANDS[-1]: print_str += '\t\t\\midrule\n'
            
    
    # table footer
    print_str += '\t\t\\bottomrule\n\t\\end{tabular}\n\\end{table}'
    
    with open(outfile, 'w') as f:
        f.write(print_str)
    

def _parse_args():
    """
    Reads and stores command line arguments.
    
    Returns:
       args parser object
    """
    parser = argparse.ArgumentParser(prog='Get average ampl by bin')
    parser.add_argument('-f', '--file', required=False, type=str, nargs="*", help='The file path of the wav file to predict MOS for.')
    parser.add_argument('-b', '--freqbin', required=False, type=int)
    return parser.parse_args()


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


def get_amp(filepath, freqbin, df, preds):
    filename = filepath.split('/')[-1]
    mag = get_spectrograms(filepath)
    avg_mag = np.mean(mag, axis=0)[freqbin]
    avg_mag_dB = 20 * np.log10(avg_mag)
        
    #row_idx = df.index[df['filename'] == filepath]
    # row doesn't exist, so let's add a new one
    # columns=set,file,true_mos,predicted_mos
    pred_mos = preds[preds['filename'] == filename]['mosnet_pred'].values[0]
    pred_dns = preds[preds['filename'] == filename]['dnsmos_pred'].values[0]
    pred_scoreq = preds[preds['filename'] == filename]['scoreq_pred'].values[0]

    if 'scaled' in filename:
        orig_filename = filename[0:filename.index('_freqbin')] + '.wav'
        scale = filename[filename.index('scaled')+6:filename.index('.wav')]
    else:
        orig_filename = filename
        scale = 1
        
    mosnet_orig_mos = preds[preds['filename'] == orig_filename]['mosnet_pred'].values[0]
    dnsmos_orig_mos = preds[preds['filename'] == orig_filename]['dnsmos_pred'].values[0]
    scoreq_orig_mos = preds[preds['filename'] == orig_filename]['scoreq_pred'].values[0]
    
    true_mos = preds[preds['filename'] == orig_filename]['mos5'].values[0]
    mosnet_delta = pred_mos - mosnet_orig_mos
    dnsmos_delta = pred_dns - dnsmos_orig_mos
    scoreq_delta = pred_scoreq - scoreq_orig_mos
    
    new_row = {'orig_file': orig_filename,
                'filename': filename,
                'scale': scale,
                'freqbin': freqbin,
                'avg_mag_raw': avg_mag,
                'avg_mag_db': avg_mag_dB,
                'mosnet_predicted_mos': pred_mos, 
                'dnsmos_predicted_mos': pred_dns, 
                'scoreq_predicted_mos': pred_scoreq,
                'true_mos_before_scaling': true_mos,
                'mosnet_orig_predicted_mos': mosnet_orig_mos,
                'dnsmos_orig_predicted_mos': dnsmos_orig_mos,
                'scoreq_orig_predicted_mos': scoreq_orig_mos,
                'mosnet_delta_mos': mosnet_delta,
                'dnsmos_delta_mos': dnsmos_delta,
                'scoreq_delta_mos': scoreq_delta}
    df.loc[len(df)] = new_row
    # row exists so let's update the average magnitude
    # else:
    #     cols = df.columns.tolist()
    #     row_idx = row_idx[0]
    #     df.iat[row_idx, cols.index('avg_mag_raw')] = avg_mag
    #     df.iat[row_idx, cols.index('avg_mag_db')] = avg_mag_dB



if __name__ == '__main__':
    main()