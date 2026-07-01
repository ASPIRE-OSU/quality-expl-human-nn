'''
This script contains all utility methods used to process the data files. 

@author: Ada Lamba
@version: 07/01/2026
'''
import ast
from collections import defaultdict
import glob
import librosa
import math
import matplotlib.colors as mcolors
from matploblib.patches import Patch
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
from PIL import Image
import random
import re
import scipy.stats
import soundfile as sf
import sys
from tqdm import tqdm

########## LIBRARY SETUP ##########
np.setterr(divide='ignore')
plt.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm'

########## CONSTANTS ##########
FS = 16000
FFT_SIZE = 320
SGRAM_DIM = FFT_SIZE // 2 + 1
HOP_LENGTH = 160
WIN_LENGTH = 320
SCALES = [0.25, 0.5, 1, 10, 50, 100, 1000]
BANDS = [0, 1, 2, 3, 4, 5, 31, 40, 41]

DATA_DIR = '../../MOS_Quality_INTERSPEECH_2020/16k_speech'  # TODO
PERTURB_DIR = 'perturbed_iucosine'
AVG_AMP_FILE = 'avg_ampl_iucosine.csv'
PREDS = 'iucosine.csv'

DATASET_STATS = 'mosnet_distr_test.pkl'

BASE_DIR = '../mosnet/results/shap'
PDP_DIR = '../mosnet/results/pdp'
PERTURB_DF = 'picked_perturbed.csv'

PRACTICE_SAMPLES = ['../../data/practice_samples/session-001_1278-F_array1_98.wav',
                    '../../data/practice_samples/session-001_1278-F_close_79.wav']

########## AUDIO PROCESSING ##########
def get_spectrograms(sound_file, fs=FS, fft_size=FFT_SIZE, phase=False, transpose=True): 
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

    if phase:
        mag, phase = librosa.magphase(linear)
        return mag, phase
    
    else:
        # magnitude spectrogram
        mag = np.abs(linear) #(1+n_fft/2, T)
        
        if transpose:
            # shape in (T, 1+n_fft/2)
            return np.transpose(mag.astype(np.float32))
        else:
            return mag 

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

def get_avg_mag(file):
    # rows = files, columns = dB magnitude by frequency bin
    avg_mags = {i: None for i in range(NUM_FREQBINS)}
    spec = get_spectrograms(os.path.join(DATA_DIR, file))
    avg_mag = np.mean(spec, axis=0)
    avg_mag_dB = to_dB(avg_mag)
    
    for i in range(NUM_FREQBINS):
        avg_mags[i] = avg_mag_dB[i]
        
    return avg_mags   

########## GET AVERAGE AMPLITUDE BY FREQUENCY BIN ##########
def get_avg_ampl_by_freq_bin():
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
        
    print('Done.')


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
    
########## DATASET DISTRIBUTIONS ##########
def get_dataset_distr(df, model='mosnet'):
    """Given a list of files, returns distributions for the truncated true MOS and predicted MOS.
    
    Args:
        files (list): a list of files

    Returns:
        tuple (dict, dict): a tuple with a dict for each distribution:
            - {TMOS_VAL: int} (true mos)
            - {PMOS_VAL: int} (predicted mos)
    """
    test_df = pd.read_csv(DATA)
    test_df = test_df[test_df['set'] == 'test']
    test_files = test_df['filename'].tolist()
    
    # get info and save
    tmos_dict_all, pmos_dict_all = get_distr(test_df)
    pickle.dump([tmos_dict_all, pmos_dict_all], open(DATASET_STATS, 'wb'))

def get_distr(df, model='mosnet'):
    tmos_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    pmos_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        tmos = int(row['scaled_mos'])
        pmos = int(row[model+'_pred'])
        tmos_dict[tmos] = tmos_dict.get(tmos, 0) + 1
        pmos_dict[pmos] = pmos_dict.get(pmos, 0) + 1
        
    return tmos_dict, pmos_dict

########## FUNDAMENTAL FREQUENCY ##########
def get_f0():
    outfile = 'f0.csv'
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith('.wav')]

    if os.path.exists(outfile):
        df = pd.read_csv(outfile)
        
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
            df.to_csv(outfile, index=False)     

        df.to_csv(outfile, index=False)

########## GET IUCOSINE DATA ##########
def get_iucosine():
    batch_folder = '/fs/ess/PAS2301/Data/Student_Data/xuandong/IU_Server_Files/Python_Project/listening_study_code/approved_batch_results_csv_COSINE'   # TODO
    files = glob.glob(f"{batch_folder}/*.csv")

    df = pd.read_csv(files[0])
    for f in tqdm(files[1:]):
        df2 = pd.read_csv(f)
        df = pd.concat([df, df2], ignore_index=True)

    # filter out columns we don't want    
    info_cols = ['WorkerId', 'AssignmentStatus']
    question_cols_regex = 'Input.quality_eval*|Answer.quality_set*'
    question_cols = df.filter(regex=question_cols_regex).columns.tolist()

    # standardize column names
    new_names = {}
    for q in question_cols:
        if 'Input' in q:
            new_names[q] = f"{q.split('.')[0]}_{q.split('_')[3]}_{q.split('_')[4]}"
        else:
            new_names[q] = f"{q.split('.')[0]}_{q.split('_')[1]}_{q.split('_')[2]}" 

    df = df.rename(columns=new_names)
    df = df[info_cols + list(new_names.values())]

    df['ratings'] = None
    df['files'] = None

    # make parallel lists of files/ratings
    for row_idx, row in tqdm(df.iterrows(), total=len(df)):
        files_ratings = {}
        for i in range(1, 15):
            for j in range(1, 4):
                files_ratings[row[f"Input_set{i}_cond{j}"].split('/')[-1]] = row[f"Answer_set{i}_cond{j}"]
        
        files, ratings = zip(*files_ratings.items())
        df.at[row_idx, 'files'] = list(files)
        df.at[row_idx, 'ratings'] = list(ratings)
        

    # drop question cols
    df = df.drop(columns=list(new_names.values()))

    # save
    df.to_csv(os.path.join(os.getcwd(), 'all_IUCOSINE_responses.csv'), index=False)
    
########## GET MAGNITUDE DISTRIBUTION ##########
def get_mag_distr(calc=False, delete=False):
    if CALC:
        df = pd.read_csv(DATA)
        avg_mag_db_distr = defaultdict(list)
    
        tmos = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            avg_mags = get_avg_mag(row['filename'])
            tmos.append(row['scaled_mos'])
            for i in range(len(avg_mags)):
                avg_mag_db_distr[i].append(avg_mags[i])
            
        pickle.dump((avg_mag_db_distr, tmos), open('mag_stats.pkl', 'wb'))

    else:
        avg_mag_db_distr, tmos = pickle.load(open('mag_stats.pkl', 'rb'))
    
    if len(BANDS) > 0:
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
                
        if delete:
            print('Deleting .png files...')
            for png in tqdm(pngs):
                if os.path.exists(png):
                    os.remove(png)
    
def plot(data, fbin, tmos):
    # set up figure
    fig, ax = plt.subplots(dpi=600, layout='constrained')
    ax.set_xlabel('Magnitude (dB)')
    ax.set_ylabel('Count')
    start = int(fbin)*50
    end = int((fbin)+1)*50
    ax.set_title(f"Average Magnitude in dB\nof Frequency Bin {fbin} ({start}-{end} Hz)")
    
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
    
    
########## AUDIO PERTURBATION ##########
def perturb_audio(f, band, scale):
    # scale audio
    mag, phase = get_spectrograms(os.path.join(DATA_DIR, f), phase=True)
    mag[band, :] *= scale

    # reconstruct audio and save
    modified = mag * phase
    y_mod = librosa.istft(modified, 
                        n_fft=FFT_SIZE,
                        hop_length=HOP_LENGTH,
                        win_length=WIN_LENGTH,
                        window=scipy.signal.windows.hamming)

    outpath = os.path.join('perturbed_iucosine/' + f.split('.wav')[0] + '_freqbin{}_scaled{}.wav'.format(band, scale))
    sf.write(outpath, y_mod, FS)
    
########## PICK FILES TO PERTURB ##########
def pick_perturb():
    # get already perturbed files
    perturbed_f = [f.split('/')[-1] for f in glob.glob('perturbed_iucosine/*.wav')]
    orig_f = list(set(['_'.join(f.split('_')[0:3]) + '.wav' for f in perturbed_f]))

    x_files = []
    # Get distribution of test files
    tmos_distr, pmos_distr = pickle.load(open(DATASET_STATS, 'rb'))

    test_df = pd.read_csv(PREDS)
    test_df = test_df[test_df['set'] == 'test']     # restrict to test set
    test_df = test_df[test_df['mos5'].notnull()]    # ignore any perturbed files (i.e., we don't have a true MOS)
    test_df['trunc'] = test_df['mos5'].apply(lambda x: int(x))
    existing_df = test_df[test_df['filename'].isin(orig_f)]

    # Get all completed analyses in output directory (i.e., our observed instances)
    observed = {1: [], 2: [], 3: [], 4: [], 5: []}
    observed_files = orig_f
    true_mos_df = pd.read_csv(PREDS)
    remove_idx = []
    test_files = test_df['filename'].tolist()
    for f in observed_files:
        remove_idx.append(test_files.index(f))
            
        # now get associated true MOS
        tmos = int(true_mos_df.loc[true_mos_df['filename'] == f]['mos5'].values[0])
        observed[tmos].append(f)

    # Remove already-analyzed files from consideration and get associated tMOS
    eligible = test_df.drop(index=remove_idx)
    eligible_files = list(set(test_files) - set(observed_files))
    eligible_cat = {1: [], 2: [], 3: [], 4: [], 5: []}
    for ef in eligible_files:
        cat = int(true_mos_df.loc[true_mos_df['filename'] == ef]['mos5'].values[0])
        eligible_cat.get(cat, []).append(ef)
        
    # Sample [args.number] files from remaining test files, 
    # following distribution of all test files
    goal_sample_sizes = {k: 25-len(observed[k]) for k in observed.keys()}
    print(goal_sample_sizes)

    for k,v in goal_sample_sizes.items():
        if v <= 0: 
            # randomly select some from what we've already used
            x_files.extend(random.sample(observed[k], 25))
        elif len(eligible_cat[k]) > 0:
            eligible_files = eligible_cat[k]
            x_files.extend(random.sample(eligible_files, v))
            
    x_test = test_df[test_df['filename'].isin(x_files)]
    with open('todo.txt', 'w') as f:
        space_sep = ' '.join(str(item) for item in x_test['filename'].tolist())
        f.write(space_sep)

    x_test = pd.concat([x_test, existing_df], ignore_index=True)
    print(len(x_test))
    print('Counts:', x_test['trunc'].value_counts())

    with open('picked_iucosine_test_perturb.txt', 'w') as f:
        # Convert all elements to strings and join them with a space
        space_separated_string = " ".join(str(item) for item in x_test['filename'].tolist())
        f.write(space_separated_string)
        
########## PLOT PERTURBATIONS ##########
def plot_perturb(delete=False):
    COLORS = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
    MARKERS = ['o', 's', 'P', '^', 'd', 'X']
    TRENDLINE = True
    SCALES = [0.25, 0.5, 1, 10.0, 50.0, 100.0, 1000.0]
    # BINS = [0, 1, 2, 3, 4, 5, 31, 40, 41]
    DISPLAY_SCALES = [0.25, 0.5, 1, 10, 50, 100, 1000]

    df = pd.read_csv(PREDS)
    perturb = pd.read_csv(PERTURB_DF)['filename'].tolist()
    size_unique = len(perturb)
    models = ['MOSNet', 'DNSMOS', 'SCOREQ']
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

########## PLOT SHAP ##########
def plot_shap():
    TO_PLOT = {'Prediction vs. Scale': False,
               'Average SHAP vs. Frequency Band': True}
    
    MOSNET_SHAP = '../mosnet/results/shap'
    DNSMOS_SHAP = '../dnsmos/results/shap'
    SCOREQ_SHAP = '../scoreq/results/shap'
    
    iucosine_df = pd.read_csv(PREDS)
    
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
        spec = get_spectrograms(os.path.join(DATA_DIR, filename), transpose=False)
        
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

########## SCALE ROOT MEAN SQUARE OF AUDIO ##########
def scale_rms(check=False, desired_rms=0.05):
    files = glob(PERTURB_DIR + '/*.wav', recursive=True)
    
    if check:
        check_rms(files, desired_rms)
    else:
        # scale practice files to 0.05
        #scale(PRACTICE_SAMPLES[0], desired_rms)
        #scale(PRACTICE_SAMPLES[1], desired_rms)

        # get all files in samples directory
        for f in tqdm(files):
            scale(f, desired_rms)
            
def check_rms(files, desired_rms):
    ''' Checks that the RMS of all provided files are near that of the desired value. 
    
    Args:
        files (list): list of files to check
        desired_rms (float): the desired root mean square value for the files

    Returns:
        list: files that are not within 1e-06 of the desired RMS
    '''
    failed_checks = []
    
    print('Checking...')
    for f in tqdm(files):
        signal, sr = librosa.load(f)
        rms = math.sqrt(np.mean(signal**2))
        
        tol = 0.001
        if (rms > desired_rms+tol) or (rms < desired_rms-tol):
            failed_checks.append((f.split('/')[-1], rms))
    
    print('Desired RMS: {}'.format(desired_rms))
    print('Tolerance: 0.001')
    print('Failed checks: {}'.format(failed_checks))
    
def scale(filepath, desired_rms):
    ''' Scales the audio file located at filepath such that it has root mean square value desired_rms.

    Args:
        filepath (str): location of the audio file to scale
        desired_rms (float): the desired root mean square value
    
    Updates:
        modifies the audio file located at filepath
    '''
    #print('Scaling...')
    # RMS_desired = sqrt(1/n sum((scale*signal)^2))
    signal, sr = librosa.load(filepath)
    rms = math.sqrt(np.mean(signal**2))
    
    # scale = sqrt((n * RMS^2_desired)/signal^2)    
    #scale = np.sqrt((len(signal) * (desired_rms**2))/signal**2)

    signal = signal/rms * desired_rms

    # save the signal
    sf.write(filepath, signal, sr, 'PCM_24')
    
########## REMOVE LOW PROBABILITY F0 CALCULATIONS ##########
def remove_low_prob_f0(datafile='f0.csv', prob_threshold=0.5):
    df = pd.read_csv(datafile)

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
        indices = np.argwhere(probs < prob_threshold).flatten()
        
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
            df.to_csv(datafile.split('.csv')[0] + '_threshold.csv', index=False)
    
    # final save 
    df.to_csv(datafile.split('.csv')[0] + '_threshold.csv', index=False)

    # display new stats
    print('After thresholding:')
    print('Minimum f0: {:.3f} Hz'.format(np.nanmin(df['fmin'].tolist())))
    print('Average f0: {:.3f} Hz'.format(np.nanmean(df['favg'].tolist())))
    print('Maximum f0: {:.3f} Hz'.format(np.nanmax(df['fmax'].tolist())))

def clean_convert(str_array):
    str_array = str_array.strip()
    str_array = str_array.replace('nan', 'None')
    str_array = re.sub(r'\s+', ',', str_array)
    if str_array[0:2] == '[,':
        str_array = '[' + str_array[2:]
    return np.array(ast.literal_eval(str_array), dtype=float)
