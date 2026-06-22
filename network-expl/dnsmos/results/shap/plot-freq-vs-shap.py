import os
import pandas as pd
import librosa
import numpy as np
import scipy.signal
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from scipy.interpolate import interp1d
from scipy import stats
import ast
import pickle
np.seterr(divide='ignore')
import soundfile as sf

from mpl_axes_aligner import align
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm'

PDP_DIR = '../pdp'
DATA_DIR = '../../../data/MOS_Quality_INTERSPEECH_2020/16k_speech'
PRED_DF = '../../data/iucosine.csv'
FS = 16000
# FFT_SIZE = 512
NFFT = FFT_SIZE = 320
SGRAM_DIM = FFT_SIZE // 2 + 1
NUM_FREQBINS = SGRAM_DIM
# HOP_LENGTH = 256
HOP_LENGTH = 160
# WIN_LENGTH = 512
WIN_LENGTH = 320
INDICES = [0, 1, 2, 3, 4, 5, 14, 15, 26, 31, 40, 41]
DEL = False
COLORS = ['red', 'darkorange', 'green', 'dodgerblue', 'darkorchid']
MARKERS = ['o', 's', 'P', '^', 'd', 'X']
TRENDLINE = True

def main():
    print('Collecting SHAP values...')
    shap_df = collect_shap()
    averages = {i: shap_df.loc[:, i].mean() for i in shap_df.columns.tolist()[1:]}
    #print(len(shap_df))
    #print(averages)
     
    # plot shap v frequency
    freqcols = [col for col in shap_df.columns if col.startswith('freqbin')]
    mean_shap = shap_df[freqcols].mean()
    mean_abs_shap = shap_df[freqcols].abs().mean()
    x = np.arange(1, len(freqcols) + 1)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=600)
    ax.bar(x, mean_shap.values, label='Average SHAP Value', color='tab:blue')
    ax.set_xlabel('Frequency Band (width = 50 Hz)')
    ax.set_ylabel('Average SHAP Value')
    ax.set_xlim(0, len(freqcols)+1)
    ax2 = ax.twinx()
    ax2.plot(x, mean_abs_shap.values, linestyle='--', label='Average SHAP Magnitude (|SHAP|)', color='tab:orange')
    ax2.set_ylabel('Average SHAP Magnitude (|SHAP|)')
    plt.title('Average SHAP Value vs. Frequency Band for DNSMOS')
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc='upper right')
    align.yaxes(ax, 0, ax2, 0)
    plt.tight_layout()
    plt.savefig('shapvfreq_dnsmos.png')
        
    #print('Collecting average magnitudes...')
    # avg_mag_df = get_mags(shap_df)
    # #print(avg_mag_df.head)
    # avg_mag_df.to_csv('shap_avg_mag.csv', index=False)
        
    # print('Collecting predicted MOS...')
    # pred_df = pd.read_csv(PRED_DF)
    # predMOS, pred_color, pred_markers = collect_mos(shap_df, pred_df, type='pred')
    # trueMOS, true_color, true_markers = collect_mos(shap_df, pred_df, type='true')
    
    # print('\nPredicted MOS')
    # shap_df.to_csv('shap_df.csv', index=False)
    # avg_mag_df.to_csv('avg_mag_df.csv', index=False)
    # plot(shap_df, avg_mag_df, 'all_mag_shap_pred.pdf', mos=predMOS, color=pred_color, markers=pred_markers, type='pred')
    #print('\nTrue MOS')
    #plot(shap_df, avg_mag_df, 'all_mag_shap_true.pdf', mos=predMOS, color=true_color, markers=true_markers, type='true')

        
def collect_shap():
    # shap_files = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith('feat_scores.csv')]
    
    # # rows = files, columns = SHAP values by frequency bin
    # shap_df = pd.DataFrame(columns=['file'] + ['freqbin_{}'.format(i) for i in range(NUM_FREQBINS)])
    # for sf in tqdm(shap_files):
    #     df = pd.read_csv(sf)
    #     df = df.rename(columns={'file': 'file'})
    #     # reorder columns to match shap_df
    #     df = df[shap_df.columns.tolist()]
    #     shap_df = pd.concat([shap_df if not shap_df.empty else None, df], ignore_index=True)
    # return shap_df
    shap_df = pd.read_csv(os.path.join(BASE_DIR, 'shap_expl_local_n10_agg_dnsmos.csv'))
    return shap_df


def get_mags(shap_df):
    # rows = files, columns = dB magnitude by frequency bin
    avg_mag_df = pd.DataFrame(columns=shap_df.columns)
    for _, row in tqdm(shap_df.iterrows(), total=len(shap_df)):
        file = row['file'].split('_feat_scores.csv')[0]
        spec = get_spectrograms(os.path.join(DATA_DIR, file))
        avg_mag = np.mean(spec, axis=0)
        avg_mag_dB = to_dB(avg_mag, already_log=False)
        new_row = pd.DataFrame([[file] + list(avg_mag_dB)], columns=avg_mag_df.columns)
        avg_mag_df = pd.concat([avg_mag_df if not avg_mag_df.empty else None, new_row], ignore_index=True)
    return avg_mag_df    
        
        
def collect_mos(shap_df, pred_df, type='true'):
    # columns = set,file,mos5,mosnet_pred,ratings,mos100,filtered,scaled_ratings,raters,type,mos10,zscores,dbscan,if,dbscan_if_decision
    color_data_mos = []
    marker_data_mos = []
    preds = []
    for _, row in tqdm(shap_df.iterrows(), total=len(shap_df)):
        file = row['file'].split('_feat_scores.csv')[0]
        if type == 'pred':
            pred_mos = pred_df[pred_df['filename'] == row['file']]['mosnet_pred'].values[0] # truncate
        elif type == 'true':
           pred_mos = int(pred_df[pred_df['filename'] == row['file']]['mos5'].values[0]) # truncate
        preds.append(pred_mos)
        color_data_mos.append(COLORS[int(pred_mos)-1])
        marker_data_mos.append(MARKERS[int(pred_mos)-1])
    return preds, color_data_mos, marker_data_mos


def plot(shap_df, avg_mag_df, pdfoutfile, mos=None, color=None, markers=None, type='pred'):
    print('Plotting...')

    for i in tqdm(INDICES):
        x_data = np.array(avg_mag_df['freqbin_{}'.format(i)].tolist())
        y_data = np.array(shap_df['freqbin_{}'.format(i)].tolist())
        
        # set up figure
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(28, 8), dpi=600)
        freqbandwidth = get_range(i, get_width())
        
        # two plots: left is SHAP values, right is PDP with MOS predictions
        # left figure: SHAP values
        axL.set_xlabel('Average Magnitude (dB)')
        axL.set_ylabel('SHAP Value')
        axL.set_title('SHAP Values for frequency band {}\n({} - {} Hz)'.format(i, freqbandwidth[0], freqbandwidth[1]))
        axL.grid()
        mscatter(x_data, y_data, c=color, m=markers, ax=axL)
        
        # right figure: PDP
        axR.set_xlabel('Average Magnitude (dB)')
        axR.set_ylabel('Prediction (MOS)')
        axR.set_title('PDP and Predicted MOS for frequency\nband {} ({} - {} Hz)'.format(i, freqbandwidth[0], freqbandwidth[1]))
        axR.grid()
        
        # plot PDP
        pdp_x, pdp_y = get_pdp('pdp_expl_global_freqbin_{}_plots.pkl'.format(i), fig, i)
        axR.plot(pdp_x, pdp_y, 'k-')
    
        # plot predicted mos
        #mscatter(x_data, mos, c=color, m=markers, ax=axR)
                
        # limit y-axis for PDP/predicted if outside MOS range
        bottom_limit, top_limit = axR.get_ylim()
        if bottom_limit < 1: bottom_limit = 1
        if top_limit > 5: top_limit = 5
        axR.set_ylim(bottom_limit, top_limit)      # limit PDP range to valid MOS scores
        
        # plot and label
        #ax.set_title('PDP and SHAP values vs. Average magnitude of\nfrequency band {} ({} - {} Hz)'.format(i, freqbandwidth[0], freqbandwidth[1]))
        
        # get outfile name
        if type == 'pred':
            #ax.set_title('Average magnitude of frequency bin {} vs. SHAP value\nCenter: {} Hz (±{} Hz)\nColored by Predicted MOS\nn={}'.format(i, get_center(i), get_width()/2, len(x_data)))
            outfile = 'mag_shap_pred_{}.png'.format(i)
        elif type == 'true':
            #ax.set_title('Average magnitude of frequency bin {} vs. SHAP value\nCenter: {} Hz (±{} Hz)\nColored by True MOS\nn={}'.format(i, get_center(i), get_width()/2, len(x_data)))
            outfile = 'mag_shap_true_{}.png'.format(i)
        else:
            #ax.set_title('Average magnitude of frequency bin {} vs. SHAP value\nCenter: {} Hz (±{} Hz)\nn={}'.format(i, get_center(i), get_width()/2, len(x_data)))
            outfile = 'mag_shap_{}.png'.format(i)
            
        # add legend if including mos predictions 
        if type is not None:
            legend_elements = [Line2D([0], [0], marker=MARKERS[i], color='w', label='{}'.format(i+1), markerfacecolor='{}'.format(COLORS[i])) for i in range(len(COLORS))]
            legend_elements.append(Line2D([0], [0], linestyle='-', color='k', label='PDP curve'))
            legend_elements.append(Line2D([0], [0], linestyle='--', color='darkgray', label='SHAP trend line'))
            title = 'True MOS' if type == 'true' else 'Predicted MOS'
            plt.legend(title=title, loc='upper left', bbox_to_anchor=(1.1, 1), handles=legend_elements)
            fig.subplots_adjust(right=0.6)
            
        #fig.suptitle('SHAP and PDP values vs. Average magnitude of frequency band {} ({} - {} Hz)'.format(i, freqbandwidth[0], freqbandwidth[1]))
            
        # make x-axes ranges consistent
        xminL, xmaxL = axL.get_xlim()
        xminR, xmaxR = axR.get_xlim()
        xmin = min(xminL, xminR)
        xmax = max(xmaxL, xmaxR)
        axL.set_xlim(xmin, xmax)
        axR.set_xlim(xmin, xmax)
            
        if TRENDLINE:
            # delete nans
            pdp_nan = pdp_x
            pdp_x = [i for i in pdp_nan if str(i) != 'nan']
            pdp_y = [pdp_y[i] for i in range(len(pdp_nan)) if str(pdp_nan[i]) != 'nan']
            #if len(pdp_nan) != len(pdp_x):
                #print('freqbin_{}: removed {} nan values (of {}) from PDP data for trendline calculation.'.format(i, len(pdp_nan)-len(pdp_x), len(pdp_nan)))
            
            x_data, y_data = zip(*sorted(zip(x_data, y_data), key=lambda pair: pair[0]))
            x_data, y_data = list(x_data), list(y_data)
            slope, intercept, r, p, std_err = stats.linregress(from_dB(x_data), y_data)
            y_trend = slope * np.array(from_dB(x_data)) + intercept
            
            if (len(y_trend) != len(pdp_y)):
                # so let's calculate y_trend using pdp_x values
                y_trend_pdp_x = slope * np.array(from_dB(pdp_x)) + intercept
            else:
                y_trend_pdp_x = y_trend
            
            axL.plot(x_data, y_trend, color='darkgrey', linestyle='--', lw=1)
            
            # correlate trendline against pdp curve
            pcc, p_val = stats.pearsonr(pdp_y, y_trend_pdp_x)
            text = f"SHAP trend line:\n$y={slope:0.3f}\;(20\log x){intercept:+0.3f}$\n$R^2 = {r**2:0.3f}$\n\nCorrelation between SHAP\ntrendline and PDP curve:\n$PCC = {pcc:0.3f}$\n$p-value={p_val:0.3e}$"
            y_offset = -0.6
            axR.text(1.1, 1 + y_offset, text, bbox=dict(facecolor='w', edgecolor='w'), transform=axR.transAxes, verticalalignment='top')
          
        fig.savefig(os.path.join(BASE_DIR, outfile), bbox_inches='tight')
        #pngs.append(os.path.join(BASE_DIR, outfile))
        plt.close()
 
    # convert to PDF
    # print('Converting to PDF...')
    # pngs.sort()
    # imgs = [Image.open(i).convert('RGB') for i in pngs]   
    # imgs[0].save(pdfoutfile, save_all=True, append_images=imgs[1:], quality=100)
            
    # if DEL:
    #     print('Deleting .png files...')
    #     for png in tqdm(pngs):
    #         if os.path.exists(png):
    #             os.remove(png)


def get_pdp(file, fig, i):
    pdp_file = os.path.join(PDP_DIR, file)
    if os.path.exists(pdp_file):
        with open(pdp_file, 'rb') as f:
            pdp_fig = pickle.load(f)
            if isinstance(pdp_fig, list):
                pdp_fig = pdp_fig[0]
            pdp_fig.set_size_inches([fig.get_figwidth(), fig.get_figheight()])
        pdp_line = pdp_fig.get_axes()[0].get_lines()[0]
        pdp_x = list(pdp_line.get_xdata())
        if isinstance(pdp_x[0], str):
            pdp_x = [float('-inf') if x == '-inf' else ast.literal_eval(x) for x in pdp_x]
        pdp_y = list(pdp_line.get_ydata())
            
        plt.close(pdp_fig)
    return pdp_x, pdp_y

def get_center(idx):
    bandwidth = get_width()
    center = idx * bandwidth
    return center


def get_range(i, width):
    return (i*width, i*width+width)


def get_width():
    max_f = FS/2
    bandwidth = max_f/(NUM_FREQBINS-1)
    return bandwidth


def audio_logpowspec(audio, nfft=NFFT, hop_length=160, sr=FS):
    # powspec.shape == (..., 1 + nfft/2, n_frames) == (..., 161, # frames)
    powspec = (np.abs(librosa.core.stft(audio, n_fft=nfft, hop_length=hop_length)))**2
    logpowspec = np.log10(np.maximum(powspec, 10**(-12)))
    return logpowspec.T    


def get_spectrograms(sound_file, fft_size=NFFT):         
    # Loading sound file
    #audio, _ = sf.read(os.path.join(DATA_DIR, sound_file))
    #logpowspec = audio_logpowspec(audio)
    #return logpowspec
    # Loading sound file
    y, _ = librosa.load(sound_file, sr=FS) # or set sr to hp.sr.

    # Preemphasis
    #y = np.append(y[0], y[1:] - PREEMPHASIS * y[:-1])

    # stft. D: (1+n_fft//2, T)
    linear = librosa.stft(y=y,
                     n_fft=FFT_SIZE, 
                     hop_length=HOP_LENGTH, 
                     win_length=WIN_LENGTH,
                     window=scipy.signal.windows.hamming,
                     )

    # magnitude spectrogram
    mag = np.abs(linear) #(1+n_fft/2, T)
    
    # shape in (T, 1+n_fft/2)
    return np.transpose(mag.astype(np.float32))


def mscatter(x,y,ax=None, m=None, **kw):
    import matplotlib.markers as mmarkers
    if not ax: ax=plt.gca()
    sc = ax.scatter(x,y,**kw)
    if (m is not None) and (len(m)==len(x)):
        paths = []
        for marker in m:
            if isinstance(marker, mmarkers.MarkerStyle):
                marker_obj = marker
            else:
                marker_obj = mmarkers.MarkerStyle(marker)
            path = marker_obj.get_path().transformed(
                        marker_obj.get_transform())
            paths.append(path)
        sc.set_paths(paths)
    return sc


# assuming dnsmos is already in log power form
def to_dB(ampl, already_log):
    if already_log:
        return 10*ampl
    else:
        return 20*np.log10(ampl)
    


def from_dB(dB):
    return [10 ** (d / 20) for d in dB]


if __name__ == '__main__':
    main()