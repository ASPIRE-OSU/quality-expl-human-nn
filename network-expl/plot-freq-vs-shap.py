import os
import argparse
import pandas as pd
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
from mpl_axes_aligner import align
import matplotlib.colors as mcolors
plt.rcParams.update({'font.size': 16})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm'

PDP_DIR = {'DNSMOS': os.path.join(os.getcwd(), 'dnsmos/results/pdp'),
           'MOSNet': os.path.join(os.getcwd(), 'mosnet/results/pdp'),
           'SCOREQ': os.path.join(os.getcwd(), 'scoreq/results/pdp')}
SHAP_DIR = {'DNSMOS': os.path.join(os.getcwd(), 'dnsmos/results/shap'),
            'MOSNet': os.path.join(os.getcwd(), 'mosnet/results/shap'),
            'SCOREQ': os.path.join(os.getcwd(), 'scoreq/results/shap')}
DATA_DIR = '../data/MOS_Quality_INTERSPEECH_2020/16k_speech'
PRED_DF = 'data/iucosine.csv'
FS = 16000
FFT_SIZE = 320
SGRAM_DIM = FFT_SIZE // 2 + 1
NUM_FREQBINS = SGRAM_DIM
HOP_LENGTH = 160
WIN_LENGTH = 320
INDICES = {'DNSMOS': [0, 1, 2, 3, 4, 5, 14, 15, 26, 31, 40, 41],
           'MOSNet': [0, 1, 2, 3, 4, 5, 6, 12, 17, 31, 40, 41],
           'SCOREQ': [0, 1, 2, 3, 4, 5, 20, 21, 26, 31, 40, 41]}
DEL = False
plt.style.use('tableau-colorblind10')
COLORS = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
MARKERS = ['o', 's', 'P', '^', 'd', 'X']
TRENDLINE = True

def main(args):
    print('Collecting SHAP values...')
    shap_df = collect_shap(SHAP_DIR[args.model], args.model)
    averages = {i: shap_df.loc[:, i].mean() for i in shap_df.columns.tolist()[1:]}
    #print(len(shap_df))
    #print(averages)

    # plot shap v frequency
    plot_shapvfreq(shap_df, args.model)

    # print('Collecting average magnitudes...')
    # avg_mag_df_path = os.path.join(SHAP_DIR[args.model], 'avg_mag_df.csv')
    # if os.path.exists(avg_mag_df_path):
    #     avg_mag_df = pd.read_csv(avg_mag_df_path)
    # else:
    #     avg_mag_df = get_mags(shap_df)
    #     print(avg_mag_df.head)
    #     avg_mag_df.to_csv(avg_mag_df_path, index=False)
        
    # print('Collecting predicted MOS...')
    # pred_df = pd.read_csv(PRED_DF)
    # predMOS, pred_color, pred_markers = collect_mos(shap_df, pred_df, args.model, type='pred')
    # # trueMOS, true_color, true_markers = collect_mos(shap_df, pred_df, type='true')
    
    # print('\nPredicted MOS')
    # # #shap_df.to_csv('shap_df.csv', index=False)
    # # #avg_mag_df.to_csv('avg_mag_df.csv', index=False)
    # plot(shap_df, avg_mag_df, 'all_mag_shap_pred.pdf', args.model, mos=predMOS, color=pred_color, markers=pred_markers, type='pred')
    # # #print('\nTrue MOS')
    # # #plot(shap_df, avg_mag_df, 'all_mag_shap_true.pdf', mos=predMOS, color=true_color, markers=true_markers, type='true')

    print('\nDone.')

        
def collect_shap(base_dir, model):
    # shap_files = [os.path.join(BASE_DIR, f) for f in os.listdir(BASE_DIR) if f.endswith('feat_scores.csv')]
    
    # # rows = files, columns = SHAP values by frequency bin
    # shap_df = pd.DataFrame(columns=['filename'] + ['freqbin_{}'.format(i) for i in range(NUM_FREQBINS)])
    # for sf in tqdm(shap_files):
    #     df = pd.read_csv(sf)
    #     df = df.rename(columns={'file': 'filename'})
    #     # reorder columns to match shap_df
    #     df = df[shap_df.columns.tolist()]
    #     shap_df = pd.concat([shap_df if not shap_df.empty else None, df], ignore_index=True)
    # return shap_df
    shap_df = pd.read_csv(os.path.join(base_dir, f'shap_expl_local_n10_agg_{model.lower()}.csv'))
    return shap_df


def get_mags(shap_df):
    # rows = files, columns = dB magnitude by frequency bin
    avg_mag_df = pd.DataFrame(columns=shap_df.columns)
    for _, row in tqdm(shap_df.iterrows(), total=len(shap_df)):
        filename = row['file'].split('_feat_scores.csv')[0]
        spec = get_spectrograms(os.path.join(DATA_DIR, filename))
        avg_mag = np.mean(spec, axis=0)
        avg_mag_dB = to_dB(avg_mag)
        new_row = pd.DataFrame([[filename] + list(avg_mag_dB)], columns=avg_mag_df.columns)
        avg_mag_df = pd.concat([avg_mag_df if not avg_mag_df.empty else None, new_row], ignore_index=True)
    return avg_mag_df    
        
        
def collect_mos(shap_df, pred_df, model, type='true'):
    # columns = set,filename,mos5,mosnet_pred,ratings,mos100,filtered,scaled_ratings,raters,type,mos10,zscores,dbscan,if,dbscan_if_decision
    color_data_mos = []
    marker_data_mos = []
    preds = []
    for _, row in tqdm(shap_df.iterrows(), total=len(shap_df)):
        filename = row['file'].split('_feat_scores.csv')[0]
        if type == 'pred':
            pred_mos = pred_df[pred_df['filename'] == row['file']][f'{model.lower()}_pred'].values[0] # truncate
        elif type == 'true':
           pred_mos = int(pred_df[pred_df['filename'] == row['file']]['mos5'].values[0]) # truncate
        preds.append(pred_mos)
        color_data_mos.append(COLORS[int(pred_mos)-1])
        marker_data_mos.append(MARKERS[int(pred_mos)-1])
    return preds, color_data_mos, marker_data_mos


def plot(shap_df, avg_mag_df, pdfoutfile, model, mos=None, color=None, markers=None, type='pred'):
    print('Plotting...')
    pdp_agg = {}

    for i in tqdm(INDICES[model]):
        x_data = np.array(avg_mag_df['freqbin_{}'.format(i)].tolist())
        y_data = np.array(shap_df['freqbin_{}'.format(i)].tolist())
        
        # set up figure
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(28, 8), dpi=600)
        freqbandwidth = get_range(i, get_width())
        
        # two plots: left is SHAP values, right is PDP with MOS predictions
        # left figure: SHAP values
        axL.set_xlabel('Average Magnitude (dB)')
        axL.set_ylabel('SHAP Value')
        axL.set_title(f'SHAP Values for Frequency Band {i}\n({freqbandwidth[0]:.0f} - {freqbandwidth[1]:.0f} Hz)')
        axL.grid()
        mscatter(x_data, y_data, c=color, m=markers, ax=axL)
        
        # right figure: PDP
        axR.set_xlabel('Average Magnitude (dB)')
        axR.set_ylabel('Prediction (MOS)')
        axR.set_title(f'PDP for Frequency Band {i}\n({freqbandwidth[0]:.0f} - {freqbandwidth[1]:.0f} Hz)')
        axR.grid()
        
        # plot PDP
        pdp_x, pdp_y = get_pdp('pdp_expl_global_freqbin_{}_plots.pkl'.format(i), fig, i, model)
        pdp_agg[i] = pdp_y
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
            legend_elements = [Line2D([0], [0], marker=MARKERS[i], color='w', label='{}'.format(i+1), markerfacecolor='{}'.format(COLORS[i])) for i in range(len(MARKERS))]
            legend_elements.append(Line2D([0], [0], linestyle='-', color='k', label='PDP curve'))
            legend_elements.append(Line2D([0], [0], linestyle='--', color='k', label='SHAP trend line'))
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
            
            axL.plot(x_data, y_trend, color='k', linestyle='--', lw=2)
            
            # correlate trendline against pdp curve
            pcc, p_val = stats.pearsonr(pdp_y, y_trend_pdp_x)
            text = f"SHAP trend line:\n$y={slope:0.3f}\;(20\log x){intercept:+0.3f}$\n$R^2 = {r**2:0.3f}$\n\nCorrelation between SHAP\ntrendline and PDP curve:\n$PCC = {pcc:0.3f}$\n$p-value={p_val:0.3e}$"
            y_offset = -0.6
            axR.text(1.1, 1 + y_offset, text, bbox=dict(facecolor='w', edgecolor='w'), transform=axR.transAxes, verticalalignment='top')
          
        fig.savefig(os.path.join(SHAP_DIR[model], outfile), bbox_inches='tight')
        #pngs.append(os.path.join(BASE_DIR, outfile))
        plt.close()
    
    pdp_agg_df = pd.DataFrame.from_dict(pdp_agg, orient='index')
    pdp_agg_df.to_csv(os.path.join(PDP_DIR[model],'agg_pdp.csv'), index=False)
 
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


def get_pdp(filename, fig, i, model):
    pdp_file = os.path.join(PDP_DIR[model], filename)
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
        
        # temp fix - PDP plots were produced with 10 as the coefficient, need to multiply by 2 to get to 20*log10
        if model == 'MOSNet':
            pdp_x = [2 * x for x in pdp_x]
        
        #if i in [1]:
        #    pdp_x, pdp_y, pdp_fig, pdp_ax = alt_plot(pdp_line, pdp_fig.get_axes()[0])
            
        plt.close(pdp_fig)
    return pdp_x, pdp_y

def get_range(i, width):
    return (i*width, i*width + width)


def get_width():
    max_f = FS/2
    bandwidth = max_f/(NUM_FREQBINS-1)
    return bandwidth


def get_spectrograms(sound_file, fs=FS, fft_size=FFT_SIZE):         
    import librosa

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



def alt_plot(line, ax):
    x_data = list(line.get_xdata())
    x_data = [float('-inf') if x == '-inf' else ast.literal_eval(x) for x in x_data]
    x_data_db = to_dB(x_data)
    y_data = list(line.get_ydata())
    y_data = from_dB(y_data)

    newfig, newax = plt.subplots()
    newfig.set_size_inches([11, 8.5])
    newax.plot(x_data_db, y_data)
    newax.set_xlabel('Magnitude (dB)')
    newax.set_ylabel(ax.get_ylabel())
    newax.set_title('Partial Dependence Plot for Frequency Band {}\nCenter: {} Hz (±{} Hz)'.format(0, 0, 31.25/2))
    newax.grid()
    
    return x_data_db, y_data, newfig, newax


def to_dB(ampl, already_log=False):
    if already_log: 
        return 10*ampl
    else:
        return 20*np.log10(ampl)


def from_dB(dB):
    return [10 ** (d / 20) for d in dB]


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


def _parse_args():
    """
    Reads and stores command line arguments.
    
    Returns:
       args parser object
    """
    parser = argparse.ArgumentParser(prog='SHAP vs. PDP plotter')
    parser.add_argument('-m', '--model', required=True, type=str, help='DNSMOS, MOSNet, or ScoreQ')
    return parser.parse_args()

def plot_shapvfreq(shap_df, model):
    freqcols = [col for col in shap_df.columns if col.startswith('freqbin')]
    mean_shap = shap_df[freqcols].mean()
    mean_abs_shap = shap_df[freqcols].abs().mean()
    x = np.arange(1, len(freqcols) + 1)
    fig, ax = plt.subplots(figsize=(11, 8), dpi=600)
    ax.bar(x, mean_shap.values, label='Average SHAP Value', color='tab:blue')
    ax.set_xlabel('Frequency Band (width = 50 Hz)')
    ax.set_ylabel('Average SHAP Value')
    ax.set_xlim(0, len(freqcols)+1)
    #ax2 = ax.twinx()
    #ax2.plot(x, mean_abs_shap.values, linestyle='--', label='Average SHAP Magnitude (|SHAP|)', color='tab:orange')
    ax.plot(x, mean_abs_shap.values, linestyle='--', label='Average SHAP Magnitude (|SHAP|)', color='tab:orange')
    #ax2.set_ylabel('Average SHAP Magnitude (|SHAP|)')
    plt.title(f'Average SHAP Value vs. Frequency Band for {model}')
    lines, labels = ax.get_legend_handles_labels()
    #lines2, labels2 = ax2.get_legend_handles_labels()
    #ax.legend(lines + lines2, labels + labels2, loc='upper right')
    ax.legend()
    #align.yaxes(ax, 0, ax2, 0)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR[model], f'shapvfreq_{model.lower()}.png'))
    print('Done.')   


if __name__ == '__main__':
    main(_parse_args())