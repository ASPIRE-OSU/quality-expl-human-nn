import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import pickle
import ast
import numpy as np
from scipy.interpolate import interp1d
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.patches import Patch
import os

VAL_SHAP_DB_FILE = 'spec_dB_values.xlsx'
DEL_FILES = True
ANNOTATE = True

def to_dB(ampl):
    return 20*np.log10(ampl)

def from_dB(dB):
    return [10 ** (d / 20) for d in dB]


def alt_plot(line, ax):
    x_data = list(line.get_xdata())
    x_data = [ast.literal_eval(x) for x in x_data]
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
        

def plot(idx, df):
    file = '/fs/ess/PAS2301/Data/Student_Data/abarach/quality/omnixai/mosnet/results/pdp/pdp_expl_global_freqbin_{}_plots.pkl'.format(idx)
    
    with open(file, 'rb') as f:
        fig = pickle.load(f)
        fig.set_size_inches([11, 8.5])
    
    for ax in fig.get_axes():
        for line in ax.get_lines():
            x_data = list(line.get_xdata())
            if isinstance(x_data[0], str):
                x_data = [float('-inf') if x == '-inf' else ast.literal_eval(x) for x in x_data]
            y_data = list(line.get_ydata())
            
            if idx in [0, 6, 8]:
                x_data, y_data, fig, ax = alt_plot(line, ax)
            
            row = df.iloc[idx]
            color = ['r', 'g', 'b']
            xs = [row['low dB'], row['med dB'], row['high dB']]
            anns = ['{:.4f}\n{:.4f} dB\nSHAP: {:.4f}'.format(row['low val'], row['low dB'], row['low SHAP']),
                    '{:.4f}\n{:.4f} dB\nSHAP: {:.4f}'.format(row['med val'], row['med dB'], row['med SHAP']),
                    '{:.4f}\n{:.4f} dB\nSHAP: {:.4f}'.format(row['high val'], row['high dB'], row['high SHAP'])]
            markers = ['ro', 'go', 'bo']
            inter = interp1d(x_data, y_data, kind='linear', bounds_error=False, fill_value='extrapolate')
            y = inter(xs)
            
            for m in range(len(markers)):
                ax.plot(xs[m], y[m], markers[m])
                plt.annotate(anns[m], (xs[m], y[m]), ha='center', fontsize=10, color=color[m], xytext=(0,5), textcoords="offset points")  

            ax.set_title(ax.get_title())
            ax.set_xlabel('Magnitude (dB)')
            
            # legend
            custom_patches = [Patch(facecolor='red', label='1.75 true MOS (2.363 predicted)'),
                              Patch(facecolor='green', label='3.0 true MOS (2.904 predicted)'),
                              Patch(facecolor='blue', label='4.75 true MOS (3.356 predicted)')]
            plt.legend(handles=custom_patches)            
            
            ax.relim()
            ax.autoscale_view()
            fig.savefig(file.split('.pkl')[0] + '.png')
            
    return file.split('.pkl')[0] + '.png'
         
df = pd.read_excel(VAL_SHAP_DB_FILE)
df.drop(columns=['Unnamed: 0'], inplace=True)
success = []
failed = []

for i in tqdm(range(48)):
    try:
        f = plot(i, df)
        success.append(f)
    except Exception as e:
        failed.append(i)
        continue

print('Failed: ', failed)

print('Converting to PDF...')
pngs = [Image.open(i).convert('RGB') for i in success]
pngs[0].save('all.pdf', save_all=True, append_images=pngs[1:], quality=100)

if DEL_FILES:
    print('Deleting .png files...')
    for png in tqdm(success):
        if os.path.exists(png):
            os.remove(png)