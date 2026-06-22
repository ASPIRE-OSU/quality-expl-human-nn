import matplotlib.pyplot as plt
import ast
from tqdm import tqdm
import pickle

dict_file = 'results/pdp/pdp_expl_global_freqbin_0-5_raw.txt'

with open(dict_file, 'r') as f:
    data = ast.literal_eval(f.read())

for feature, data in tqdm(data.items()):
    feature_num = int(feature.split('_')[-1])
    outfile = '_'.join(dict_file.split('_')[:-3]) + '_{}'.format(feature)

    x = data['values']
    y = data['scores']
    
    # x-axis (values) = log_10 (max(|STFT(f,t)|^2, 10^-12))
    # goal: 10 * log_10 (max(|STFT(f,t)|^2, 10^-12))
    # todo: multiply by 10 before graphing
    x = [10 * i for i in  x]
    
    # replot
    fig, axes = plt.subplots(1, 1, squeeze=False)
    plt.sca(axes[0, 0])
    plt.plot(x, y)
    plt.xlabel('Magnitude (dB)')    # note: power (dB) = magnitude (dB)
    plt.ylabel('Prediction (MOS)')
    plt.title('Partial Dependence Plot for Frequency Band {}\n{} - {} Hz'.format(feature_num, feature_num*50, feature_num*50+50))
    plt.savefig(outfile + '.png', bbox_inches='tight')
    
    with open(outfile + '_plots.pkl', 'wb') as of:
        pickle.dump(fig, of, protocol=pickle.HIGHEST_PROTOCOL)