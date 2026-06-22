"""
Gets the distribution of various factors (e.g., true MOS, system, task) and compares it to the distribution of 
the same factor in the overall dataset.

Note: the whole/overall set here refers to the test set.

@author: Ada Lamba
@version: 07/21/2025
"""
import os
import glob
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from tqdm import tqdm
import pickle

DATASET_STATS = '../../data/mosnet_distr_all.pkl'
TEST_DATA = '../../data/omnixai_mosnet_test.pkl'
MOS_SCORES = '../../data/mosnet_predictions.csv'
OUTPUT_PATH = '../shap'

def main():
    mos_df = pd.read_csv(MOS_SCORES)
    
    # check if we have already generated the whole dataset stats
    if os.path.exists(DATASET_STATS):
        tmos_dict_all, pmos_dict_all, system_dict_all, task_dict_all, target_dict_all, source_dict_all =\
            pickle.load(open(DATASET_STATS, 'rb'))

    # if not, collect all files so we can run same analysis as for shap dataset
    else:
        wholeset_files = list(pd.read_pickle(TEST_DATA).index)
        
        # get info and save
        tmos_dict_all, pmos_dict_all, system_dict_all, task_dict_all, target_dict_all, source_dict_all =\
            get_distr(wholeset_files, mos_df)
        pickle.dump([tmos_dict_all, pmos_dict_all, system_dict_all, task_dict_all, target_dict_all, source_dict_all], open(DATASET_STATS, 'wb'))

    # get all shap result files and get info 
    shap_files = ['_'.join(f.split('/')[-1].split('_')[4:9]) for f in glob.glob(OUTPUT_PATH + '/shap_expl_local_*_feat_scores.csv')]
    tmos_dict_shap, pmos_dict_shap, system_dict_shap, task_dict_shap, target_dict_shap, source_dict_shap =\
        get_distr(shap_files, mos_df)
        
    pairs = [('True MOS (all)', tmos_dict_all, 'True MOS (SHAP)', tmos_dict_shap),
             ('Predicted MOS (all)', pmos_dict_all, 'Predicted MOS (SHAP)', pmos_dict_shap), 
             ('System ID (all)', system_dict_all, 'System ID (SHAP)', system_dict_shap),
             ('Task (all)', task_dict_all, 'Task (SHAP)', task_dict_shap), 
             ('Target (all)', target_dict_all, 'Target (SHAP)', target_dict_shap), 
             ('Source (all)', source_dict_all, 'Source (SHAP)', source_dict_shap)]
        
    # visualize distributions and get t-test values
    analyze(pairs, save=True)

def get_distr(files, mos_df):
    """Given a list of files, returns distributions for the truncated true MOS, predicted MOS, system, task, source, and target. 

    Args:
        files (list): a list of files

    Returns:
        tuple (dict, dict, dict, dict, dict): a tuple with a dict for each distribution:
            - {TMOS_VAL: number} (true mos)
            - {PMOS_VAL: number} (predicted mos)
            - {SYSTEM_ID: number}
            - {TASK: number}
            - {TARGET: number}
            - {SOURCE: number}
    """
    tmos_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    pmos_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    system_dict = {}
    task_dict = {'HUB': 0, 'SPO': 0}
    target_dict = {}
    source_dict = {}
    
    print('Collecting files...')
    for f in tqdm(files):
        # [system_id]_[TRG]_[SRC]_#####_[TASK].[wav/h5]
        sys_id, trg, src, _, task = f.split('.')[0].split('_')
        
        # update dicts
        system_dict[sys_id] = system_dict.get(sys_id, 0) + 1
        target_dict[trg] = target_dict.get(trg, 0) + 1
        source_dict[src] = source_dict.get(src, 0) + 1
        task_dict[task] = task_dict.get(task, 0) + 1
        
        # get (truncated) mos scores
        tmos = int(mos_df.loc[mos_df['file'] == f.split('.')[0]]['true_mos'].values[0])
        pmos = int(mos_df.loc[mos_df['file'] == f.split('.')[0]]['predicted_mos'].values[0])
        tmos_dict[tmos] = tmos_dict.get(tmos, 0) + 1
        pmos_dict[pmos] = pmos_dict.get(pmos, 0) + 1
        
    return tmos_dict, pmos_dict, system_dict, task_dict, target_dict, source_dict


def analyze(pairs, save=True):
    """For each pair provided, plots a double histogram to visually compare the distributions and
    reports the t-test value between each element of the pair.

    Args:
        pairs (list of tuples): a list of tuples where each tuple contains a pair of distributions to compare
                                and has format (description for p1, p1, description for p2, p2)
        save (bool, optional): whether to save the output visualization. Defaults to True. Only the t-test
                               results will be printed if false.
    """
    num_rows = 2
    num_cols = math.ceil(len(pairs)/2)
    fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 12))
    row_ctr = 0
    col_ctr = 0
    
    print('Analyzing...')
    for (d1, p1, d2, p2) in tqdm(pairs):
        categorical = False
        
        # make sure both sets have same number of categories
        if p1.keys() != p2.keys():
            all_keys = set(p1.keys()) | set(p2.keys())
            p1 = {key: p1.get(key, 0) for key in all_keys}
            p2 = {key: p2.get(key, 0) for key in all_keys}
        
        # if numeric: t-test
        if isinstance(list(p1.keys())[0], (int, float, complex)) and not isinstance(list(p1.keys())[0], bool):
            s1 = [item for item, count in p1.items() for _ in range(count)]
            s2 = [item for item, count in p2.items() for _ in range(count)]
            
            # Welch's t-test: do not assume equal variance
            stat, p_val = stats.ttest_ind(s1, s2, equal_var=False)
            
        # if categorical: chi-squared goodness of fit test
        else: 
            categorical = True
            all_keys = sorted(list(p1.keys()))
            expected = [p1.get(k) for k in all_keys]
            observed = [p2.get(k) for k in all_keys]
            
            # scale observed to match expected
            observed_scaled = [i * sum(expected) / sum(observed) for i in observed]
            stat, p_val = stats.chisquare(f_obs=observed_scaled, f_exp=expected)
        
        if not save:
            print(f'{d1} vs. {d2} (stat, p-val): {stat:.3f}, {p_val:.3f}')
            
        # plotting
        current_ax = ax[row_ctr][col_ctr]
        x = np.arange(len(p1.keys()))
        width = 0.30
        
        p1_bar = current_ax.bar(x, list(p1.values()), width, label=d1)
        p2_bar = current_ax.bar(x + width, list(p2.values()), width, label=d2)
        
        current_ax.bar_label(p1_bar, padding=3)
        current_ax.bar_label(p2_bar, padding=3)
        current_ax.set_ylabel('Count')
        current_ax.set_title(f'{d1} Counts vs. {d2} Counts\nstatistic: {stat:.3f}, p-value: {p_val:.4f}')
        current_ax.set_xticks(x + width, p1.keys())
        current_ax.legend(loc='best')
        
        if categorical: 
            current_ax.set_xticklabels(p1.keys(), rotation=45, ha='right')
        
        if col_ctr + 1 < num_cols:
            col_ctr = col_ctr + 1
        else:
            col_ctr = 0
            row_ctr = row_ctr + 1
            
    plt.savefig('distr_comp.png', bbox_inches='tight')


if __name__ == '__main__': 
    main()