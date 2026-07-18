import pandas as pd
from tqdm import tqdm
import json
from collections import defaultdict
import scipy
from scipy.stats import ttest_ind
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import string
import ast
import seaborn as sns
import sys

INDICES = [0, 1, 2, 3, 4, 5, 6, 12, 14, 15, 17, 20, 21, 26, 31, 40, 41]
plt.style.use('tableau-colorblind10')
COLORS = ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
MARKERS = ['o', 's', 'P', '^', 'd', 'X']
SCALES = [0.5, 1, 10.0, 1000.0]
DISPLAY_SCALES = [0.5, 1, 10,  1000]
plt.rcParams.update({'font.size': 20})
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm'

df = pd.read_excel("../data/all/clean.xlsx")
map_df = pd.read_csv("../data/map_to_original_name.csv")

# check that there are no spaces at the end of column names
df.columns = df.columns.str.strip()

# fix mismatching column names
# yes, it would be easier to fix on Qualtrics, but here we are
df.columns = df.columns.str.replace(r"factor_", "factors_", regex=True)
df.columns = df.columns.str.replace(r"factors ", "factors", regex=True)
df.columns = df.columns.str.replace(r"Q14.2-factors_", "Q14.3-factors_", regex=True)

# maps english to code
FACTOR_CHOICES_LONG = {'Hindered understanding': -1, 'Did not affect understanding': 0, 'Aided understanding': 1}
FACTOR_CHOICES_SHORT = {'Hurt': -1, 'Neutral': 0, 'Helped': 1}
FACTORS = {'noise': 'Noise',
           'snr': 'SNR',
           'distortion': 'Distortion',
           'type': 'Noise Type'}

# practice audios
practice_files = map_df[(map_df['batch'] == 'practice') & (map_df['group'] == 'group1')]    # doesn't matter which group is used here since they're the same
p1_a1_file = practice_files[practice_files['new_filename'] == 'practice0-audio1.wav']['old_filename'].values[0]
p1_a2_file = practice_files[practice_files['new_filename'] == 'practice0-audio2.wav']['old_filename'].values[0]
p2_a1_file = practice_files[practice_files['new_filename'] == 'practice1-audio1.wav']['old_filename'].values[0]
p2_a2_file = practice_files[practice_files['new_filename'] == 'practice1-audio2.wav']['old_filename'].values[0]

# manual map
PART_MAP = {184: 'group3',
            182: 'group2', 175: 'group2', 186: 'group2', 176: 'group2', 178: 'group2', 185: 'group2', 179: 'group2', 183: 'group2', 
                177: 'group2', 172: 'group2', 174: 'group2', 171: 'group2', 
            180: 'group1', 173: 'group1', 181: 'group1', 196: 'group1', 190: 'group1', 200: 'group1', 191: 'group1', 193: 'group1', 
                187: 'group1', 199: 'group1', 189: 'group1', 195: 'group1', 198: 'group1', 192: 'group1', 194: 'group1', 
                197: 'group1', 188: 'group1'
            }

def main():
    ratings_df, pref_df, factors_df = organize_data()
    compare_distr(ratings_df)
    plot_perturb(ratings_df)
    plot_bars(factors_df)
    plot_heatmap(factors_df)

def organize_data(save=True):
    # format
    # filename: [mos_rating1, mos_rating2, ...]
    ratings_df = pd.DataFrame(columns=['filename', 'ratings'])

    # format
    # (filename1, filename2): (# preferred filename1 over filename2, # preferred filename2 over filename1, # equal)
    pref_df = pd.DataFrame(columns=['file_pair', 'num_pref_A', 'num_pref_B', 'num_pref_equal'])

    # format
    # filename: [(noise:-1/0/1, snr:-1/0/1, distortion:-1/0/1, type:-1/0/1), ...]
    # where -1 indicates that the factor hurt the quality, 0 is neutral, and 1 is helped
    # each tuple corresponds to one participant
    factors_df = pd.DataFrame(columns=['filename', 'noise', 'snr', 'distortion', 'type'])

    for _, row in tqdm(df[1:].iterrows(), total=len(df[1:])):
        batch = int(row['batch'])
        group = 'group1' if 'prelim' in row['group'] else row['group'].split('_')[0]

        # handle group4 which are individually mapped
        pid = int(row['participant_id'])
        if pid in PART_MAP.keys():
            group = PART_MAP[pid]

        # PRACTICE FILES    
        for i, p in enumerate([(p1_a1_file, p1_a2_file), (p2_a1_file, p1_a2_file)]):
            # ratings
            ratings_df = get_ratings(row, ratings_df, f'QP{i+1}.1-quality_1', p[0])
            ratings_df = get_ratings(row, ratings_df, f'QP{i+1}.1-quality_2', p[1])

            # preference
            pref_df, pref_file = get_preference(row, pref_df, f'QP{i+1}.2-preference', p[0], p[1])
            
            # factors
            factors_df = get_factors(row, factors_df, f'QP{i+1}.3-factors', pref_file)

        
        # MAIN QUESTIONS
        for i in range(16):
            # file names for this pair
            p = (map_df[(map_df['group'] == group) & (map_df['new_filename'] == f'batch{batch}_pair{i}-audio1.wav')]['old_filename'].values[0],
                 map_df[(map_df['group'] == group) & (map_df['new_filename'] == f'batch{batch}_pair{i}-audio2.wav')]['old_filename'].values[0])
            
            # ratings
            ratings_df = get_ratings(row, ratings_df, f'Q{i+1}.1-quality_1', p[0])
            ratings_df = get_ratings(row, ratings_df, f'Q{i+1}.1-quality_2', p[1])

            # preference
            pref_df, pref_file = get_preference(row, pref_df, f'Q{i+1}.2-preference', p[0], p[1])

            # factors
            factors_df = get_factors(row, factors_df, f'Q{i+1}.3-factors', pref_file)

    # get average MOS rating and true MOS rating
    # average MOS
    ratings_df['average_mos'] = ratings_df['ratings'].apply(lambda x: sum(x)/float(len(x)))
    factors_df = factors_df.merge(ratings_df[['filename', 'average_mos']], on='filename', how='left')

    # get true MOS
    src_df = pd.read_csv('../data/src-files.csv')
    ratings_df = ratings_df.merge(src_df[['filename', 'scaled_mos']], on='filename', how='left')
    ratings_df = ratings_df.rename(columns={'scaled_mos': 'true_mos'})

    # true MOS for practice files
    iucosine_df = pd.read_csv('../../omnixai/data/iucosine.csv')
    p1_tmos = float(iucosine_df[iucosine_df['filename'] == p1_a2_file]['mos5'].values[0])
    p2_tmos = float(iucosine_df[iucosine_df['filename'] == p2_a1_file]['mos5'].values[0])
    ratings_df.loc[ratings_df['filename'] == p1_a2_file, 'true_mos'] = p1_tmos
    ratings_df.loc[ratings_df['filename'] == p2_a1_file, 'true_mos'] = p2_tmos
    
    # get number of ratings
    ratings_df['num_ratings'] = ratings_df['ratings'].apply(len)
    
    # get batch/group number
    map_df['batch'] = map_df['batch'].apply(lambda x: int(x) if x != 'practice' else x)
    map_agg = (
        map_df[['old_filename', 'batch', 'group']].groupby('old_filename')
        .agg({
            'group': lambda x: list(x.unique()),
            'batch': lambda x: list(x.unique())
        }).reset_index()
    )
    ratings_df = ratings_df.merge(map_agg, how='left', left_on='filename', right_on='old_filename').drop(columns='old_filename')
    ratings_df['group'] = ratings_df['group'].apply(lambda x: x if isinstance(x, list) else [])
    ratings_df['batch'] = ratings_df['batch'].apply(lambda x: x if isinstance(x, list) else [])
    ratings_df = ratings_df[['group', 'batch', 'filename', 'average_mos', 'true_mos', 'num_ratings', 'ratings']]

    # save
    for df_obj in [(ratings_df, '../results/mos_ratings.csv'), 
                (pref_df, '../results/preferences.csv'),
                (factors_df, '../results/factors.csv')]:
        df_obj[0].to_csv(df_obj[1], index=False)
    
    return ratings_df, pref_df, factors_df

def get_ratings(row, df, question, file):
    rating = int(row[question])
    mask = df['filename'] == file
    if mask.any():
        df.loc[mask, 'ratings'] = df.loc[mask, 'ratings'].apply(
            lambda x: x + [rating]
        )
    else:
        new_row = pd.DataFrame({'filename': [file], 'ratings': [[rating]]})
        df = pd.concat([df, new_row], ignore_index=True)
    
    return df

def get_preference(row, df, question, fileA, fileB):
    preference = row[question]
    mask = df['file_pair'] == f'{fileA} / {fileB}'
    
    if preference == 'B': pref_file = fileB
    else: pref_file = fileA

    if mask.any():
        if preference == 'A': df.loc[mask, 'num_pref_A'] += 1
        elif preference == 'B': df.loc[mask, 'num_pref_B'] += 1
        else: df.loc[mask, 'num_pref_equal'] += 1
    else:
        new_row = pd.DataFrame({'file_pair': [f'{fileA} / {fileB}'], 
                           'num_pref_A': [int(preference == 'A')], 
                           'num_pref_B': [int(preference == 'B')], 
                           'num_pref_equal': [int(preference == 'No preference')]})
        df = pd.concat([df, new_row], ignore_index=True)

    return df, pref_file

def get_factors(row, df, question, pref_file):
    noise = FACTOR_CHOICES_LONG[row[f'{question}_1']]
    snr = FACTOR_CHOICES_LONG[row[f'{question}_2']]
    distortion = FACTOR_CHOICES_LONG[row[f'{question}_3']]
    ntype = FACTOR_CHOICES_LONG[row[f'{question}_4']]

    mask = df['filename'] == pref_file
    if mask.any():
        df.loc[mask, 'noise'] = df.loc[mask, 'noise'].apply(lambda x: x + [noise])
        df.loc[mask, 'snr'] = df.loc[mask, 'snr'].apply(lambda x: x + [snr])
        df.loc[mask, 'distortion'] = df.loc[mask, 'distortion'].apply(lambda x: x + [distortion])
        df.loc[mask, 'type'] = df.loc[mask, 'type'].apply(lambda x: x + [ntype])
    else:
        new_row = pd.DataFrame({'filename': [pref_file], 
                                'noise': [[noise]], 
                                'snr': [[snr]],
                                'distortion': [[distortion]], 
                                'type': [[ntype]]})
        df = pd.concat([df, new_row], ignore_index=True)
    
    return df

def compare_distr(ratings_df):
    # t-test to see if average MOS for source files differ between true and collected MOS
    # t-test of two independent samples
    mos_data = ratings_df[~ratings_df['filename'].str.contains('scaled')]
    collected = mos_data['average_mos'].tolist()
    true = mos_data['true_mos'].tolist()
    ttest = ttest_ind(true, collected)
    annot_str = "T-test results between IUCOSINE-collected\nand this study's MOS\n"
    annot_str += f"(including practice): {ttest.statistic:.2f}, p-value: {ttest.pvalue:.2f}\n"

    mos_data_no_practice = mos_data[(mos_data['filename'] != p1_a2_file) & (mos_data['filename'] != p2_a1_file)]
    collected_no_practice = mos_data_no_practice['average_mos'].tolist()
    true_no_practice = mos_data_no_practice['true_mos'].tolist()
    ttest_no_practice = ttest_ind(true_no_practice, collected_no_practice)
    annot_str += f"(not including practice): {ttest_no_practice.statistic:.2f}, p-value: {ttest_no_practice.pvalue:.2f}\n"

    # plot average MOS
    
    # just source signals
    files = list(string.ascii_uppercase[:len(mos_data['filename'].tolist())])
    filemapping_str = ""
    for f in range(len(files)):
        filemapping_str += f"{files[f]}: {mos_data['filename'].tolist()[f]}\n"
        
    #print(f'\nFile mapping: {filemapping}\n')
    fig, ax = plt.subplots(figsize=(12,9), dpi=600)
    ax.plot(files, 
            true, 
            linestyle='None',
            marker='o', 
            markersize=6, 
            label=f'IUCOSINE Original Collection', 
            color=mcolors.to_rgb('darkorange'))

    ax.plot(files,
            collected, 
            linestyle='None',
            marker='s',
            markersize=6,
            label=f'Our collection', 
            color=mcolors.to_rgb('dodgerblue'))
    
    # annotate with t-test results and file mapping
    annot_str = "File Mapping\n" + annot_str + '\n' + filemapping_str
    #ax.text(1.07, 0, annot_str, fontsize=8, transform=ax.transAxes, va='bottom')

    # dumbbell plot
    for i in range(len(true)):
        ax.plot([i,i], [true[i], collected[i]], color='black', linestyle='--', linewidth=3, alpha=0.5)

    ax.set_xlabel('File')
    ax.set_ylabel('Mean Opinion Score (MOS)')
    ax.set_title(f'Comparison of MOS scores collected from\noriginal IUCOSINE and our studies')
    ax.set_ylim(0.5, 5.5)
    ax.axhspan(1, 5, facecolor='gainsboro', alpha=0.5, label='Valid MOS Range')
    ax.grid(alpha=0.5)
    ax.legend(loc='upper right', fontsize=14) 
    plt.savefig(f'../results/iucosine_v_ours_mos.png', bbox_inches='tight')
    plt.clf()

def plot_perturb(ratings_df):
    # plot difference after perturbation for each file
    colors = COLORS[:9]
    models = ['DNSMOS', 'MOSNet', 'ScoreQ']
    frequency_bands = [(3, 4, 5, 14, 15, 26, 31, 40, 41), 
                       (3, 4, 5, 6, 12, 17, 31, 40, 41),
                       (2, 3, 4, 5, 20, 20, 31, 40, 41)]

    for m, fb in zip(models, frequency_bands):
        fig, ax = plt.subplots(figsize=(12, 7), dpi=600)
        for freq, c in zip(fb, colors):
            scale_df = ratings_df[ratings_df['filename'].str.contains(f'freqbin{freq}_')]
            scale_files = scale_df['filename'].tolist()
            orig_files = list(set([f[:f.index('_freqbin')] + '.wav' for f in scale_files]))
            for f in orig_files:
                scale_df = pd.concat([scale_df, ratings_df[ratings_df['filename'] == f]])

            mos = {i: [] for i in SCALES}

            for i, row in scale_df.iterrows():
                f = row['filename']
                if 'scaled' in f:
                    scale = float(f[f.index('scaled')+6:f.index('.wav')])
                else:
                    scale = 1
            
                mos[scale] = mos[scale] + [row['average_mos']]
                
            avgs = [np.mean(mos[s]) for s in SCALES]
            
            ax.plot(SCALES, 
                    avgs, 
                    linestyle='-', 
                    linewidth=2, 
                    marker='o', 
                    markersize=4, 
                    label=f'Freq. Band {freq}', 
                    alpha=0.8,
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
        ax.legend(loc='upper right', fontsize=14) 
        plt.savefig(f'../results/{m}_scale_pred_listening_study.png', bbox_inches='tight')
        plt.clf()

def plot_bars(factors_df):
    response_types = FACTOR_CHOICES_SHORT.values()
    # Count responses per factor
    counts = {factor: {r: 0 for r in response_types} for factor in FACTORS.values()}

    for factor, Factor in FACTORS.items():
        for responses in factors_df[factor]:
            for r in responses:
                if r in response_types:
                    counts[Factor][r] += 1

    # Create dataframe for plotting
    count_df = pd.DataFrame(counts).T
    count_df["total"] = count_df.sum(axis=1)
    percentage_df = count_df.div(count_df["total"], axis=0) * 100

    # Plot stacked bar chart
    fig, ax = plt.subplots(figsize=(12, 9), dpi=600)

    bottom = np.zeros(len(count_df))
    #colors = {-1: mcolors.to_rgb('tab:red'), 0: mcolors.to_rgb('tab:blue'), 1: mcolors.to_rgb('tab:green')}
    colors = {-1: COLORS[5], 0: COLORS[2], 1: COLORS[0]}
    hatches = {-1: '/', 0: None, 1: '\\'}

    for R, r in FACTOR_CHOICES_SHORT.items():
        bars = ax.bar(
            count_df.index,
            count_df[r],
            bottom=bottom,
            label=f"{R}",
            color=colors[r],
            hatch=hatches[r]
        )

        # Annotate percentages
        for i, bar in enumerate(bars):
            pct = percentage_df.iloc[i][r]
            if pct > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom[i] + bar.get_height() / 2,
                    f"{pct:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=16,
                    color="white",
                    fontweight='bold'
                )

        bottom += count_df[r].values

    # Formatting
    ax.set_ylim(0, 4000)
    ax.set_ylabel("Count")
    plt.xlabel("Factors")
    ax.set_title("Perceived Effect Frequency by Factor")
    ax.legend(loc='upper center', ncols=3)
    plt.tight_layout()
    plt.savefig(f"../results/factor_count_stackedbar.png", bbox_inches='tight')
    plt.clf()

def plot_heatmap(factors_df):
    # Convert to long format
    rows = []
    for _, row in factors_df.iterrows():
        for factor in FACTORS:
            for response in row[factor]:
                rows.append({
                    "factor": factor,
                    "response": response,
                    "average_mos": row["average_mos"]
                })

    long_df = pd.DataFrame(rows)

    # Map response values to labels
    response_map = {1: "Helped", 0: "Neutral", -1: "Hurt"}
    long_df["response"] = long_df["response"].map(response_map)

    # Aggregate MOS
    heatmap_df = (
        long_df
        .groupby(["factor", "response"])["average_mos"]
        .mean()
        .reset_index()
        .pivot(index="factor", columns="response", values="average_mos")
    )


    # rearrange heatmap so we get the order; hurt, neutral, helped
    heatmap_df = heatmap_df[['Hurt', 'Neutral', 'Helped']]

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(12, 9), dpi=600)
    sns.heatmap(
        heatmap_df,
        annot=True,
        cmap="flare_r", # more grayscale and colorblind-friendly than Spectral
        fmt=".2f",
        cbar_kws={"label": "Average MOS"}
    )

    plt.title("Average MOS by Factor and Response")
    plt.ylabel("Factor")
    plt.xlabel("Response")
    plt.tight_layout()
    plt.savefig(f"../results/factor_mos_heatmap.png", bbox_inches='tight')

if __name__ == '__main__':
    main()