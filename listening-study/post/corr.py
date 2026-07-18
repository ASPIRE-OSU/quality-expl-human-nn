import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

FREQBINS = [0, 1, 2, 3, 4, 5, 6, 12, 14, 15, 17, 20, 21, 26, 31, 40, 41]
HUMAN_DATA = '../results/mos_ratings.csv'
MACHINE_DATA = '../../omnixai/data/iucosine.csv'
SCALES = [0.5, 10, 1000]

def main():
    output_str = ''
    output_df = pd.DataFrame(columns=['model', 'freqbin', 'pcc', 'p-value'])
    
    # get quality listening study data
    print('Preparing...')
    human_df = pd.read_csv(HUMAN_DATA)
    human_df = human_df[['filename', 'average_mos']]
    
    # get MOSNet, DNSMOS, ScoreQ perturbation data
    nn_df = pd.read_csv(MACHINE_DATA)
    perturbed_files = nn_df[nn_df['filename'].str.contains('freqbin')]['filename'].tolist()
    orig_files = list(set([f"{i[: i.index('_freqbin')]}.wav" for i in perturbed_files]))
    files_of_interest =  perturbed_files + orig_files
    nn_df = nn_df[nn_df['filename'].isin(files_of_interest)]
    nn_df = nn_df[['filename', 'mosnet_pred', 'dnsmos_pred', 'scoreq_pred']]
    
    # track the scale and frequency bin for each file as its own column
    split_freqbin_scale_columns(human_df)
    split_freqbin_scale_columns(nn_df)       
    
    # organize data by frequency bin
    nn_df.to_csv('nn_df.csv', index=False)
    nn_df = nn_df[['freqbin', 'scale', 'mosnet_pred', 'dnsmos_pred', 'scoreq_pred']]
    human_df = human_df.groupby(by=['freqbin', 'scale'], as_index=False, dropna=False)['average_mos'].mean()
    nn_df = nn_df.groupby(by=['freqbin', 'scale'], as_index=False, dropna=False).mean()

    # for each frequency bin, calculate correlation between each model's predictions
    # and human ratings
    print('Calculating...\n')
    for fb in FREQBINS:
        output_str += f'FREQUENCY BIN: {fb}\n' 
        for model in ['mosnet', 'dnsmos', 'scoreq']: 
            output_str += f'    MODEL: {model}\n'
            model_vals = []
            human_vals = []
            
            # add original
            model_vals.append(nn_df[(nn_df['freqbin'].isna()) & (nn_df['scale'] == 1)][f'{model}_pred'].values[0])
            human_vals.append(human_df[(human_df['freqbin'].isna()) & (human_df['scale'] == 1)]['average_mos'].values[0])
            
            for s in SCALES:
                model_vals.append(nn_df[(nn_df['freqbin'] == fb) & (nn_df['scale'] == s)][f'{model}_pred'].values[0])
                human_vals.append(human_df[(human_df['freqbin'] == fb) & (human_df['scale'] == s)]['average_mos'].values[0])
        
            corr, p = pearsonr(human_vals, model_vals)
            output_str += f'    PCC: {corr:.4f}, p-value: {p:.4f}\n\n'
            row = pd.DataFrame.from_dict({'model': [model],
                                          'freqbin': [fb],
                                          'pcc': [corr],
                                          'p-value': [p]})
            output_df = pd.concat([output_df, row], ignore_index=False) if len(output_df) > 0 else row
            
    
    # write results to file
    output_file = 'corr_output.txt'
    with open(output_file, 'w') as f:
        f.write(output_str)
    print(f'Saved to: {output_file}')
    
    output_df.to_csv(output_file.split('.txt')[0]+'.csv', index=False)
    

def split_freqbin_scale_columns(df):
    df['freqbin'] = None
    df['scale'] = None
    
    new_rows = []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        filename = row['filename']
        
        if 'scaled' in filename:
            freqbin = int(filename[filename.index('_freqbin') + len('_freqbin'): filename.index('_scaled')])
            scale = float(filename[filename.index('_scaled') + len('_scaled'): filename.index('.wav')])
            df.at[i, 'freqbin'] = freqbin
            df.at[i, 'scale'] = scale
        else:
            # an original, unperturbed file
            df.at[i, 'scale'] = 1
            
            # need to duplicate the row for each other frequency bin
            for fb in FREQBINS[1:]:
                new_row = row.to_dict()
                new_row['freqbin'] = fb
                new_row['scale'] = 1
                new_rows.append(new_row)
            new_rows_df = pd.DataFrame(new_rows)

    df = pd.concat([df, new_rows_df], ignore_index=True)
    return df

if __name__ == '__main__':
    main()