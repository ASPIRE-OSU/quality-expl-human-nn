import os
import pandas as pd
import pickle
from tqdm import tqdm


DATASET_STATS = 'data/mosnet_distr_test.pkl'
DATA = 'iucosine.csv'


def main():
    test_df = pd.read_csv(DATA)
    test_df = test_df[test_df['set'] == 'test']
    test_files = test_df['filename'].tolist()
    
    # get info and save
    tmos_dict_all, pmos_dict_all = get_distr(test_df)
    pickle.dump([tmos_dict_all, pmos_dict_all], open(DATASET_STATS, 'wb'))
        

def get_distr(df, model='mosnet'):
    """Given a list of files, returns distributions for the truncated true MOS and predicted MOS.
    
    Args:
        files (list): a list of files

    Returns:
        tuple (dict, dict): a tuple with a dict for each distribution:
            - {TMOS_VAL: int} (true mos)
            - {PMOS_VAL: int} (predicted mos)
    """
    tmos_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    pmos_dict = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        tmos = int(row['scaled_mos'])
        pmos = int(row[model+'_pred'])
        tmos_dict[tmos] = tmos_dict.get(tmos, 0) + 1
        pmos_dict[pmos] = pmos_dict.get(pmos, 0) + 1
        
    return tmos_dict, pmos_dict


if __name__ == '__main__':
    main()