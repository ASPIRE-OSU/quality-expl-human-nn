import os
import scoreq
import pandas as pd
from tqdm import tqdm
import glob

IUCOSINE = '../data/iucosine.csv'
PERTURB_DIR = '../data/perturbed_iucosine'
DATA_DIR = '../../data/MOS_Quality_INTERSPEECH_2020/16k_speech'

def main():
    # load model
    nr_scoreq = scoreq.Scoreq(data_domain='natural', mode='nr')

    # load data
    df = pd.read_csv(IUCOSINE)

    # batch predict
    input_name = nr_scoreq.session.get_inputs()[0].name

    for i, row in tqdm(df.iterrows(), total=len(df)):
        f = row['filename']
        if 'scaled' in f:
            score = nr_scoreq.predict(os.path.join(PERTURB_DIR, f))
        else:
            score = nr_scoreq.predict(os.path.join(DATA_DIR, f))
        df.at[i, 'scoreq_pred'] = score

    df.to_csv(IUCOSINE, index=False)
    

if __name__ == '__main__':
    main()
        