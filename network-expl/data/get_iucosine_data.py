import glob
import numpy as np
import os
import pandas as pd
from tqdm import tqdm

batch_folder = '/fs/ess/PAS2301/Data/Student_Data/xuandong/IU_Server_Files/Python_Project/listening_study_code/approved_batch_results_csv_COSINE'
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