## This script compiles all of the clean data.
##
## @author: Ada Lamba
## @version: 01/12/2026

import os
import pandas as pd
from tqdm import tqdm

# ignore performance warnings from using pd.insert
from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

def main():
    # get group data
    group_paths = []
    print('Collecting group data...')
    for dirname in tqdm(os.listdir(os.getcwd())):
        if os.path.isdir(dirname):
            if 'all' in dirname:
                continue
            group_paths.append(os.path.join(dirname, "clean.xlsx"))

    all_df = pd.read_excel(group_paths[0])

    # make sure 'batch' is second column
    batch = all_df.pop('batch')
    all_df.insert(1, 'batch', batch)

    # append groups
    print('Joining data...')
    for d in tqdm(group_paths[1:]):
        group_df = pd.read_excel(d)

        # drop subtitle row
        group_df = group_df[group_df.participant_id != -1]

        # add group number column
        if 'group' not in group_df.columns.values:
            group_df.insert(0, 'group', d.split('/')[0])

        all_df = pd.concat([all_df, group_df], ignore_index=True, sort=False)
    
    # save
    all_df = all_df.sort_values('participant_id')
    all_df.reset_index(drop=True, inplace=True)
    all_df.to_excel("clean.xlsx", index=False)


if __name__ == '__main__':
    main()