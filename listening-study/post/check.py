# Study:Listening Study for Effects on Human MOS Ratings
#
# This script checks and reports two things:
# 1. Each batch contains one pair containing the same audio twice. It is expected that the participant
# has no preference between these signals as they are the same. This program identifies any samples that 
# do not meet the above criteria, as they may warrant further investigation. 
#
# 2. Each question has three parts: the rating, the preference, and the table. Each audio must be played
# at least once.

# author: Ada Lamba
# version: 01/12/2026


import argparse
import pandas as pd

MAPPER_FILE = '../data/map_to_original_name.csv'

def main():
    args = parse_args()
    df = pd.read_excel(args.file, keep_default_na=False)
    map_df = pd.read_csv(MAPPER_FILE)
    check_refs(df, map_df)
    print('')
    check_plays(df)
    # print('')
    #check_nas(df)
    return

def check_plays(df):
    """For each participant, checks that the q[question #]-audio1_played and 
    q[question #]-audio2_played values are at least 1.

    Args:
        - df (pd.DataFrame): the input data
    """
    # get the column names for the played data
    play_data_cols = ['p1-audio1_played', 'p1-audio2_played', 'p2-audio1_played', 'p2-audio2_played']
    for i in range(1, 17):
        play_data_cols.append('q{}-audio1_played'.format(i))
        play_data_cols.append('q{}-audio2_played'.format(i))

    fails = []

    # iterate over all participatns
    for _, row in df[2:].iterrows():
        part_fails = []
        # iterate over all played data columns
        for col in play_data_cols:
            play_num = int(row[col])
            if play_num < 1:
                part_fails.append(col)
        
        # if a participant failed any checks, add them to the list
        if len(part_fails) > 0:
            fails.append((int(row['participant_id']), part_fails))

    # print results
    if len(fails) == 0:
        print('All participants passed audio play checks.')
    else:
        print('Participants that did not pass audio play checks: {}'.format(fails))


def check_refs(df, map_df):
    """For each sample, checks that no preference was indicated.
       If the criteria is not met, the participant number is output.

       Args:
         - df (pd.DataFrame): the input data
         - map_df (pd.DataFrame): the batch signals data
    """
    fails = []
    refs = []

    # generate list of files in reference pairs
    for i in range(0, len(map_df), 2):
        row_a = map_df.iloc[i]
        row_b = map_df.iloc[i + 1]
        if row_a['old_filename'] == row_b['old_filename']:
            batch_num = row_a['batch']
            q = int(row_a['new_filename'][row_a['new_filename'].index('pair')+4: row_a['new_filename'].index('-')]) + 1
            refs.append((batch_num, q))

    # iterate over all samples
    for _, row in df[1:].iterrows():
        # get reference signal question numbers
        batch = int(row['batch'])
        ref_questions = [i[1] for i in refs if i[0] == batch]

        # check that preference for reference question is none
        # preferences questions are always Q*.2-preference
        for i in ref_questions:
            ref_pref_col = 'Q' + i + '.2-preference'
            pref = row[ref_pref_col]
            if pref != 'No preference':
                fails.append(str(row['participant_id']) + ' (question {})'.format(ref_trans_col))
                continue
    
    # print results
    if len(fails) == 0:
        print('All participants passed reference checks.')
    else:
        print('Participants that did not pass reference checks: {}'.format(fails))


def parse_args():
    """
    Reads and stores command line arguments.
    
    Returns:
       args parser object
    """
    parser = argparse.ArgumentParser(prog='Reference signal checker')
    parser.add_argument('--file', required=True, type=str, help='The input file path.')
     
    return parser.parse_args()


if __name__ == '__main__':
    main()