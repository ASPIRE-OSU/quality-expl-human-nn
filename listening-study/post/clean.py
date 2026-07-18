# Study: Listening Study for Effects on Human MOS Ratings
#
# The Qualtrics survey for this study was built using template questions that then feed into
# each question using piped text fields. As a result, the responses that come out of Qualtrics
# use the piped text field name. This script converts those piped text field names back to their
# current value. 
#
# Additionally, this script removes unnecessary columns, splits out the incomplete (did not reach 
# practice questions) samples, removes the Prolific IDs, and replaces them with a participant number. 
#
# Note that if the piped text field value changes, this script needs to be updated and the Qualtrics
# survey will not show that change. 
#
# author: Ada Lamba
# version: 01/12/2026

import argparse
import pandas as pd
import random
import warnings
import os

# 1 - bad, 2 - poor, 3 - fair, 4 - good, 5 - excellent
MOS_PIPED_TEXT_FIELDS_COLS = {'${q://QID2/AnswerDescription/1}': 1,
                              '${q://QID2/AnswerDescription/2}': 2,
                              '${q://QID2/AnswerDescription/3}': 3, 
                              '${q://QID2/AnswerDescription/4}': 4, 
                              '${q://QID2/AnswerDescription/5}': 5
                             }
MOS_PIPED_TEXT_FIELDS_ROWS = {'[QID2-QuestionText] - [QID2-ChoiceDescription-1]': 'A',
                              '[QID2-QuestionText] - [QID2-ChoiceDescription-2]': 'B'
                             }

SIGNAL_PREFERENCE_PIPED_TEXT_COLS = {'${q://QID3/ChoiceDescription/1}': 'A',
                                '${q://QID3/ChoiceDescription/2}': 'B',
                                '${q://QID3/ChoiceDescription/3}': 'No preference',
                               }

SIGNAL_PREFERENCE_PIPED_TEXT_ROWS = {'[QID3-QuestionText]': 'preference'}

# noise: "Amount of noise (sounds heard in the recording that are not the speaker)"
# snr: "Loudness of speech compared to noise (the loudness of the speech as it relates to the loudness of noise)"
# distortion: "Audio distortion (e.g., clipping, static, reverberation)"
# type: "Type of noise (the type of noise, not its loudness. For example, traffic noise versus background music versus other speakers)"
FACTORS_PIPED_TEXT_FIELDS_COLS = {'${q://QID17/AnswerDescription/1}': 'Aided understanding',
                                  '${q://QID17/AnswerDescription/2}': 'Hindered understanding',
                                  '${q://QID17/AnswerDescription/3}': 'Did not affect understanding'
                                 }
FACTORS_PIPED_TEXT_FIELDS_ROWS = {'[QID17-QuestionText] - [QID17-ChoiceDescription-1]': 'noise',
                                  '[QID17-QuestionText] - [QID17-ChoiceDescription-2]': 'snr',
                                  '[QID17-QuestionText] - [QID17-ChoiceDescription-3]': 'distortion',
                                  '[QID17-QuestionText] - [QID17-ChoiceDescription-4]': 'type',
                                  }

ALL_DATA = '../data/clean.xlsx'

def main():
    args = parse_args()

    if args.out_file is None:
        args.out_file = args.in_file[:args.in_file.index('.xlsx')] + '_clean.xlsx'
    if args.incomplete_file is None:
        args.incomplete_file = args.in_file[:args.in_file.index('.xlsx')] + '_incomplete.xlsx'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_excel(args.in_file,keep_default_na=False)

    # remove unused columns
    clean_columns(df)

    # remove incomplete samples: defined as anyone who did not get to the practice questions
    incomplete_df = remove_incompletes(df)
    incomplete_df.to_excel(args.incomplete_file, index=False)

    # assign participant numbers and save map
    assign_participant_id(df, args.participant_map)
    df.sort_values('participant_id', axis=0, inplace=True)

    # convert piped text field names to values
    replaceTextFields(df)
    
    # add group column (useful later when compiling all data)
    groupname = args.in_file.split('/')[2]
    df.insert(0, 'group', groupname)

    # drop row 1 (import IDs) and save
    df.to_excel(args.out_file, index=False)

    # display stats: mean, standard deviation, minimum cutoff
    display_time_stats(df)

    return

def clean_columns(df):
    """ Removes the following columns from the dataframe: 'StartDate', 'EndDate', 'Status', 'IPAddress',
        'ResponseId', 'RecipientLastName', 'RecipientFirstName', 'RecipientEmail', 
        'ExternalReference', 'LocationLatitude', 'LocationLongitude', 'DistributionChannel', 'UserLanguage', 
        'Q_RecaptchaScore', 'Q_RelevantIDDuplicate', 'Q_RelevantIDDuplicateScore', 'Q_RelevantIDFraudScore', 
        'Q_RelevantIDLastStartDate', 'Q_.3.TEMPLATE_1', 'Q_.3.TEMPLATE_2', 'Q_.3.TEMPLATE_3', 'Q_.3.TEMPLATE_4'

    Args:
        df (pd.DataFrame): the data to remove columns from

    Updates:
        df: such that none of the listed columns are present
    """
    # update consent question to not be so long
    df.loc[0, 'Q2.consent'] = ''
    df.drop(columns=['StartDate', 'EndDate', 'Status','IPAddress', 'ResponseId', 
                     'RecipientLastName', 'RecipientFirstName', 'RecipientEmail', 'ExternalReference', 'LocationLatitude', 
                     'LocationLongitude', 'DistributionChannel', 'UserLanguage', 'Q_.2.QUALITYTEMPLATE_1', 
                     'Q_.2.QUALITYTEMPLATE_2', 'Q_.3.PREFTEMPLATE', 'Q_.4.FACTOR_TEMPLATE_1', 'Q_.4.FACTOR_TEMPLATE_2', 'Q_.4.FACTOR_TEMPLATE_3',
                     'Q_.4.FACTOR_TEMPLATE_4'],
        inplace=True)

def remove_incompletes(df):
    """ Removes the rows of df that correspond to an incomplete (did not get to the practice questions)
        samples and puts them in a second df, which is returned.

    Args:
        df (pd.DataFrame): the original data frame, possibly containing incomplete samples
    
    Updates:
        df: such that there are no values df['Q3.1'] == nan
    
    Returns:
        pd.DataFrame: a data frame containing only the incomplete samples from df. 
    """
    incomplete_df = pd.DataFrame(columns=df.columns)

    # get incomplete rows: df['Finished'] is false or df['Q185'] is nan
    # If a participant does not meet the screening criteria, their survey is still marked finished.
    # Q12.environment is the "do you commit to providing your best answers question" and occurs after the 
    # screening question. 
    # Do not consider df['Finished'] == False and df['Progress'] == 99 as incomplete - this happens
    # if the participant did not hit the sumbit button but did everything else. 
    incomplete_rows = df.loc[((df['Finished'] == 'False') & (df['Progress'] != 99)) | (df['Q12.environment'] == '')]

    # add rows to second df
    incomplete_df = pd.concat([incomplete_df, pd.DataFrame(incomplete_rows)], ignore_index=True)

    # remove rows from original df
    df.drop(incomplete_rows.index, inplace=True)

    # to match the complete samples, assign a participant number (negative). 
    # since Prolific IDs are not recorded, we'll just fill the na's with decrementing numbers.

    return incomplete_df

def assign_participant_id(df, filepath=None):
    """Given a dataframe, map Prolific IDs to (random) participant IDs, save map to an external file, and
    remove Prolific IDs from the dataframe.

    Args:
        df (pd.DataFrame): the dataframe containing the original data.
        filepath (str): the filepath to the existing participant IDs map file, if it exists. 

    Updates:
        df: Removed column 'Q3.1' which corresponds to Prolific ID, added column 'participant_id'.
    """
    # assign participant number randomly
    df_participants = pd.DataFrame(columns=['number', 'prolific_id'])

    # check if particpant map exists, and if so, read those numbers
    #df['Q3.prolific_id'].fillna(-1, inplace=True)
    df.fillna({'Q3.prolific_id ': -1}, inplace=True)
    prolific_id = list(df['Q3.prolific_id '].values)[1:]
    participant_nos = []
    if filepath is not None:
        df_participants = pd.read_excel(filepath)
        participant_nos = df_participants['number'].values.tolist()

        # check if participant id map exists, if so, remove from assignment list
        prolific_id = [i for i in prolific_id if str(i) not in df_participants['prolific_id'].values.tolist()]
        
    # assign ids (shuffle so we assign randomly)
    num = max(participant_nos)+1 if len(participant_nos) > 0 else 1
    random.shuffle(prolific_id)
    
    for i in prolific_id:
        # assign participant no.
        df_participants.loc[len(df_participants)] = [num, i]
        num += 1

    # save id map
    if filepath is None:
        filepath = '../data/participant_map.xlsx'
    df_participants.to_excel(filepath, index=False)

    # update df with participant ids
    df['participant_id'] = -1
    df.insert(0, 'participant_id', df.pop('participant_id'))
    for i, row in df.iloc[1:].iterrows():
        prolific_id = str(row['Q3.prolific_id '])
        participant_id = df_participants.loc[df_participants['prolific_id'] == prolific_id, 'number'].values[0]
        df.at[i, 'participant_id'] = participant_id

    # remove prolific ids
    df.drop(columns=['Q3.prolific_id '], inplace=True)

    return


def display_time_stats(df):
    """Given a dataframe, displays the mean, standard deviation, and minimum cutoff time of the elapsed time column.
    Minimum cutoff time is defined as MEAN - 2*(STDEV). 
    """
    # calculate mean, stdev, min cutoff
    time_col = 'Duration (in seconds)'

    # only consider those who made it to the practice questions. These rows are determined by having a positive 
    # participant id and having a batch number assigned.
    mean_s = pd.to_numeric(df.loc[(df['participant_id']>-1) & (len(df['batch'])>0), time_col]).mean()
    stdev_s = pd.to_numeric(df.loc[(df['participant_id']>-1) & (len(df['batch'])>0), time_col]).std()
    mincutoff_s = mean_s - (2*stdev_s)

    # convert seconds to minutes and print
    print('Group statistics')
    print('Mean (min): {:.2f}'.format(mean_s/60.0))
    print('Std. Dev. (min): {:.2f}'.format(stdev_s/60.0))
    print('Minimum cutoff time (min): {:.2f}'.format(mincutoff_s/60.0))

    # calculate total stats as well
    if (os.path.isfile(ALL_DATA)):
        all_df = pd.read_excel(ALL_DATA)
        mean_all_s = pd.to_numeric(all_df.loc[(all_df['participant_id']>-1) & (len(all_df['batch'])>0), time_col]).mean()
        stdev_all_s = pd.to_numeric(all_df.loc[(all_df['participant_id']>-1) & (len(all_df['batch'])>0), time_col]).std()
        mincutoff_all_s = mean_all_s - (2*stdev_all_s)

        # convert seconds to minutes and print
        print('\nAggregate statistics')
        print('Mean (min): {:.2f}'.format(mean_all_s/60.0))
        print('Std. Dev. (min): {:.2f}'.format(stdev_all_s/60.0))
        print('Minimum cutoff time (min): {:.2f}'.format(mincutoff_all_s/60.0)) 

    return


def replaceTextFields(df: pd.DataFrame):
    """Given a dataframe, updates any known piped text field names with their values

    Args:
        df (pd.DataFrame): the dataframe containing the original data with piped text field names. 
        
    Updates:
        df: Any piped text field names are replaced with their values.
    """
    # rename columns in first  from *AnswerDescription* to Aided/hindered/did not affect understanding
    for col_name in df.columns:
        # only affect the columns that have piped text
        if 'QID2' in str(df.loc[0, col_name]): 
            # replace choice text
            for row in MOS_PIPED_TEXT_FIELDS_ROWS.keys():
                df.loc[0, col_name] = df.loc[0, col_name].replace(row, MOS_PIPED_TEXT_FIELDS_ROWS[row])

        if 'QID3' in str(df.loc[0, col_name]): 
            # replace choice text
            for row in SIGNAL_PREFERENCE_PIPED_TEXT_ROWS.keys():
                df.loc[0, col_name] = df.loc[0, col_name].replace(row, SIGNAL_PREFERENCE_PIPED_TEXT_ROWS[row])

        if 'QID17' in str(df.loc[0, col_name]):
            # replace choice text
            for row in FACTORS_PIPED_TEXT_FIELDS_ROWS.keys():
                df.loc[0, col_name] = df.loc[0, col_name].replace(row, FACTORS_PIPED_TEXT_FIELDS_ROWS[row])
    
    # rename row values from *ChoiceDescription* to 
    df.replace(FACTORS_PIPED_TEXT_FIELDS_COLS, inplace=True)
    df.replace(MOS_PIPED_TEXT_FIELDS_COLS, inplace=True)
    df.replace(SIGNAL_PREFERENCE_PIPED_TEXT_COLS, inplace=True)

    return
    

def parse_args():
    """
    Reads and stores command line arguments.
    
    Returns:
       args parser object
    """
    parser = argparse.ArgumentParser(prog='Piped Text Field Replacer')
    parser.add_argument('--in_file', '-i', required=True, type=str, help='The file path to the original data file.')
    parser.add_argument('--out_file', '-o', required=False, type=str, help='The file path to save the modified file to.')
    parser.add_argument('--incomplete_file', '-inc', required=False, default=None, type=str, help='The file path to save the incomplete responses to.')
    parser.add_argument('--participant_map', '-p', required=False, default=None, type=str, help='The file path of the existing participant id/prolific id map.')
    
    return parser.parse_args()
    

if __name__ == '__main__':
    main()
    
    