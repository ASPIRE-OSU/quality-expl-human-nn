from ast import literal_eval
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

DATA = '../results/mos_ratings.csv'
FILE_MAP = '../data/map_to_original_name.csv'
OUTLIER_THRESHOLD = 2.5
RESPONSES = '../data/all/clean.xlsx'

def main():
    """
    Given the raw data from the IUCOSINE quality listening study dataset, prepares and cleans the dataset for use. 
    This includes removing outliers and normalizing participant responses.
    """
    # read/prepare dataset
    df = prep_data()
    
    # calculate z-scores and remove any identified outliers
    zscores(df)
    
    # use DBSCAN and isolation forests (IF) to remove any participants who consistently disagree with others
    dbscan_if(df)
    
    # normalize ratings by participant
    df = normalize(df)
    
    # calculate update mos from normalized data
    get_mos(df)
    
    # save processed data
    df['raters'] = df['raters'].apply(lambda x: list(set(x)))
    filename_out = '../results/normalized_mos_ratings.csv'
    df = df[['filename', 'average_mos', 'mos_normalized', 'raters', 'ratings_normalized', 'ratings', 'zscores', 'dbscan', 'if', 'dbscan_if_decision', 'filtered']]
    df.to_csv(filename_out, index=False)
    print(f'Saved to: {filename_out}')
    
  
def dbscan_if(data: pd.DataFrame):
    # DBSCAN - use default parameters
    print('Calculating DBSCAN...')
    def dbscan_by_file(ratings):
        flat = np.array(ratings).reshape(-1, 1)
        db = DBSCAN().fit(flat)
        db_flags = [0 if label == -1 else 1 for label in db.labels_]
        return db_flags
    
    data['dbscan'] = data['filtered'].apply(dbscan_by_file)
    
    # isolation forest (IF)
    print('Calculating IF...')
    def if_by_file(ratings):
        flat = np.array(ratings).reshape(-1, 1)
        if_labels = IsolationForest(random_state=19).fit(flat).predict(flat)
        if_flags = [0 if label == 0 else 1 for label in if_labels]
        return if_flags
    
    data['if'] = data['filtered'].apply(if_by_file)
    
    # remove rating if both DBSCAN and IF identify as an outlier
    print('Calculating ensemble decision...')
    def dbscan_if_ensemble(row):
        db_labels = row['dbscan']
        if_labels = row['if']
        ens_flags = [0 if (db_labels[i] == if_labels[i] == 0) else 1 for i in range(len(db_labels))]
        return ens_flags
    
    data['dbscan_if_decision'] = data.apply(dbscan_if_ensemble, axis=1)
    data['filtered'] = data.apply(lambda row: [rating for rating, decision in zip(row['filtered'], row['dbscan_if_decision'])
                                                if decision == 1],
                                    axis = 1)
   

def get_mos(df: pd.DataFrame):
    """
    Averages the column 'ratings_normalized' to get a normalized MOS score
    
    Args:
        df (pd.DataFrame): a dataframe containing the normalized data
    """
    print('Calculating new MOS...')
    df['mos_normalized'] = df['ratings_normalized'].apply(np.mean)
    

def get_raters(data: pd.DataFrame, participants: pd.DataFrame):
    """
    Given the participants dataframe, adds a column to the data dataframe which corresponds to which raters
    produced each rating.
    
    Args:
        data (pd.DataFrame): the ratings data
        participants (pd.DataFrame): the dataframe containing invidividual participant data
    """
    print('Getting rater information...')
    batch_map = pd.read_csv(FILE_MAP)
    data['raters'] = None

    for _, row in tqdm(participants.iloc[1:].iterrows(), total=len(participants.iloc[1:])):   # skip first row which is question text
        rater = row['participant_id']
        batch = row['batch']
        
        # map rater to a list of files rated
        rated_files = batch_map[batch_map['batch'] == batch]['old_filename'].values.tolist()
        
        # update data to include participant_id for applicable files
        for rf in rated_files:
            mask = data['filename'] == rf
            data.loc[mask, 'raters'] = data.loc[mask, 'raters'].apply(lambda x: x + [rater] if x is not None else [rater])
            
    return data


def normalize(df: pd.DataFrame):
    """
    Use MinMax scaling to normalize participant ratings.
    
    Args:
        df (pd.DataFrame): cleaned (no outliers) rating data
    """
    print('Normalizing ratings...')
    scaler = MinMaxScaler(feature_range=(1,5))
    df_no_lists = df.explode('raters').explode('ratings')
        
    df_no_lists['ratings_normalized'] = (
        df_no_lists.groupby('raters')['ratings']
            .transform(lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).ravel())
    )
    df_no_lists['ratings_normalized'] = df_no_lists['ratings_normalized'].apply(lambda x: int(x) if not pd.isna(x) else x)
    
    norm_df = (
        df_no_lists.groupby('filename', as_index=False).agg({
            'raters': set,
            'average_mos': 'mean',
            'ratings_normalized': list,
            'ratings': list, 
            'zscores': list, 
            'dbscan': list, 
            'if': list, 
            'dbscan_if_decision': list,
            'filtered': list
        })
    )
    
    return norm_df
    
    
def prep_data():
    """
    Reads in the ratings data, converts strings to numeric types, and adds columns for the z-scores/decision 
    flags and the DBSCAN and IF values/decision flags.

    Returns:
        pd.DataFrame: a dataframe containing all data and columns as described.
    """
    print('Preparing data...')
    df = pd.read_csv(DATA)
    participant_df = pd.read_excel(RESPONSES)

    # convert "[...]" to [...]
    df['ratings'] = df['ratings'].apply(lambda x: literal_eval(x))
    
    # add columns to store later results in
    df['zscores'] = None                # 1 if |z-score| > threshold, else 0
    df['dbscan'] = None                 # raw DBSCAN values
    df['if'] = None                     # raw IF values
    df['dbscan_if_decision'] = None     # 1 if BOTH DBSCAN and IF identify rating as an outlier, else 0
    df['filtered'] = None               # ratings that passed z-score, DBSCAN, and IF tests
    
    # add parallel list to 'ratings' giving which worker gave each rating
    df = get_raters(df, participant_df)
    
    return df
  
        
def zscores(data: pd.DataFrame):
    """ 
    Computes the z-score for each rating and removes any ratings whose z-score has absolute value > threshold
    
    Args:
        data (pd.DataFrame): the data to compute z-scores for. Columns ['filename', 'ratings'] must exist.
    """
    print('Computing z-scores...')
    assert 'filename' in data.columns, "Error: Column 'filename' must exist within dataframe passed to zscores()."
    assert 'ratings' in data.columns, "Error: Column 'ratings' must exist within dataframe passed to zscores()."
    
    def zscore_flags(ratings):
        flat = np.asarray(ratings, dtype=float)
        std = flat.std()
        if std < 1e-8:
            # all ratings are basically the same, no outliers
            return [1] * len(flat)
        
        z = (flat - flat.mean())/std
        return [0 if abs(v) > OUTLIER_THRESHOLD else 1 for v in z]
    
    data['zscores'] = data['ratings'].apply(zscore_flags)
    
    # remove identified outliers
    data['filtered'] = data.apply(lambda row: [rating for rating, decision in zip(row['ratings'], row['zscores'])
                                        if decision == 1], 
                                    axis = 1)
    

if __name__ == '__main__':
    main()