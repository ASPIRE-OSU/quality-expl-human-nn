import os
import glob
import pandas as pd
from tqdm import tqdm
import random
import numpy as np
import shutil

BASE_PATH = os.getcwd()
SRC_FILES = os.path.join(BASE_PATH, 'src-files.csv')
MAP = os.path.join(BASE_PATH, 'map_to_original_name.csv')
DATA_DIR = '/fs/ess/PAS3309/Data/Speech/MOS_Quality_INTERSPEECH_2020/16k_speech'    # TODO
PERTURB_DIR = '../../network-expl/data/perturbed-iucosine'
BINS = [21]
SCALES = [0.5, 1, 10.0, 1000.0]
GOAL_PARTICIPANTS = 90
GOAL_RATINGS_PER_AUDIO = 5
BATCHES = GOAL_PARTICIPANTS//GOAL_RATINGS_PER_AUDIO
MOVE_FILES = True

df = pd.read_csv(SRC_FILES)
files = df['filename'].tolist()
perturbed_files = [f for f in glob.glob('sid*.wav')]

# make pairs of scales for each file/frequency bin and arrange into "blocks"
# "block": all perturbed pairs associated with a specific file and frequency bin
# For example, block_1 = [(block1_freqbin0_scale1.wav), (block1_freqbin0_scale0.5.wav), (block1_freqbin0_scale10.wav), (block1_freqbin0_scale1000.wav)]
blocks = []
for f in tqdm(files):
    for b in BINS:
        block = []
        for s in SCALES:
            if s == 1:
                pair = [f, f]
            else:
                perturbed_file = f"{f.split('.wav')[0]}_freqbin{b}_scaled{s}.wav"
                pair = [f, perturbed_file]
            # shuffle pair order
            random.shuffle(pair)
            block.append(pair)
        # shuffle block order
        random.shuffle(block)
        blocks.append(block)

# for each batch, randomly select 4 blocks
# 16 pairs per participant, 4 pairs per block => 4 blocks per participant
subset_size = 4
random.shuffle(blocks)
batches = [blocks[i:i+subset_size] for i in range(0, len(blocks), subset_size)]

# flatten batches to be list of pairs in order, not list of blocks and then shuffle the pairs
flattened_batches = []
for batch in batches:
    flattened_batch = []
    for block_idx, block in enumerate(batch):
        for pair in block:
            flattened_batch.append((pair[0], pair[1], block_idx))
    random.shuffle(flattened_batch)
    flattened_batches.append(flattened_batch)
    
# save batch results to file
batch_df = pd.DataFrame(columns=['batch_num', 'block', 'signal_num', 'filename'])
for i, batch in enumerate(flattened_batches):
    signal_num = 0
    for tup in batch:
        freqbin = None
        for string in tup[0:2]:
            if 'freqbin' in string:
                freqbin = int(string.split('_')[3].split('freqbin')[-1])
            
            batch_df.loc[len(batch_df)] = [i, tup[2], signal_num, string]
            signal_num += 1
batch_df.to_csv('batches.csv', index=False)      

if MOVE_FILES:
    print('Moving files...')
    if os.path.exists(MAP): map_df = pd.read_csv(MAP)
    else: map_df = pd.DataFrame(columns=['group', 'batch', 'old_filename', 'new_filename'])
    
    for i, row in tqdm(batch_df.iterrows(), total=len(batch_df)):
        batch_folder = f"batch{row['batch_num']}"
        if not os.path.exists(batch_folder):
            os.makedirs(batch_folder)
        
        # get audio-1 (even signals) and audio-2 (odd signals)
        signal_num = row['signal_num']
        is_audio1 = (signal_num % 2 == 0)
        
        if is_audio1:
            new_filename = f"{batch_folder}_pair{signal_num//2}-audio1.wav"
        else:
            new_filename = f"{batch_folder}_pair{signal_num//2}-audio2.wav"
            
        # rename old file - assuming file is in current directory
        # need to check if a source file, and if so, copy from original data directory
        if 'freqbin' not in row['filename']:
            shutil.copy(os.path.join(DATA_DIR, row['filename']), os.getcwd())
        else:
            shutil.copy(os.path.join(PERTURB_DIR, row['filename']), os.getcwd())
        os.rename(row['filename'], new_filename)
        
        # move into new folder 
        shutil.move(new_filename, batch_folder)
            
        # update map to original name file
        map_df.loc[len(map_df)] = ['group2', row['batch_num'], row['filename'], new_filename]
    
    map_df.to_csv(MAP, index=False)
    
        