import pandas as pd
import glob
import os
import pickle
import random

DISTR_FILE = 'mosnet_distr_test.pkl'

# get already perturbed files
perturbed_f = [f.split('/')[-1] for f in glob.glob('perturbed_iucosine/*.wav')]
orig_f = list(set(['_'.join(f.split('_')[0:3]) + '.wav' for f in perturbed_f]))

x_files = []
# Get distribution of test files
tmos_distr, pmos_distr = pickle.load(open(DISTR_FILE, 'rb'))

test_df = pd.read_csv('iucosine.csv')
test_df = test_df[test_df['set'] == 'test']     # restrict to test set
test_df = test_df[test_df['mos5'].notnull()]    # ignore any perturbed files (i.e., we don't have a true MOS)
test_df['trunc'] = test_df['mos5'].apply(lambda x: int(x))
existing_df = test_df[test_df['filename'].isin(orig_f)]

# Get all completed analyses in output directory (i.e., our observed instances)
observed = {1: [], 2: [], 3: [], 4: [], 5: []}
observed_files = orig_f
true_mos_df = pd.read_csv('iucosine.csv')
remove_idx = []
test_files = test_df['filename'].tolist()
for f in observed_files:
    remove_idx.append(test_files.index(f))
        
    # now get associated true MOS
    tmos = int(true_mos_df.loc[true_mos_df['filename'] == f]['mos5'].values[0])
    observed[tmos].append(f)

# Remove already-analyzed files from consideration and get associated tMOS
eligible = test_df.drop(index=remove_idx)
eligible_files = list(set(test_files) - set(observed_files))
eligible_cat = {1: [], 2: [], 3: [], 4: [], 5: []}
for ef in eligible_files:
    cat = int(true_mos_df.loc[true_mos_df['filename'] == ef]['mos5'].values[0])
    eligible_cat.get(cat, []).append(ef)
    
# Sample [args.number] files from remaining test files, 
# following distribution of all test files
goal_sample_sizes = {k: 25-len(observed[k]) for k in observed.keys()}
print(goal_sample_sizes)

for k,v in goal_sample_sizes.items():
    if v <= 0: 
        # randomly select some from what we've already used
        x_files.extend(random.sample(observed[k], 25))
    elif len(eligible_cat[k]) > 0:
        eligible_files = eligible_cat[k]
        x_files.extend(random.sample(eligible_files, v))
        
x_test = test_df[test_df['filename'].isin(x_files)]
with open('todo.txt', 'w') as f:
    space_sep = ' '.join(str(item) for item in x_test['filename'].tolist())
    f.write(space_sep)

x_test = pd.concat([x_test, existing_df], ignore_index=True)
print(len(x_test))
print('Counts:', x_test['trunc'].value_counts())

with open('picked_iucosine_test_perturb.txt', 'w') as f:
    # Convert all elements to strings and join them with a space
    space_separated_string = " ".join(str(item) for item in x_test['filename'].tolist())
    f.write(space_separated_string)
    