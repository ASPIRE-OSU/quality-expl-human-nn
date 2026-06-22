import pandas as pd

TARGET = 10

all_files = pd.read_csv('mosnet/results/files.csv')['file'].tolist()
curr_files = pd.read_csv('scoreq/results/files.csv')['file'].tolist()

# remove _feat_scores.csv
all_files = [f[:f.index('_feat_scores.csv')] for f in all_files]
curr_files = [f[:f.index('_feat_scores.csv')] for f in curr_files]

# remove common files
diff = list(set(all_files) - set(curr_files))

# pick and report files
print('Picked files:')
for f in diff[:TARGET]:
    print(f + ' \\')