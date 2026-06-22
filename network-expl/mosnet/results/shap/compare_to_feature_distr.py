import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

shap_df = pd.read_csv('shap_df.csv').set_index('filename')
mag_df = pd.read_csv('shap_avg_mag.csv').set_index('filename')

#avg_shap = shap_df.mean().tolist()
avg_shap = shap_df.abs().mean().tolist()
avg_mag = mag_df.mean().tolist()
x = [i for i in range(len(shap_df.columns))]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["DejaVu Serif"]
plt.rcParams['mathtext.fontset'] = 'dejavuserif' # or 'cm'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=600)
ax1.plot(x, avg_shap)
ax2.plot(x, avg_mag)
ax1.set_xlabel('Frequency Band (50 Hz)')
ax2.set_xlabel('Frequency Band (50 Hz)')
ax1.set_ylabel('Average SHAP Value')
ax2.set_ylabel('Average Magnitude')
ax1.set_title('SHAP Distribution by Frequency Band')
ax2.set_title('Feature Distribution of Audio Signals')
fig.savefig('shap_vs_mag.png', bbox_inches='tight')
plt.clf()

# now look at correlation of features for these samples
# rows = trials, columns = variables
corr = mag_df.corr(method='pearson')

plt.figure(figsize=(24, 20), dpi=600)
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.title('Feature Magnitude Correlation')
plt.savefig('feat_corr.png', bbox_inches='tight')
plt.clf()

# now look at correlation of our identified features for these samples
# rows = trials, columns = variables
corr = mag_df[[f'freqbin_{i}' for i in [0, 1, 2, 3, 4, 5, 31, 40, 41]]].corr(method='pearson')

plt.figure(figsize=(14, 10), dpi=600)
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidth=0.5)
plt.title('Feature Magnitude Correlation')
plt.savefig('feat_corr_simplified.png', bbox_inches='tight')
plt.clf()