# Plots/reports and saves the following demographic information:
#   - Study Duration
#   - Race + Ethnicity
#   - Age Group
#   - Gender
#   - Web Browser
#
# @author: Ada Lamba
# @version: 01/20/2026


import ast
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy.stats import zscore

AGG = False
VERBOSE = True
# Set the default font family to 'serif'
plt.rcParams["font.family"] = "serif"

# Specifically set the 'serif' font to 'DejaVu Serif'
# This assumes DejaVu Serif is installed on your system.
plt.rcParams["font.serif"] = ["DejaVu Serif"]

# If you are also using MathText, set the fontset for that as well
plt.rcParams["mathtext.fontset"] = "dejavuserif"


def main():
    args = parse_args()
    
    # get all data
    all_df = pd.read_excel(args.input)

    # drop first row - it's a subtitle and we don't want to plot it
    all_df.drop(index=0, inplace=True)

    # remove date/size from group names
    all_df['group'] = all_df['group'].apply(lambda x: x.split('_')[0])

    if AGG:
        fig, ax = plt.subplots(2, 2, figsize=(12, 10), dpi=600)
        plot_duration(all_df, ax[0,0], agg=True)
        plot_race_ethnicity(all_df, ax[0, 1], agg=True)
        plot_age_group(all_df, ax[1,0], agg=True)
        plot_gender(all_df, ax[1,1], agg=True)
    else: 
        plot_duration(all_df)
        plot_race_ethnicity(all_df)
        plot_age_group(all_df)
        plot_gender(all_df)
        plot_browser(all_df)

    if AGG:
        fig.tight_layout()
        fig.savefig('../results/agg_demo.png', bbox_inches='tight')

    

def plot_duration(all_df: pd.DataFrame, ax=None, agg=False):
    ''' Plots and saves the average duration for each survey group as well as the aggregated data in the 
    form of a box-and-whisker plot.

    Args:
        all_df (pd.DataFrame): all groups' survey response data.
    '''
    # remove outliers extreme outliers
    zscores = np.abs(zscore(pd.to_numeric(all_df['Duration (in seconds)'])))
    threshold = 5
    df = all_df[zscores < threshold].copy()
    
    # get duration in minutes
    df['Participant Duration'] = pd.to_numeric(df.apply(lambda row: int(row['Duration (in seconds)'])/60.0, axis=1), errors='coerce')

    if agg:
        ax = df.plot(kind='box', vert=False, ax=ax, column='Participant Duration', title='Participant Duration', ylabel='', 
                      xlabel='Duration (minutes)', grid=True)
        ax.set_yticklabels([])
    
    else:
        ax = df.plot(kind='box', column='Participant Duration', by='group', 
                     grid=True, xlabel='Group', ylabel='Duration (minutes)')

    if VERBOSE:
        print('Participant Duration:')
        print(df['Participant Duration'].describe())
        print()
        
    if not agg: plt.savefig('../results/duration.png', bbox_inches='tight')


def plot_race_ethnicity(all_df: pd.DataFrame, ax=None, agg=False):
    ''' Plots and saves the participant race/ethnicity information. 

    Args:
        all_df (pd.DataFrame): all groups' survey response information
    '''
    col_name = 'Q10.race '

    # display categories
    categories = sorted(['Hispanic or Latino', 'Native American or\nAlaska Native', 'Asian', 'Black or African\nAmerican', 
                  'Native Hawaiian or\nOther Pacific Islander', 'White', 'Other'])

    # get pivot table of race counts
    all_df[col_name] = all_df[col_name].apply(lambda x: x.split(','))
    df = all_df.explode(col_name)
    pivot = df.pivot_table(index='group', columns=col_name, aggfunc='size', fill_value=0)
    pivot = pivot.T.sort_index()
    group_order = ['prelim'] + sorted(list(set(pivot.columns.tolist()) - {'prelim'}))
    pivot = pivot[group_order]

    if agg:
        pivot = pivot.T
        pivot.loc['Total'] = pivot.sum()
        pivot.loc['Total'].plot(kind='bar', ax=ax, title='Participant Race/Ethnicity', xlabel='Race/Ethnicity', ylabel='Count')
        categories = [c for c in categories if c.replace('\n', ' ') in pivot.columns.tolist()]

    else:
        ax = pivot.plot(kind='bar', title='Participant Race/Ethnicity by Group', stacked=True, xlabel='Race/Ethnicity', ylabel='Count')
        ax.legend(title='')
        categories = [c for c in categories if c.replace('\n', ' ') in pivot.index.tolist()]

    # bar labels
    # totals = pivot.sum(axis=1)
    # ax.bar_label(ax.containers[-1], labels=[f'{h:g}' for h in totals], label_type='edge', padding=5)
    # ax.set_ylim(0, totals.max() * 1.1)
    
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.grid()
    ax.set_axisbelow(True)
    
    if VERBOSE:
        pivot.loc['Total'] = pivot.sum()
        pivot['Total'] = pivot.sum(axis=1)
        print('Race/Ethnicity:')
        print(pivot)
        print()

    if not agg: plt.savefig('../results/race_ethnicity.png', bbox_inches='tight')


def plot_gender(all_df: pd.DataFrame, ax=None, agg=False):
    '''Plots and saves the participants' reported gender. 

    Args:
        - all_df (pd.DataFrame): all groups' survey response information
    '''
    # get pivot table of gender counts
    pivot = all_df.pivot_table(index='group', columns='Q9.gender', aggfunc='size', fill_value=0)
    pivot = pivot.T.sort_index()
    group_order = ['prelim'] + sorted(list(set(pivot.columns.tolist()) - {'prelim'}))
    pivot = pivot[group_order]

    if agg:
        pivot = pivot.T
        pivot.loc['Total'] = pivot.sum()
        pivot.loc['Total'].plot(kind='bar', ax=ax, title='Participant Genders', xlabel='Gender', rot=0, ylabel='Count')

    else:
        ax = pivot.plot(kind='bar', title='Participant Gender by Group', stacked=True, rot=0, xlabel='Gender', ylabel='Count')
        ax.legend(title='')
    
    # bar labels
    # totals = pivot.sum(axis=1)
    # ax.bar_label(ax.containers[-1], labels=[f'{h:g}' for h in totals], label_type='edge', padding=5)
    # ax.set_ylim(0, totals.max() * 1.1)
    
    ax.grid()
    ax.set_axisbelow(True)


    if VERBOSE:
        pivot.loc['Total'] = pivot.sum()
        pivot['Total'] = pivot.sum(axis=1)
        print('Gender:')
        print(pivot)
        print()
        
    if not agg: plt.savefig('../results/gender.png', bbox_inches='tight')


def plot_age_group(all_df: pd.DataFrame, ax=None, agg=False):
    '''Plots and saves the participants' reported age group. 

    Args:
        - all_df (pd.DataFrame): all groups' survey response information
    '''
    # get pivot table of age counts
    pivot = all_df.pivot_table(index='group', columns='Q4.age ', aggfunc='size', fill_value=0)
    pivot = pivot.T.sort_index()
    group_order = ['prelim'] + sorted(list(set(pivot.columns.tolist()) - {'prelim'}))
    pivot = pivot[group_order]


    if agg:
        pivot = pivot.T
        pivot.loc['Total'] = pivot.sum()
        pivot.loc['Total'].plot(kind='bar', ax=ax, title='Participant Ages', xlabel='Age Range', rot=0, ylabel='Count')
    
    else:
        ax = pivot.plot(kind='bar', title='Participant Age by Group', stacked=True, rot=0, xlabel='Age Range', ylabel='Count')
        ax.legend(title='')

    # bar labels
    # totals = pivot.sum(axis=1)
    # ax.bar_label(ax.containers[-1], labels=[f'{h:g}' for h in totals], label_type='edge', padding=5)
    # ax.set_ylim(0, totals.max() * 1.1)
   
    ax.grid()
    ax.set_axisbelow(True)

    if VERBOSE:
        pivot.loc['Total'] = pivot.sum()
        pivot['Total'] = pivot.sum(axis=1)
        print('Age:')
        print(pivot)
        print()
        
    if not agg: plt.savefig('../results/age.png', bbox_inches='tight')


def plot_browser(all_df: pd.DataFrame, ax=None, agg=False):
    '''Plots and saves the participants' reported web browser.

    Args:
        - all_df (pd.DataFrame): all groups' survey response information
    '''
    # get pivot table of browser counts
    pivot = all_df.pivot_table(index='group', columns='Q8.web_browser', aggfunc='size', fill_value=0)
    pivot = pivot.T.sort_index()
    group_order = ['prelim'] + sorted(list(set(pivot.columns.tolist()) - {'prelim'}))
    pivot = pivot[group_order]

    if agg:
        pivot = pivot.T
        pivot.loc['Total'] = pivot.sum()
        pivot.loc['Total'].plot(kind='bar', ax=ax, title='Participant Web Browsers', xlabel='Browser', rot=0, ylabel='Count')
    
    else:
        ax = pivot.plot(kind='bar', title='Participant Web Browser by Group', stacked=True, rot=0, xlabel='Browser', ylabel='Count')
        ax.legend(title='')

    # bar labels
    # totals = pivot.sum(axis=1)
    # ax.bar_label(ax.containers[-1], labels=[f'{h:g}' for h in totals], label_type='edge', padding=5)
    # ax.set_ylim(0, totals.max() * 1.1)

    ax.grid()
    ax.set_axisbelow(True)

    if not agg: plt.savefig('../results/webbrowser.png', bbox_inches='tight')


def parse_args():
    """
    Reads and stores command line arguments.
    
    Returns:
       args parser object
    """
    parser = argparse.ArgumentParser(prog='Duration Plotter')
    parser.add_argument('--input', required=True, type=str, help='The input file path.')
    return parser.parse_args()


if __name__ == '__main__':
    main()