###
# Framework implementation of [OmniXAI](https://github.com/salesforce/OmniXAI/tree/main)'s Tabular
# explainer for [MOSNet](https://github.com/lochenchou/MOSNet), DNSMOS, and 
# [ScoreQ](https://github.com/alessandroragano/scoreq).
#
# @author: Ada Lamba
# @version: 09/29/2025
###

import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import pandas as pd
import numpy as np
from omnixai.data.tabular import Tabular
from omnixai.explainers.tabular import TabularExplainer
import csv
from tqdm import tqdm
import argparse
import pickle
import glob
import random
from omnixai.sampler.tabular import Sampler

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATA_DIR = '../data/MOS_Quality_INTERSPEECH_2020/16k_speech'
DATA_LIST_PATH = 'data/iucosine.csv'

FS = 16000
NFFT = 320
NUM_FREQBINS = int(1 + NFFT/2)
HOP_LENGTH = 160
WIN_LENGTH = 320

PREPPED_DATA = {'MOSNet': ['data/omnixai_mosnet_iucosine_train.pkl',
                           'data/omnixai_mosnet_iucosine_test.pkl'],
                'DNSMOS': ['data/omnixai_dnsmos_iucosine_train.pkl',
                           'data/omnixai_dnsmos_iucosine_test.pkl'],
                'ScoreQ': ['data/omnixai_scoreq_iucosine_train.pkl',
                           'data/omnixai_scoreq_iucosine_test.pkl']}

OUTPUT_PATHS = {'MOSNet': 'mosnet/results',
                'DNSMOS': 'dnsmos/results',
                'ScoreQ': 'scoreq/results'}

DISTR_FILE = 'data/mosnet_distr_test.pkl'


def main(args):
    if args.model == 'MOSNet':
        from mosnet.mosnet_tabregression_freqfeats import MOSNet_OmniXAI as mtrf
        model_obj = mtrf(FS, NFFT, HOP_LENGTH, WIN_LENGTH)
        
    elif args.model == 'DNSMOS': 
        from dnsmos.dnsmos_tabregression_freqfeats import DNSMOS_OmniXAI as mtrf
        model_obj = mtrf(args.bak_ovr_model_path, FS, NFFT, HOP_LENGTH, args.input_length)
        
    elif args.model == 'ScoreQ':
        from scoreq.scoreq import Scoreq
        from scoreq.scoreq_tabregression_freqfeats import ScoreQ_OmniXAI as mtrf
        model_obj = mtrf(FS, NFFT, HOP_LENGTH)
    
    # Get all data
    args = _parse_args()
    train_data, train_files, test_data, test_files = get_data(args)
    
    # Sample part of training data for DNSMOS as it runs into memory issues
    #if args.model == 'DNSMOS':
    #    train_data = Sampler.subsample(train_data, fraction=0.5)
    
    # Initialize explainer
    print('Initializing explainers')
    explainers = TabularExplainer(
        explainers=['shap','pdp'],
        mode='regression',
        data=train_data,
        model=model_obj.bulk_predict,
        params={
            # TODO - change grid resolution to match # of SHAP values
            "pdp": {"grid_resolution": 53},
            "shap": {"nsamples": args.neighborhood}
        }
    )
    
    # Get samples to explain
    to_explain, to_explain_files = get_files_to_explain(args, model_obj, test_data, test_files)

    # Generate explanations
    print('Generating explanations...')
    if args.local_exp:
        local_expl(args, explainers, to_explain, to_explain_files)
    
    if args.global_exp:
        global_expl(args, explainers)
    
    # Update files.csv (which tracks which files we have results for)
    # note that we *could* just add our list of files to the existing sheet but by redetermining which files are 
    # present, we are being extra careful to avoid a mismatch 
    results_files = ['_'.join(f.split('/')[-1].split('_')[4:9]) for f in glob.glob(OUTPUT_PATHS[args.model] + '/shap/shap_expl_local_*_feat_scores.csv')]
    results_files_df = pd.DataFrame(results_files, columns=['file'])
    results_files_df.to_csv(os.path.join(OUTPUT_PATHS[args.model], 'files.csv'), index=False)
    
    # aggregate shap files
    if args.local_exp:
        print('Aggregating SHAP values...')
        agg_path = os.path.join(OUTPUT_PATHS[args.model], f'shap/shap_expl_local_n{args.neighborhood}_')
        print(f'{agg_path}{results_files[0]}')
        df = pd.read_csv(str(agg_path) + str(results_files[0]))
        for rf in results_files[1:]:
            df2 = pd.read_csv(f'{agg_path}{rf}')
            df = pd.concat([df, df2], ignore_index=True)
        df.to_csv(os.path.join(f'{agg_path}agg.csv'), index=False)
        
    print('Done.')
    

def get_files_to_explain(args, model_obj, test_df, test_files):
    """Selects test data instances to explain based off args. There are several selection strategies:
            1. Random sampling (args.number): determines PDF of all test instances and samples such that
               the true MOS of already explained files match that distribution
            2. Specific choice (args.index): select a file based off its position in the test files list.
            3. Specific file choice (args.source): select a file based off a given filename
            4. Default: explain the first file in the test files list.

    Args:
        args (argparser): program arguments
        test_data (TabularExplainer): all test instance data
        test_files (list): list of filenames for all test data
        
    Returns:
        (TabularExplainer, list): a tuple containing a selected subset of test_data and the list of 
                                  associated file names
    """
    print('Picking files to explain')
    # Convert tabular object to DataFrame so we can use pandas operations
    test_df = test_df.to_pd()
    
    # 1. Random sampling
    if args.number is not None:
        x_files = []
        # Get distribution of test files
        tmos_distr, pmos_distr = pickle.load(open(DISTR_FILE, 'rb'))
        
        # Get all completed analyses in output directory (i.e., our observed instances)
        observed = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        observed_files = []
        true_mos_df = pd.read_csv(DATA_LIST_PATH)
        remove_idx = []
        for file in glob.glob(OUTPUT_PATHS[args.model] + '/shap/shap_expl_local_*_feat_scores.csv'):
            # remove directory path and get filename 
            # format: [dir]/*/shap_expl_local_n##_[***_*******_*******_*****_***.h5]_feat_scores.csv
            # 10 _ and we want everything between numbers (0-indexed): 4-9
            orig_filename = file.split('/')[-1]
            orig_filename = '_'.join(orig_filename.split('_')[4:7])
            if orig_filename in test_files:
                observed_files.append(orig_filename)
                remove_idx.append(test_files.index(orig_filename))
                
            # now get associated true MOS
            tmos = int(true_mos_df.loc[true_mos_df['filename'] == orig_filename]['mos5'].values[0])
            observed[tmos] = observed.get(tmos, 0) + 1
        
        # Remove already-analyzed files from consideration and get associated tMOS
        eligible = test_df.drop(index=remove_idx)
        eligible_files = list(set(test_files) - set(observed_files))
        eligible_cat = {1: [], 2: [], 3: [], 4: [], 5: []}
        for ef in eligible_files:
            cat = int(true_mos_df.loc[true_mos_df['filename'] == ef]['mos5'].values[0])
            eligible_cat.get(cat, []).append(ef)
            
        # Sample [args.number] files from remaining test files, 
        # following distribution of all test files
        target_size = sum(tmos_distr.values()) 
        target_proportions = {k: v/target_size for k,v in tmos_distr.items()}
        total_sample_size = len(observed_files) + args.number
        goal_sample_sizes = {k: max(int(v * total_sample_size)-observed[k], 0) for k, v in target_proportions.items()}
        
        for k,v in goal_sample_sizes.items():
            if v <= 0: 
                continue
            
            eligible_files = eligible_cat[k]
            x_files.extend(random.sample(eligible_files, v))
        
        # if len(x_files) > args.number, random pick
        if len(x_files) > args.number:
            x_files = random.sample(x_files, args.number)
                
        picked_idx = [test_files.index(f) for f in x_files]
        x_test = test_df.iloc[picked_idx]
        print('Picked files:', x_files)

    # use specific test file index
    elif args.index is not None:
        if args.index_end is not None:
            data_range = (args.index, args.index_end)
        else: 
            data_range = (args.index, args.index+1)
            
        x_test = test_df.iloc[data_range[0]:data_range[1]]
        x_files = pd.read_csv(DATA_LIST_PATH)['filename'].tolist()[data_range[0]:data_range[1]]
        
    # use a specific file name
    elif args.source is not None:
        if not isinstance(args.source, list):
            args.source = [args.source]
        x_test, _ = model_obj.prep_data(args.source)
        x_files = x_test.index.to_list()

    # default to first test in list
    else:
        data_range = (0, 1)
        x_test = test_df.iloc[data_range[0]:data_range[1]]
        x_files = pd.read_csv(DATA_LIST_PATH)['filename'].tolist()[data_range[0]:data_range[1]]
        
    x_test = x_test.reset_index(drop=True) 
    x_test = Tabular(x_test, target_column=None)
    return x_test, x_files


def get_data(args):
    all_data = pd.read_csv(DATA_LIST_PATH)
    test_data = all_data[all_data['set'] == 'test']
    test_files = test_data['filename'].tolist()
    train_data = all_data[all_data['set'] == 'train']
    train_files = train_data['filename'].tolist()

    if args.prep_data:
        train_df, max_frames = model_obj.prep_data(train_files, PREPPED_DATA[args.model][0], 'training data')    
        test_df, max_frames = model_obj.prep_data(test_files, PREPPED_DATA[args.model][1], 'test data')    
        
    # we already prepped the data so read that
    else:
        print('Loading data...')
        train_df = pd.read_pickle(PREPPED_DATA[args.model][0])
        test_df = pd.read_pickle(PREPPED_DATA[args.model][1])

    # remove file index
    train_files = list(train_df.index)
    test_files = list(test_df.index)
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # put data into a Tabular object
    train_data = Tabular(train_df, target_column=None)
    test_data = Tabular(test_df, target_column=None)
    
    return train_data, train_files, test_data, test_files


def global_expl(args, explainers, suppress_image=False):
    print('(Global)')
    global_explanations = explainers.explain_global(params={"pdp": {"features":args.features}})
    
    print("PDP results:")    
    # get center frequencies for each band
    bands = [int(f.split('_')[1]) for f in args.features]
    fmax = FS/2
    bandwidth = fmax/(NUM_FREQBINS-1)
    centers = [i * bandwidth for i in range(NUM_FREQBINS)]
    fnames = ['Frequency Band {}\nCenter: {} Hz (±{} Hz)'.format(i, centers[i], bandwidth) for i in bands]
    
    already_log = (args.model == 'DNSMOS')
    plots = global_explanations["pdp"].plot(xlabel='Magnitude (dB)', 
                                            ylabel='Prediction (MOS)', 
                                            feature_names=fnames, 
                                            already_log=already_log,
                                            dB=True)
    
    # save image and data files
    for p_idx in range(len(plots)):
        if not suppress_image:
            output_path = os.path.join(OUTPUT_PATHS[args.model], 'pdp/pdp_expl_global_{}.png'.format(args.features[p_idx]))
            plots[p_idx].savefig(output_path, bbox_inches='tight')
            
        with open(os.path.join(OUTPUT_PATHS[args.model], 'pdp/pdp_expl_global_{}_plots.pkl'.format(args.features[p_idx])), 'wb') as file:
            pickle.dump(plots[p_idx], file, protocol=pickle.HIGHEST_PROTOCOL)
            
    return plots
    
    
def local_expl(args, explainers, x_test, x_files):
    """Performs local explanation via SHAP explanation. 

    Args:
        args (argparser): input arguments to the program
        explainers (TabularExplainer): initialized explainer objects
    """
    print('(Local)')
    local_explanations = explainers.explain(X=x_test)
    
    print('Visualizing explanations...') 
    local_explanations["shap"].plot(output_path=os.path.join(OUTPUT_PATHS[args.model], 'shap/shap_expl_local_n{}'.format(args.neighborhood)), 
                                    files=x_files,
                                    fnames=['freqbin_{}'.format(i) for i in range(NUM_FREQBINS)], 
                                    suppress_image=True)


def _parse_args():
    """
    Reads and stores command line arguments.
    
    Returns:
       args parser object
    """
    parser = argparse.ArgumentParser(prog='OmniXAI for MOSNet')
    parser.add_argument('-m', '--model', required=True, type=str, help='MOSNet, ScoreQ, or DNSMOS')
    parser.add_argument('-p', '--prep_data', required=False, action='store_true', default=False)
    parser.add_argument('-le', '--local_exp', required=False, action='store_true', default=False)
    parser.add_argument('-ge', '--global_exp', required=False, action='store_true', default=False)
    parser.add_argument('-s', '--source', nargs='*', required=False, type=str, default=None, help='filepath to specific file to test')
    parser.add_argument('-n', '--neighborhood', required=False, type=int, default=1)
    parser.add_argument('-i', '--index', required=False, type=int, default=None, help='The index of the test sample to use.')
    parser.add_argument('-ie', '--index_end', required=False, type=int, default=None, help='The index of the text sample to end at.')
    parser.add_argument('-f', '--features', required=False, type=str, nargs="*", default=[], help='A list of target features to explain (global only).')
    parser.add_argument('-r', '--number', required=False, type=int, default=None, help='Instead of providing indices of files to analyze, use this argument to specify the number to analyze and the script will randonly choose some not present in the output directory.')
    parser.add_argument('-mbo', "--bak_ovr_model_path", default='./dnsmos/bak_ovr.onnx', help='Path to ONNX or ckpt model for BAK and OVR prediction')
    parser.add_argument('-l', "--input_length", type=int, default=9)

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()
    main(args)