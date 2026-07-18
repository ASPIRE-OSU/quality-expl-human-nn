'''
This program scales all the audio files in COSINE+VOiCES/batches to have the same root mean square (RMS) value as the practice 
samples located in COSINE+VOiCES/practice_samples

@author: Ada Lamba
@version: 09/04/2024
'''
import os
from tqdm import tqdm
import librosa
import numpy as np
import math
import soundfile as sf
from glob import glob

DESIRED_RMS = 0.05
# TODO
PRACTICE_SAMPLES = ['/fs/ess/PAS3309/abarach/COSINE+VOiCES/practice_samples/session-001_1278-F_array1_98.wav',
                    '/fs/ess/PAS3309/abarach/COSINE+VOiCES/practice_samples/session-001_1278-F_close_79.wav']
DATA_FOLDER = '/fs/ess/PAS3309/abarach/quality/listening-study/data'

CHECK = False

def main():
    # get RMS of practice files
    # RMS = 0.05000000409781916 and 0.04999999944120645
    files = glob(DATA_FOLDER + '/**/*.wav', recursive=True)
    
    if CHECK:
        check_rms(files, DESIRED_RMS)
    else:
        # scale practice files to 0.05
        #scale(PRACTICE_SAMPLES[0], DESIRED_RMS)
        #scale(PRACTICE_SAMPLES[1], DESIRED_RMS)

        # get all files in samples directory
        for f in tqdm(files):
            scale(f, DESIRED_RMS)
            
            
def check_rms(files, desired_rms):
    ''' Checks that the RMS of all provided files are near that of the desired value. 
    
    Args:
        files (list): list of files to check
        desired_rms (float): the desired root mean square value for the files

    Returns:
        list: files that are not within 1e-06 of the desired RMS
    '''
    failed_checks = []
    
    print('Checking...')
    for f in tqdm(files):
        signal, sr = librosa.load(f)
        rms = math.sqrt(np.mean(signal**2))
        
        tol = 0.001
        if (rms > desired_rms+tol) or (rms < desired_rms-tol):
            failed_checks.append((f.split('/')[-1], rms))
    
    print('Desired RMS: {}'.format(desired_rms))
    print('Tolerance: 0.001')
    print('Failed checks: {}'.format(failed_checks))
    
    

def scale(filepath, desired_rms):
    ''' Scales the audio file located at filepath such that it has root mean square value desired_rms.

    Args:
        filepath (str): location of the audio file to scale
        desired_rms (float): the desired root mean square value
    
    Updates:
        modifies the audio file located at filepath
    '''
    #print('Scaling...')
    # RMS_desired = sqrt(1/n sum((scale*signal)^2))
    signal, sr = librosa.load(filepath)
    rms = math.sqrt(np.mean(signal**2))
    
    # scale = sqrt((n * RMS^2_desired)/signal^2)    
    #scale = np.sqrt((len(signal) * (desired_rms**2))/signal**2)

    signal = signal/rms * desired_rms

    # save the signal
    sf.write(filepath, signal, sr, 'PCM_24')
    

if __name__ == "__main__":
    main()