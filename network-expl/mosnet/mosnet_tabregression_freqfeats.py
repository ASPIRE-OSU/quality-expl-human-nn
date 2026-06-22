###
# Implementation of [OmniXAI](https://github.com/salesforce/OmniXAI/tree/main)'s Tabular
# explainer for [MOSNet](https://github.com/lochenchou/MOSNet).
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
import csv
from tqdm import tqdm
import pickle
import tensorflow
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.constraints import max_norm
import scipy
import scipy.signal
import librosa
import random
random.seed(1984)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATA_DIR = '../../data/MOS_Quality_INTERSPEECH_2020/16k_speech'
PERTURB_DIR = '../data/perturbed_iucosine'
MODEL_PATH = 'mosnet-50HzBands.h5'
DATA_LIST_PATH = '../data/iucosine.csv'
FS = 16000
FFT_SIZE = 320
SGRAM_DIM = FFT_SIZE // 2 + 1
HOP_LENGTH = 160
WIN_LENGTH = 320


def get_spectrograms(sound_file, fs=FS, fft_size=FFT_SIZE): 
    # Loading sound file
    y, _ = librosa.load(sound_file, sr=fs) # or set sr to hp.sr.

    # Preemphasis
    #y = np.append(y[0], y[1:] - PREEMPHASIS * y[:-1])

    # stft. D: (1+n_fft//2, T)
    linear = librosa.stft(y=y,
                     n_fft=fft_size, 
                     hop_length=HOP_LENGTH, 
                     win_length=WIN_LENGTH,
                     window=scipy.signal.windows.hamming,
                     )

    # magnitude spectrogram
    mag = np.abs(linear) #(1+n_fft/2, T)
    
    # shape in (T, 1+n_fft/2)
    return np.transpose(mag.astype(np.float32))


class MOSNet_OmniXAI:
    def __init__(self, fs, nfft, hop_length, win_length):
        mosnet_model = CNN50()
        mosnet_model = mosnet_model.build()
        mosnet_model.load_weights(MODEL_PATH)
        self.model = mosnet_model
        self.fs = fs
        self.nfft = nfft
        self.num_freqbins = int(1+nfft/2)
        self.hop_length = hop_length
        self.win_length = win_length


    # Build bulk prediction method
    def bulk_predict(self, tensor_array):
        #print('Bulk predict called.')
        predictions = []
        if isinstance(tensor_array, Tabular):
            tensor_array = tensor_array.to_pd()
            tensor_array = tensor_array.to_numpy()
        elif isinstance(tensor_array, pd.DataFrame):
            tensor_array = tensor_array.to_numpy() 
        # elif isinstance(tensory_array, np.ndarray)
        for row in tensor_array:
            # row corresponds to one input
            # convert from cols: freq_bins with values = [time0, time1, ...]
            # to a ndarray with rows = times and columns = frequencies
            # Equivalent code for dataframe: df.apply(lambda col: col.explode()).reset_index(drop=True)
            input_array = np.stack(row).T
            
            # input_array = pd.DataFrame(input_array).T
            # input_array.columns = ['freqbin_{}'.format(i) for i in range(NUM_FREQBINS)]
            #input = input_array.apply(lambda col: col.explode()).reset_index(drop=True)
            # now we've got a dataframe of shape (_, 257) and need to convert to ndarray (1, _, 257)
            # input = input_array.to_numpy()
            input = input_array[np.newaxis, ...]
            
            # [avg, _] = mosnet.predict(t, verbose=0, batch_size=1)
            [avg, _] = self.model.predict(input, verbose=0, batch_size=1)
            predictions.append(avg[0][0])
        return np.asarray(predictions)


    def prep_data(self, input_file, output_file=None, name='data', print_pad=True):
        print('Preparing {}...'.format(name))
        columns = ['filename'] + ['freqbin_{}'.format(i) for i in range(self.num_freqbins)]
        df = pd.DataFrame(columns=columns)
        max_length = 0
        for f in tqdm(input_file):
            # spec.shape == (frames, freqbins==161)
            spec = get_spectrograms(os.path.join(DATA_DIR, f))
            spec_t = np.transpose(spec)
            
            if spec.shape[0] > max_length:
                max_length = spec.shape[0]

            # convert spec to be [freqbin1 value @ frame 1, freqbin1 value @ frame 2, ...]
            flattened_spec = [spec_t[i] for i in range(len(spec_t))]
            flattened_spec = [f] + flattened_spec
            new_row = pd.DataFrame([flattened_spec], columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
            
        df = self.pad_data(df, max_length)
        df = df.set_index('filename')
        if output_file is not None:
            df.to_pickle(output_file)
        if print_pad:
            print(max_length)
        return df, max_length


    def pad_data(data, pad_len=None):
        print('Padding data...')
        if pad_len is None:
            pad_len = max(len(x) for x in data)
        df = data.map(lambda x: np.pad(x, (0, pad_len-len(x)), mode='constant'))
        return df
    

class CNN50(object):
    
    def __init__(self):
        print('CNN-50 init')
        
    def build(self, print_info=True):

        _input = keras.Input(shape=(None, 161))
        
        re_input = layers.Reshape((-1, 161, 1), input_shape=(-1, 161))(_input)
        
        # CNN
        conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(re_input)
        conv1 = (Conv2D(16, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv1 = (Conv2D(16, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv1)
        
        conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv1)
        conv2 = (Conv2D(32, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv2 = (Conv2D(32, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv2)
        
        conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv2)
        conv3 = (Conv2D(64, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv3 = (Conv2D(64, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv3)
        
        conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv3)
        conv4 = (Conv2D(128, (3,3), strides=(1, 1), activation='relu', padding='same'))(conv4)
        conv4 = (Conv2D(128, (3,3), strides=(1, 3), activation='relu', padding='same'))(conv4)
        
        # DNN
        flatten = TimeDistributed(layers.Flatten())(conv4)
        dense1=TimeDistributed(Dense(64, activation='relu'))(flatten)
        dense1=Dropout(0.3)(dense1)

        frame_score=TimeDistributed(Dense(1), name='frame')(dense1)

        average_score=layers.GlobalAveragePooling1D(name='avg')(frame_score)
        
        model = Model(outputs=[average_score, frame_score], inputs=_input)
        
        # print architecture info
        if print_info:
            print(model.summary())
        
        return model
