import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from sklearn.base import BaseEstimator

import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
import numpy.polynomial.polynomial as poly
from tqdm import tqdm
from omnixai.data.tabular import Tabular
import pandas as pd
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATA_DIR = '../../data/Speech/MOS_Quality_INTERSPEECH_2020/16k_speech'
PERTURB_DIR = '../data/perturbed_iucosine'


class DNSMOS_Estimator(BaseEstimator):
    def __init__(self, *, param=161):
        self.param = param
        self.omnixai = DNSMOS_OmniXAI('bak_ovr.onnx', 16000, 320, 160, 9)
        self._estimator_type = 'regressor'
        
    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def predict(self, X):
        # X is 2D - convert to 3D
        X = X.reshape(X.shape[0], 161, 901)
        return self.omnixai.bulk_predict(X)
        

class DNSMOS_OmniXAI:
    def __init__(self, bak_ovr_model_path, fs, nfft, hop_length, input_length):
        self.model = ort.InferenceSession(bak_ovr_model_path)
        self.fs = fs
        self.nfft = nfft
        self.num_freqbins = int(1+nfft/2)
        self.hop_length = hop_length
        self.input_length = input_length
        
        # from: DNS-Challenge/DNS-Challenge/DNSMOS/dnsmos_local.py
        # Coefficients for polynomial fitting
        self.coefs_sig = np.array([9.651228012789436761e-01, 6.592637550310214145e-01, 
                                   7.572372955623894730e-02])
        self.coefs_bak = np.array([-3.733460011101781717e+00,2.700114234092929166e+00,
                                   -1.721332907340922813e-01])
        self.coefs_ovr = np.array([8.924546794696789354e-01, 6.609981731940616223e-01,
                                   7.600269530243179694e-02])


    def prep_data(self, input_file, output_file=None, name='data', print_pad=True):
        print('Prepping {}...'.format(name))
        columns = ['filename'] + ['freqbin_{}'.format(i) for i in range(self.num_freqbins)]
        max_length = 0
        rows = []
        # read each filename and convert to spectrogram
        # use a while loop without iterrows so we can delete rows we're done with
        for f in tqdm(input_file):
            if 'scaled' in f:
                audio, _ = sf.read(os.path.join(PERTURB_DIR, f))
            else:    
                audio, _ = sf.read(os.path.join(DATA_DIR, f))
        
            if len(audio) < 2*self.fs:
                # audio is too short, skip processing
                continue
            
            len_samples = int(self.input_length*self.fs)
            while len(audio) < len_samples:
                audio = np.append(audio, audio)
                
            num_hops = int(np.floor(len(audio)/self.fs) - self.input_length)+1
            hop_len_samples = self.fs
            
            for idx in range(num_hops):
                audio_seg = audio[int(idx*hop_len_samples) : int((idx+self.input_length)*hop_len_samples)]
                # output shape: (time, freqbin) = (901, 161)
                input_features = self.audio_logpowspec(audio=audio_seg).astype('float32')
                input_features_t = np.transpose(np.array(input_features))
                
                flattened = [f] + [input_features_t[i] for i in range(len(input_features_t))]
                rows.append(flattened)
                    
        # clean up some memory in case that dataframe is big
        del audio, input_features, flattened
                
        df = pd.DataFrame(rows, columns=columns)
        df = df.set_index('filename')
        if output_file is not None:
            df.to_pickle(output_file)
        
        return df, None


    def audio_logpowspec(self, audio):
        # powspec.shape == (..., 1 + nfft/2, n_frames) == (..., 161, # frames)
        powspec = (np.abs(librosa.core.stft(audio, n_fft=self.nfft, hop_length=self.hop_length)))**2
        logpowspec = np.log10(np.maximum(powspec, 10**(-12)))
        return logpowspec.T        
    
    
    def bulk_predict(self, data):
        print('Bulk predict called')
                
        if isinstance(data, Tabular):
            data = data.to_pd()
            data = data.to_numpy()
        elif isinstance(data, pd.DataFrame):
            data = data.to_numpy()
            
        inputs = []
        for i in range(len(data)):
            # DNSMOS can predict speech quality (SIG), background noise quality (BAK), and overall quality (OVRL), we only care about OVRL
            #input_features = row.drop('file')
            input_features = data[i].tolist()
            
            # transpose if needed
            if len(input_features) == self.num_freqbins:
                input_features = [list(row) for row in zip(*input_features)]
            
            input_features = [np.asarray(row, dtype=np.float32) for row in input_features]
            inputs.append(input_features)
        
        input_split = np.array_split(inputs, 100)
        predictions = []
        for i in input_split:
            onnx_inputs_bak_ovr = {inp.name: i for inp in self.model.get_inputs()}
            mos_bak_ovr = self.model.run(None, onnx_inputs_bak_ovr)
            predictions.extend(poly.polyval(mos_bak_ovr[0][:, 2], self.coefs_ovr))
        
        return np.asarray(predictions)
    