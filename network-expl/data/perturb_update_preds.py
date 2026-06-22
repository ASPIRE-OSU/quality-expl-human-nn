import pandas as pd
import os
import glob
from tqdm import tqdm
import onnxruntime as ort
import tensorflow
from tensorflow import keras
from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Dense, Dropout, Conv2D
from tensorflow.keras.layers import LSTM, TimeDistributed, Bidirectional
from tensorflow.keras.constraints import max_norm
import math
from urllib.request import urlretrieve
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy.polynomial.polynomial as poly
import soundfile as sf
import scipy.signal

PREDS = 'iucosine.csv'
PERTURB_DIR = 'perturbed_iucosine'
DATA_DIR = '../../data/MOS_Quality_INTERSPEECH_2020/16k_speech'
MODEL_PATH = '../dnsmos/bak_ovr.onnx'
PRE_TRAINED = '../mosnet/mosnet-50HzBands.h5'
FS = 16000
FFT_SIZE = NFFT = 320
NUM_FREQBINS = SGRAM_DIM = FFT_SIZE // 2 + 1
HOP_LENGTH = 160
WIN_LENGTH = 320
INPUT_LENGTH = 9
COEFS_OVR = np.array([8.924546794696789354e-01, 6.609981731940616223e-01,
                        7.600269530243179694e-02])
MOSNET_PREDS = False
DNSMOS_PREDS = False
SCOREQ_PREDS = True

def main():
    # existing prediction data
    df = pd.read_csv(PREDS)

    # perturbed files
    all_files = [f.split('/')[-1] for f in glob.glob(PERTURB_DIR + '/*.wav')]

    # update prediction data to include all perturbed files
    print('Updating prediction file to include all files...')
    files_to_add = [f for f in all_files if f not in df['filename'].tolist()]
    for f in files_to_add:
        orig_filename = '_'.join(f.split('_')[:3]) + '.wav'
        audiotype = df[df['filename'] == orig_filename]['type'].values[0]
        new_row = {'filename': f, 'set': 'test', 'type': audiotype}
        df.loc[len(df)] = new_row

    if MOSNET_PREDS: 
        print('Predicting MOSNet...')
        MOSNet = CNN50()
        model = MOSNet.build(print_info=False)
        model.load_weights(PRE_TRAINED)
        
        # get files from original prediction data for which we don't have a prediction
        mosnet_files = df[df['mosnet_pred'].isnull()]['filename'].tolist()
        
        for f in tqdm(mosnet_files):
            if 'freqbin' in f:
                fpath = os.path.join(PERTURB_DIR, f)
            else: 
                fpath = os.path.join(DATA_DIR, f)
            
            row_idx = df.index[df['filename'] == f].tolist()[0]
            
            pred = predict_mosnet(model, fpath)
            df.at[row_idx, 'mosnet_pred'] = pred
        df.to_csv(PREDS, index=False)
    
    if DNSMOS_PREDS:
        print('Predicting DNSMOS...')
        session_bak_ovr = ort.InferenceSession(MODEL_PATH)
        
        # get files from original prediction data for which we don't have a prediction
        dnsmos_files = df[df['dnsmos_pred'].isnull()]['filename'].tolist()
        
        for f in tqdm(dnsmos_files):
            if 'freqbin' in f:
                fpath = os.path.join(PERTURB_DIR, f)
            else: 
                fpath = os.path.join(DATA_DIR, f)
            
            row_idx = df.index[df['filename'] == f].tolist()[0]
            
            pred = predict_dnsmos(session_bak_ovr, fpath)
            df.at[row_idx, 'dnsmos_pred'] = pred
        df.to_csv(PREDS, index=False)
       
    if SCOREQ_PREDS: 
        print('Predicting ScoreQ...')
        nr_scoreq = Scoreq(data_domain='natural', mode='nr')
        
        # get files from original prediction data for which we don't have a prediction
        scoreq_files = df[df['scoreq_pred'].isnull()]['filename'].tolist()
        
        for i, f in tqdm(enumerate(scoreq_files), total=len(scoreq_files)):
            if 'freqbin' in f:
                fpath = os.path.join(PERTURB_DIR, f)
            else: 
                fpath = os.path.join(DATA_DIR, f)
            
            row_idx = df.index[df['filename'] == f].tolist()[0]
            
            pred = predict_scoreq(nr_scoreq, fpath)
            df.at[row_idx, 'scoreq_pred'] = pred
            
            if i % 100 == 0:
                df.to_csv(PREDS, index=False)
                
    df.to_csv(PREDS, index=False)

def predict_mosnet(model, f):
    mag = get_spectrograms(f)
    timestep = mag.shape[0]
    mag = np.reshape(mag,(1, timestep, SGRAM_DIM))
    [avg, _] = model.predict(mag, verbose=0, batch_size=1)
    return avg[0][0]

def predict_dnsmos(session_bak_ovr, f):
    # dnsmos
    audio, _ = sf.read(os.path.join(DATA_DIR, f))
    inputs = []
    len_samples = int(INPUT_LENGTH*FS)
    while len(audio) < len_samples:
        audio = np.append(audio, audio)
        
    num_hops = int(np.floor(len(audio)/FS) - INPUT_LENGTH)+1
    hop_len_samples = FS
    
    for idx in range(num_hops):
        audio_seg = audio[int(idx*hop_len_samples) : int((idx+INPUT_LENGTH)*hop_len_samples)]
        # output shape: (time, freqbin) = (901, 161)
        input_features = audio_logpowspec(audio=audio_seg, sr=FS).astype('float32')
        
        inputs.append(input_features)
     
    if len(inputs) > 0: 
        onnx_inputs_bak_ovr = {inp.name: inputs for inp in session_bak_ovr.get_inputs()}
        mos_bak_ovr = session_bak_ovr.run(None, onnx_inputs_bak_ovr)
        outs = poly.polyval(mos_bak_ovr[0][:, 2], COEFS_OVR)
        pred = np.mean(outs)
        
    return pred


def predict_scoreq(model, f):
    score = model.predict(f)
    return score


# MOSNet 
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
 

# DNSMOS
def audio_logpowspec(audio, nfft=NFFT, hop_length=160, sr=FS):
    # powspec.shape == (..., 1 + nfft/2, n_frames) == (..., 161, # frames)
    powspec = (np.abs(librosa.core.stft(audio, n_fft=nfft, hop_length=hop_length)))**2
    logpowspec = np.log10(np.maximum(powspec, 10**(-12)))
    return logpowspec.T  


# ScoreQ
# The wav2vec 2.0 model's CNN feature extractor has a total stride of 320
PADDING_MULTIPLE = 320

def dynamic_pad(x, multiple=PADDING_MULTIPLE, dim=-1, value=0):
    """Pads the input tensor to be a multiple of PADDING_MULTIPLE."""
    tsz = x.size(dim)
    required_len = math.ceil(tsz / multiple) * multiple
    remainder = required_len - tsz
    pad_offset = (0,) * (-1 - dim) * 2
    return F.pad(x, pad_offset + (0, remainder), value=value)

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(n - self.n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

# PyTorch classes needed for the use_onnx=False fallback
class TripletModel(nn.Module):
    def __init__(self, ssl_model, ssl_out_dim, emb_dim=256):
        super(TripletModel, self).__init__()
        self.ssl_model = ssl_model
        self.ssl_features = ssl_out_dim
        self.embedding_layer = nn.Sequential(nn.ReLU(), nn.Linear(self.ssl_features, emb_dim))
    
    def forward(self, wav, phead=False):
        wav = wav.squeeze(1)
        res = self.ssl_model(wav, mask=False, features_only=True)
        x = res['x']
        x = torch.mean(x, 1)
        if phead:
            x = self.embedding_layer(x)
        x = torch.nn.functional.normalize(x, dim=1)
        return x

class MosPredictor(nn.Module):
    def __init__(self, pt_model, emb_dim=768):
        super(MosPredictor, self).__init__()
        self.pt_model = pt_model
        self.mos_layer = nn.Linear(emb_dim, 1)
        
    def forward(self, wav):
        x = self.pt_model(wav, phead=False)
        if len(x.shape) == 3: x.squeeze_(2)
        out = self.mos_layer(x)
        return out


class Scoreq():
    """
    Main class for handling the SCOREQ audio quality assessment model.
    Defaults to using high-performance ONNX models.
    """
    def __init__(self, data_domain='natural', mode='nr', use_onnx=True):
        """
        Initializes the Scoreq object.

        Args:
            data_domain (str): Domain of audio ('natural' or 'synthetic').
            mode (str): Mode of operation ('nr' or 'ref').
            use_onnx (bool): If True (default), uses fast ONNX models. If False, falls back to original PyTorch/fairseq method.
        """
        self.data_domain = data_domain
        self.mode = mode
        self.use_onnx = use_onnx
        self.model = None
        self.session = None
        self.device = 'cpu'

        if self.use_onnx:
            self._init_onnx()
        else:
            self._init_pytorch()

    def _init_onnx(self, print_arch=True):
        """Initializes the ONNX Runtime session."""
        import onnxruntime as ort
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else ['CPUExecutionProvider']
        
        domain_part = 'telephone' if self.data_domain == 'natural' else 'synthetic'
        mode_part = 'adapt_nr' if self.mode == 'nr' else 'fixed_nmr'
        onnx_filename = f"{mode_part}_{domain_part}.onnx"
        
        ZENODO_ONNX_URLS = {
            'adapt_nr_telephone.onnx': 'https://zenodo.org/records/15739280/files/adapt_nr_telephone.onnx',
            'fixed_nmr_telephone.onnx': 'https://zenodo.org/records/15739280/files/fixed_nmr_telephone.onnx',
            'adapt_nr_synthetic.onnx': 'https://zenodo.org/records/15739280/files/adapt_nr_synthetic.onnx',
            'fixed_nmr_synthetic.onnx': 'https://zenodo.org/records/15739280/files/fixed_nmr_synthetic.onnx',
        }
        
        model_url = ZENODO_ONNX_URLS.get(onnx_filename)
        if not model_url:
            raise ValueError(f"Invalid model combination: domain='{self.data_domain}', mode='{self.mode}'")
            
        model_path = self._download_model(onnx_filename, model_url, cache_dir_name="onnx-models")
        print(model_path)
        
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.device = self.session.get_providers()[0]
        print(f"SCOREQ (ONNX) initialized on provider: {self.device}")

    def _download_model(self, filename, url, cache_dir_name):
        """Helper to download a model from a URL with a progress bar."""
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "scoreq", cache_dir_name)
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, filename)

        if not os.path.exists(model_path):
            print(f"Downloading {filename}...")
            try:
                with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=filename) as t:
                    urlretrieve(url, model_path, reporthook=t.update_to)
                print("Download complete.")
            except Exception as e:
                print(f"Error downloading model: {e}")
                if os.path.exists(model_path): os.remove(model_path)
                raise e
        
        return model_path

    def predict(self, test_path, ref_path=None):
        """Makes predictions on audio files."""
        if self.use_onnx:
            return self._predict_onnx(test_path, ref_path)
        else:
            return self._predict_pytorch(test_path, ref_path)

    def _predict_onnx(self, test_path, ref_path=None):
        """Prediction using the ONNX model."""
        input_name = self.session.get_inputs()[0].name
        
        test_wave_raw = self.load_processing(test_path)
        test_wave_padded = dynamic_pad(test_wave_raw).numpy()
        
        if self.mode == 'nr':
            score = self.session.run(None, {input_name: test_wave_padded})[0].item()
        elif self.mode == 'ref':
            if ref_path is None: raise ValueError("ref_path must be provided for reference mode.")
            ref_wave_raw = self.load_processing(ref_path)
            ref_wave_padded = dynamic_pad(ref_wave_raw).numpy()
            
            test_emb = self.session.run(None, {input_name: test_wave_padded})[0]
            ref_emb = self.session.run(None, {input_name: ref_wave_padded})[0]
            score = np.linalg.norm(test_emb - ref_emb).item()
        else:
            raise ValueError("Invalid mode specified.")
            
        return score

    def _predict_pytorch(self, test_path, ref_path=None):
        """Prediction using the original PyTorch model."""
        test_wave_raw = self.load_processing(test_path)
        test_wave_padded = dynamic_pad(test_wave_raw).to(self.device)
        
        with torch.no_grad():
            if self.mode == 'nr':
                score = self.model(test_wave_padded).item()
            else:
                if ref_path is None: raise ValueError("ref_path must be provided.")
                ref_wave_raw = self.load_processing(ref_path)
                ref_wave_padded = dynamic_pad(ref_wave_raw).to(self.device)
                
                test_emb = self.model(test_wave_padded)
                ref_emb = self.model(ref_wave_padded)
                score = torch.cdist(test_emb, ref_emb).item()
        return score

    def load_processing(self, filepath, target_sr=16000):
        """Loads and preprocesses an audio file."""
        #wave, sr = torchaudio.load(filepath)
        wave_np, sr = librosa.load(filepath)
        wave = torch.from_numpy(wave_np).unsqueeze(0)
        if wave.shape[0] > 1: wave = wave.mean(dim=0, keepdim=True)
        if sr != target_sr: wave = torchaudio.transforms.Resample(sr, target_sr)(wave)
        return wave
    
    
if __name__ == '__main__':
    main()