###
# Implementation of [OmniXAI](https://github.com/salesforce/OmniXAI/tree/main)'s Tabular
# explainer for [ScoreQ](https://github.com/alessandroragano/scoreq).
#
# @author: Ada Lamba
# @version: 09/29/2025
###
import os
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

import torch
import librosa
import numpy as np
from tqdm import tqdm
import math
import torch.nn.functional as F
import pandas as pd
from omnixai.data.tabular import Tabular
import csv
import tensorflow as tf
import pickle    
import os
from urllib.request import urlretrieve
import torch.nn as nn
import torchaudio

# The wav2vec 2.0 model's CNN feature extractor has a total stride of 320
PADDING_MULTIPLE = 320

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
DATA_PATH = '../../data/MOS_Quality_INTERSPEECH_2020/16k_speech'
PERTURB_DIR = '../data/perturbed_iucosine'

class ScoreQ_OmniXAI:
    # The wav2vec 2.0 model's CNN feature extractor has a total stride of 320
    def __init__(self, fs, nfft, hop_length, padding_multiple=320):
        self.model = Scoreq(data_domain='natural', mode='nr')
        self.fs = fs
        self.nfft = nfft
        self.num_freqbins = int(1+nfft/2)
        self.hop_length = hop_length
        self.padding_multiple = padding_multiple
    

    def bulk_predict(self, data):
        print('Bulk predict called.')
        preds = []
        
        if isinstance(data, Tabular):
            data = data.to_pd()
            data = data.to_numpy()
        elif isinstance(data, pd.DataFrame):
            data = data.to_numpy() 
        
        inputs = []
        for flat_spec in data:
            # unflatten spectrogram
            if len(flat_spec) == self.num_freqbins: flat_spec = flat_spec.T
            spec = np.asarray([np.asarray(row) for row in flat_spec])
            
            # convert from spectrogram to waveform
            wave = librosa.istft(spec,
                                n_fft=self.nfft,
                                hop_length=self.hop_length)
            inputs.append(wave)
        
        """Prediction using the ONNX model."""
        def arr_split(x, max_size=100):
            return np.array_split(x, math.ceil(len(x)/max_size))
        
        inputs = arr_split(inputs)
        for i in inputs:           
            outputs = self.model.session.run(None, {inp.name: i.astype(np.float32) for inp in self.model.session.get_inputs()})
            scores = outputs[0]
            preds.extend([s.item() for s in scores])

        return np.asarray(preds)


    def prep_data(self, input_file, output_file=None, name='data', print_pad=True, target_sr=16000):
        print('Preparing {}...'.format(name))
        
        columns = ['filename'] + ['freqbin_{}'.format(i) for i in range(self.num_freqbins)]
        rows = []
        
        # check pad length that ScoreQ data will expect
        example_data = pd.read_pickle('/fs/ess/PAS3309/abarach/quality/omnixai/data/omnixai_scoreq_iucosine_test.pkl').iloc[0]
        max_length = example_data['freqbin_0'].shape[0]
        
        for f in tqdm(input_file):
            if 'scaled' in f:
                wave_np, sr = librosa.load(os.path.join(PERTURB_DIR, f), sr=self.fs)
            else:    
                wave_np, sr = librosa.load(os.path.join(DATA_PATH, f), sr=self.fs)
            
            # scoreq.load_processing():
            # get waveform
            wave = torch.from_numpy(wave_np).unsqueeze(0)
            if wave.shape[0] > 1: wave = wave.mean(dim=0, keepdim=True)
            wave_padded = np.asarray(self.dynamic_pad(wave))
            
            # convert to spectrogram
            spec = librosa.stft(y=wave_padded,
                                n_fft=self.nfft,
                                hop_length=self.hop_length)
            spec = spec[0]
            flat_spec = [f] + [spec[i] for i in range(len(spec))]
            rows.append(flat_spec)
            
            if spec.shape[1] > max_length:
                max_length = spec.shape[1]
            
        df = pd.DataFrame(rows, columns=columns)
        df = df.set_index('filename')
        
        if print_pad:
            print(max_length)
        
        df = self.pad_data(df, max_length)
        if output_file is not None:
            df.to_pickle(output_file)
            
        return df, None
        

    # from scoreq.py
    def dynamic_pad(self, x, multiple=None, dim=-1, value=0):
        """Pads the input tensor to be a multiple of self.padding_multiple."""
        if multiple is None:
            multiple = self.padding_multiple
        tsz = x.size(dim)
        required_len = math.ceil(tsz / multiple) * multiple
        remainder = required_len - tsz
        pad_offset = (0,) * (-1 - dim) * 2
        return F.pad(x, pad_offset + (0, remainder), value=value)

    # result = np.zeros(reference_shape)
    # result[:array.shape[0],:array.shape[1],:array.shape[2]] = array
    def pad_data(self, data, pad_len=None):
        print('Padding data...')
        if pad_len is None:
            pad_len = max(len(x) for x in data)
        df = data.map(lambda x: np.pad(x, (0, pad_len-len(x)), mode='constant'))
        return df


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

    def _init_pytorch(self):
        """Initializes the original PyTorch/fairseq model."""
        try:
            import fairseq
        except ImportError:
            raise ImportError(
                "PyTorch/fairseq mode requires 'fairseq' and 'torch'. "
                "Please install them with: pip install scoreq[pytorch]"
            )
        
        print("Initializing in PyTorch mode. `fairseq` and `torch` are required.")
        if torch.cuda.is_available(): self.device = 'cuda'
        else: self.device = 'cpu'

        url_w2v = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt"
        CHECKPOINT_PATH = self._download_model("wav2vec_small.pt", url_w2v, "pt-models")
        
        # Temporarily monkey-patch torch.load to default to weights_only=False.
        # This is necessary because fairseq's internal loading function does not
        # expose this argument, and it's required for newer PyTorch versions to
        # load old checkpoints containing non-tensor data.
        original_torch_load = torch.load
        try:
            def new_torch_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
            
            torch.load = new_torch_load
            
            w2v_model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([CHECKPOINT_PATH])
        finally:
            torch.load = original_torch_load
        
        ssl_model = w2v_model[0]
        ssl_model.remove_pretraining_modules()

        pt_model = TripletModel(ssl_model, ssl_out_dim=768, emb_dim=256)
        
        if self.mode == 'nr': model = MosPredictor(pt_model, emb_dim=768)
        else: model = pt_model
            
        PT_URLS = {
            ('natural', 'nr'): 'https://zenodo.org/records/13860326/files/adapt_nr_telephone.pt',
            ('natural', 'ref'): 'https://zenodo.org/records/13860326/files/fixed_nmr_telephone.pt',
            ('synthetic', 'nr'): 'https://zenodo.org/records/13860326/files/adapt_nr_synthetic.pt',
            ('synthetic', 'ref'): 'https://zenodo.org/records/13860326/files/fixed_nmr_synthetic.pt',
        }
        model_key = (self.data_domain, self.mode)
        model_url = PT_URLS.get(model_key)
        if not model_url:
            raise ValueError(f"Invalid model combination: domain='{self.data_domain}', mode='{self.mode}'")
        
        MODEL_PATH = self._download_model(os.path.basename(model_url), model_url, "pt-models")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=self.device, weights_only=False))
        
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        print(f'SCOREQ (PyTorch) running on: {self.device}')

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