import argparse
import os
import librosa
import scipy.signal
import soundfile as sf


DATA_DIR = '../../data/MOS_Quality_INTERSPEECH_2020/16k_speech'
FS = 16000
# FFT_SIZE = 512
FFT_SIZE = 320
NUM_FREQBINS = FFT_SIZE // 2 + 1
# HOP_LENGTH = 256
HOP_LENGTH = 160
# WIN_LENGTH = 512
WIN_LENGTH = 320


def _parse_args():
    """
    Reads and stores command line arguments.
    
    Returns:
       args parser object
    """
    parser = argparse.ArgumentParser(prog='Perturb audio')
    parser.add_argument('-b', '--bin', required=True, type=int, help='The index of the frequency bin to modify')
    parser.add_argument('-s', '--scale', required=False, type=float, default=0.5, help='The factor to scale the frequency bin by')
    parser.add_argument('-f', '--file', required=True, type=str, help='The audio file to perturb')
    
    return parser.parse_args()

args = _parse_args()

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

    mag, phase = librosa.magphase(linear)
    return mag, phase

# scale audio
mag, phase = get_spectrograms(os.path.join(DATA_DIR, args.file))
mag[args.bin, :] *= args.scale

# reconstruct audio and save
modified = mag * phase
y_mod = librosa.istft(modified, 
                      n_fft=FFT_SIZE,
                      hop_length=HOP_LENGTH,
                      win_length=WIN_LENGTH,
                      window=scipy.signal.windows.hamming)

outpath = os.path.join('perturbed_iucosine/' + args.file.split('.wav')[0] + '_freqbin{}_scaled{}.wav'.format(args.bin, args.scale))
sf.write(outpath, y_mod, FS)
