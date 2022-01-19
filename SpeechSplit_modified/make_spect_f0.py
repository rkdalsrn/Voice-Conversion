import os
import sys
import pickle
import numpy as np
import soundfile as sf
from scipy import signal
import librosa
from librosa.filters import mel
from numpy.random import RandomState
from pysptk import sptk
from utils import butter_highpass
from utils import speaker_normalization
from utils import pySTFT
from glob import iglob
from pydub import AudioSegment


#create path variable
PATH = 'data_VCTK/wav48_silence_trimmed'
saveDir = 'assets/datas'
dirName_flac, subdirList_flac, _ = next(os.walk(PATH))

#Convert flac to wav and Save in same path
 for subdir in sorted(subdirList_flac):
     for filename in iglob(os.path.join(PATH, subdir, '*_mic1.flac')):
         print(filename)
         w_data, w_sr = sf.read(filename)
         filename_wav = filename.replace("flac", "wav")
         print(filename_wav)
         sf.write(filename_wav, w_data, w_sr, format='WAV', endian='LITTLE', subtype='PCM_16')

#Join each small wav files and Create a merged file in ./assets/datas/*.wav
 for subdir in sorted(subdirList_flac):
     print(subdir)
     file_Dir_save = os.path.join(saveDir, subdir)
     if not os.path.exists(file_Dir_save):
         os.makedirs(file_Dir_save)

     combine = AudioSegment.empty()
     for filename in iglob(os.path.join(PATH, subdir, '*_mic1.wav')):
         sound = AudioSegment.from_file(filename, format='wav')
         combine = combine + sound

     file_name = str(subdir + '.wav')
     combine.export(os.path.join(saveDir, subdir, file_name), format='wav')



mel_basis = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
min_level = np.exp(-100 / 20 * np.log(10))
b, a = butter_highpass(30, 16000, order=5)

spk2gen = pickle.load(open('assets/spk2gen.pkl', "rb"))

 
# Modify as needed
rootDir = 'assets/datas'
targetDir_f0 = 'assets/raptf0'
targetDir = 'assets/spmel'

   
dirName, subdirList, _ = next(os.walk(rootDir))
print('Found directory: %s' % dirName)

def down_sample(input_wav, origin_sr, resample_sr):
    y, sr = librosa.load(input_wav, sr=origin_sr)
    resample_file = librosa.resample(y, sr, resample_sr)
    return resample_file

for subdir in sorted(subdirList):
    if not os.path.exists(os.path.join(targetDir, subdir)):
        os.makedirs(os.path.join(targetDir, subdir))
    if not os.path.exists(os.path.join(targetDir_f0, subdir)):
        os.makedirs(os.path.join(targetDir_f0, subdir))
    if spk2gen[subdir] == 'M':
        lo, hi = 50, 250
    elif spk2gen[subdir] == 'F':
        lo, hi = 100, 600
    else:
        raise ValueError
    prng = RandomState(int(subdir[1:]))
    # read audio file
    fileName = str(subdir + '.wav')
    fileName_Dir = os.path.join(dirName, subdir, fileName)
    print(fileName)
    x, fs = sf.read(fileName_Dir)
    x = down_sample(fileName_Dir, fs, 16000)
    fs = 16000
    if x.shape[0] % 256 == 0:
        x = np.concatenate((x, np.array([1e-06])), axis=0)
    y = signal.filtfilt(b, a, x)
    wav = y * 0.96 + (prng.rand(y.shape[0]) - 0.5) * 1e-06

    # compute spectrogram
    D = pySTFT(wav).T
    D_mel = np.dot(D, mel_basis)
    D_db = 20 * np.log10(np.maximum(min_level, D_mel)) - 16
    S = (D_db + 100) / 100

    # extract f0
    f0_rapt = sptk.rapt(wav.astype(np.float32) * 32768, fs, 256, min=lo, max=hi, otype=2)
    index_nonzero = (f0_rapt != -1e10)
    mean_f0, std_f0 = np.mean(f0_rapt[index_nonzero]), np.std(f0_rapt[index_nonzero])
    f0_norm = speaker_normalization(f0_rapt, index_nonzero, mean_f0, std_f0)

    assert len(S) == len(f0_rapt)
    np.save(os.path.join(targetDir, subdir, fileName[:-4]),
            S.astype(np.float32), allow_pickle=False)
    np.save(os.path.join(targetDir_f0, subdir, fileName[:-4]),
            f0_norm.astype(np.float32), allow_pickle=False)
