
import numpy as np
import scipy.io.wavfile
import torch
import torchaudio
from torch.optim import Optimizer
import time
import os
import timeit
#change the folder_path 
folder_path1 = '/media/audio/rpp/utilities/rpp-unittests/HOST_NEW/audio/../../../TEST_AUDIO_FILES/eight_samples_single_channel_src1/'

file_list = os.listdir(folder_path1)
tot_time = 0
for i in range(100):
    for f in file_list:
        filename = folder_path1+f
        waveform,sample_rate = torchaudio.load(filename)
        spectro = torchaudio.transforms.Spectrogram(n_fft=512,win_length=512,center =True,power=2)
        # MelSpectro = torchaudio.transforms.MelSpectrogram(sample_rate)
        todecible = torchaudio.transforms.AmplitudeToDB(stype="amplitude", top_db=80)
        switcher={
            1:spectro(waveform),
            # 2:MelSpectro(waveform),
            3:todecible(waveform)
        }
        start = timeit. default_timer()
        switcher[3]
        tot_time = tot_time + (timeit. default_timer()-start)
print((tot_time/100) * 1000000)