# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 23:33:33 2024

@author: Jeremi
"""

import mne
import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from pan_tompkins_debug import pan_tompkins_qrs
from pan_tompkins_debug import heart_rate

fs = 1000
directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

raw = mne.io.read_raw_brainvision("20240725_X00_mat5mins/20240725_X00_jikken_0003.vhdr")
raw.load_data()

# Reconstruct the original events from our Raw object
events, event_ids = mne.events_from_annotations(raw)

raw.set_channel_types({'ECG':'ecg'})
raw.set_channel_types({'vEOG':'eog'})
raw.set_channel_types({'hEOG':'eog'})

tmin = events[3,0]/fs
tmax = events[-3,0]/fs

# iir_params = dict(order=2, ftype='butter', output = 'sos')
# raw_ecg = raw.copy().crop(tmin = tmin, tmax = tmax).pick_types(eeg=False, eog=False, ecg=True).filter(0.5, 150, picks = 'ecg', method ='iir', iir_params = iir_params)
raw_ecg = raw.copy().crop(tmin = tmin, tmax = tmax).pick_types(eeg=False, eog=False, ecg=True)

mne_ecg, mne_time = raw_ecg[:]
mne_ecg = np.squeeze(-mne_ecg)

b, a = signal.butter(2, [0.5, 150], 'bandpass', output= 'ba', fs=fs)
filtered = signal.filtfilt(b,a,mne_ecg)

mne_ecg = filtered.copy()

QRS = pan_tompkins_qrs()
output = QRS.solve(mne_ecg, fs)

hr = heart_rate(mne_ecg, fs)
result = hr.find_r_peaks()
result = np.array(result)

# Clip the x locations less than 0 (Learning Phase)
result = result[result > 0]

# Plotting the R peak locations in ECG signal
plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(0, len(mne_ecg)+1, 500))
plt.plot(mne_time*fs, mne_ecg, color = 'blue')        
plt.scatter(result, mne_ecg[result], color = 'red', s = 50, marker= '*')
# plt.axvline(x = 300000, color = 'r')
# plt.axvline(x = 600000, color = 'r')
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("R Peak Locations")