# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:43:19 2023

@author: Jeremi
"""

import mne
import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from pan_tompkins import pan_tompkins_qrs
from pan_tompkins import heart_rate

fs = 1000
directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

raw = mne.io.read_raw_brainvision("20240418_mat5mins/20240418_B98_jikken_0003.vhdr")
raw.load_data()

# Reconstruct the original events from our Raw object
events, event_ids = mne.events_from_annotations(raw)

raw.set_channel_types({'ECG':'ecg'})
raw.set_channel_types({'vEOG':'eog'})
raw.set_channel_types({'hEOG':'eog'})

tmin = events[3,0]/fs
tmax = events[-1,0]/fs

# b, a = signal.butter(2, [0.5, 150], 'bandpass', output= 'ba', fs=fs)

iir_params = dict(order = 2, ftype='butter', output= 'ba')
filt = mne.filter.create_filter(data = None, sfreq = fs, l_freq = 0.5, h_freq = 150, method = 'iir', iir_params = iir_params)

raw_ecg = raw.copy().crop(tmin = tmin, tmax = tmax).pick_types(eeg=False, eog=False, ecg=True).filter(l_freq = 0.5, h_freq = 150, picks = 'ecg', method ='iir', iir_params = filt)
# raw_ecg = raw.copy().crop(tmin = tmin, tmax = tmax).pick_types(eeg=False, eog=False, ecg=True)

mne_ecg, mne_time = raw_ecg[:]
mne_ecg = np.squeeze(-mne_ecg)

# filt = mne.filter.create_filter(mne_ecg, sfreq = fs, l_freq = 0.5, h_freq = 150, method = 'iir', iir_params = iir_params)

QRS = pan_tompkins_qrs()
output = QRS.solve(mne_ecg, fs)

# Find the R peak locations
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

r_peak = np.unique(result)

###Pre Process, Data Cleaning ECG###
# a = []
# a += [value for value in r_peak if value <237]
# new_r_peak = [value for value in r_peak if value not in a]
# new_r_peak = np.array(new_r_peak)

r_peak = result[result>=237]
###Pre Process, Data Cleaning ECG###

result = r_peak.copy()
rri = r_peak.copy()
rri[1:] = rri[1:] - rri[:-1]



####################Bandpassed Signal####################
b, a = signal.butter(2, [0.5, 150], 'bandpass', fs = fs)
filtered = signal.filtfilt(b,a,np.squeeze(mne_ecg))
# filtered = filtered.T
out_filt = QRS.solve(filtered,fs)
hr = heart_rate(filtered, fs)
result_filt = hr.find_r_peaks()
result_filt = np.array(result_filt)

# Clip the x locations less than 0 (Learning Phase)
result_filt = result_filt[result_filt > 0]

plt.figure(figsize = (20,4), dpi = 100)
plt.xticks(np.arange(0, len(mne_ecg)+1, 500))
plt.plot(filtered, color = 'blue')        
plt.scatter(result_filt, filtered[result_filt], color = 'red', s = 50, marker= '*')
# plt.axvline(x = 300000, color = 'r')
# plt.axvline(x = 600000, color = 'r')
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("R Peak Locations")


