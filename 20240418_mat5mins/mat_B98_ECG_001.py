# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:43:19 2023

@author: Jeremi
"""

import mne
import os
import numpy as np
import scipy
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
from pan_tompkins import pan_tompkins_qrs
from pan_tompkins import heart_rate

fs = 1000
directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

raw = mne.io.read_raw_brainvision("20240418_mat5mins/20240418_B98_jikken_0003.vhdr")

# Reconstruct the original events from our Raw object
events, event_ids = mne.events_from_annotations(raw)

raw.set_channel_types({'ECG':'ecg'})
raw.set_channel_types({'vEOG':'eog'})
raw.set_channel_types({'hEOG':'eog'})

tmin = events[3,0]/fs
tmax = events[-1,0]/fs

raw_ecg = raw.copy().crop(tmin = tmin, tmax = tmax).pick_types(eeg=False, eog=False, ecg=True)

mne_ecg, mne_time = raw_ecg[:]
mne_ecg = mne_ecg.T
mne_ecg = -mne_ecg

QRS = pan_tompkins_qrs()
output = QRS.solve(mne_ecg, fs)
bpass = QRS.bpf(mne_ecg)
der = QRS.derivative(bpass,fs)
sqr = QRS.squaring(der)
mwin = QRS.moving_window_integration(sqr, fs)

start_plot = 300
stop_plot = 3300

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex = True)
plt.xticks(np.arange(start_plot,stop_plot, 250))
ax1.plot(mne_ecg[start_plot:stop_plot])
ax2.plot(bpass[start_plot:stop_plot])
ax3.plot(der[start_plot:stop_plot])
ax4.plot(sqr[start_plot:stop_plot])
ax5.plot(mwin[start_plot:stop_plot])
ax1.title.set_text("Raw Signal")
ax2.title.set_text('Bandpassed Signal')
ax3.title.set_text('Derivative Signal')
ax4.title.set_text('Squared Signal')
ax5.title.set_text('Moving Window Integrated Signal')
plt.xlabel('Samples')
plt.ylabel('mV')
plt.show()

# # Plotting raw signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(start_plot,stop_plot, 250))
# plt.plot(mne_ecg[start_plot:stop_plot])
# # plt.axvline(x = 300000, color = 'r')
# # plt.axvline(x = 600000, color = 'r')
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Raw Signal")

# # Plotting bandpassed signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(start_plot,stop_plot, 250))
# plt.plot(bpass[start_plot:stop_plot])
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Bandpassed Signal")

# # Plotting derived signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(start_plot,stop_plot, 250))
# plt.plot(der[start_plot:stop_plot])
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Derivative Signal")

# # Plotting squared signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(start_plot,stop_plot, 250))
# plt.plot(sqr[start_plot:stop_plot])
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Squared Signal")

# # Plotting moving window integrated signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(start_plot,stop_plot, 250))
# plt.plot(mwin[start_plot:stop_plot])
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Moving Window Integrated Signal")


# Find the R peak locations
hr = heart_rate(mne_ecg, fs)
result = hr.find_r_peaks()
result = np.array(result)

# Clip the x locations less than 0 (Learning Phase)
result = result[result > 0]

# Plotting the R peak locations in ECG signal
plt.figure(figsize = (20,4), dpi = 100)
plt.xticks(np.arange(0, len(mne_ecg)+1, 500))
plt.plot(mne_ecg, color = 'blue')        
plt.scatter(result, mne_ecg[result], color = 'red', s = 50, marker= '*')
# plt.axvline(x = 300000, color = 'r')
# plt.axvline(x = 600000, color = 'r')
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("R Peak Locations")

r_peak = np.unique(result)

###Pre Process, Data Cleaning ECG###
a = []
a += [value for value in r_peak if value <237]
new_r_peak = [value for value in r_peak if value not in a]
new_r_peak = np.array(new_r_peak)

r_peak = new_r_peak.copy()
###Pre Process, Data Cleaning ECG###

result = r_peak.copy()
rri = r_peak.copy()
rri[1:] = rri[1:] - rri[:-1]
