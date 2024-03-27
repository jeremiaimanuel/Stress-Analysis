# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 16:10:45 2024

@author: jeje_
"""

import mne
import numpy as np
import scipy
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
from pan_tompkins import pan_tompkins_qrs
from pan_tompkins import heart_rate

fs = 1000
raw = mne.io.read_raw_brainvision("20231019_B68_stroopv1_0002.vhdr")

raw.set_channel_types({'ECG':'ecg'})
raw.set_channel_types({'vEOG':'eog'})
raw.set_channel_types({'hEOG':'eog'})

raw_ecg = raw.crop(tmin = 34.047, tmax = 934.062).copy().pick_types(eeg=False, eog=False, ecg=True)

mne_ecg, mne_time = raw_ecg[:]
mne_ecg = -mne_ecg.T

QRS = pan_tompkins_qrs()
output = QRS.solve(mne_ecg, fs)
bpass = QRS.bpf(mne_ecg)
der = QRS.derivative(bpass,fs)
sqr = QRS.squaring(der)
mwin = QRS.moving_window_integration(sqr, fs)

start_plot = 300
stop_plot = 3300

# Plotting raw signal
plt.figure(figsize = (20,4), dpi = 100)
plt.xticks(np.arange(start_plot,stop_plot, 250))
plt.plot(mne_ecg[start_plot:stop_plot])
# plt.axvline(x = 300000, color = 'r')
# plt.axvline(x = 600000, color = 'r')
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("Raw Signal")

# Plotting bandpassed signal
plt.figure(figsize = (20,4), dpi = 100)
plt.xticks(np.arange(start_plot,stop_plot, 250))
plt.plot(bpass[start_plot:stop_plot])
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("Bandpassed Signal")

# Plotting derived signal
plt.figure(figsize = (20,4), dpi = 100)
plt.xticks(np.arange(start_plot,stop_plot, 250))
plt.plot(der[start_plot:stop_plot])
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("Derivative Signal")

# Plotting squared signal
plt.figure(figsize = (20,4), dpi = 100)
plt.xticks(np.arange(start_plot,stop_plot, 250))
plt.plot(sqr[start_plot:stop_plot])
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("Squared Signal")

# Plotting moving window integrated signal
plt.figure(figsize = (20,4), dpi = 100)
plt.xticks(np.arange(start_plot,stop_plot, 250))
plt.plot(mwin[start_plot:stop_plot])
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("Moving Window Integrated Signal")


# Find the R peak locations
hr = heart_rate(mne_ecg, fs)
result = hr.find_r_peaks()
result = np.array(result)

# Clip the x locations less than 0 (Learning Phase)
result = result[result > 0]

# Calculate the heart rate
heartRate = (60*fs)/np.average(np.diff(result[1:]))
print("Heart Rate",heartRate, "BPM")

# Plotting whole ECG signal
plt.figure(figsize = (20,4), dpi = 100)
plt.xticks(np.arange(0,len(mne_ecg)+1, 500))
plt.plot(mne_ecg[:])
plt.axvline(x = 300000, color = 'r')
plt.axvline(x = 600000, color = 'r')
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("Raw Signal")

# Plotting the R peak locations in ECG signal
plt.figure(figsize = (20,4), dpi = 100)
plt.xticks(np.arange(0, len(mne_ecg)+1, 500))
plt.plot(mne_ecg, color = 'blue')        
plt.scatter(result, mne_ecg[result], color = 'red', s = 50, marker= '*')
plt.axvline(x = 300000, color = 'r')
plt.axvline(x = 600000, color = 'r')
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("R Peak Locations")

r_peak1 = np.unique(result)
r_peak = result.copy()