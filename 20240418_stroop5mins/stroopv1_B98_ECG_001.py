# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:43:19 2023

@author: Jeremi
"""

import mne
import numpy as np
import scipy
import os
from scipy import signal
from scipy.fft import fft, fftfreq
from matplotlib import pyplot as plt
from pan_tompkins import pan_tompkins_qrs
from pan_tompkins import heart_rate

fs = 1000
directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

raw = mne.io.read_raw_brainvision("20240418_stroop5mins/20240418_B98_jikken_0001.vhdr")

raw.set_channel_types({'ECG':'ecg'})
raw.set_channel_types({'vEOG':'eog'})
raw.set_channel_types({'hEOG':'eog'})

raw_ecg = raw.crop(tmin = 52.356, tmax = 943.601).copy().pick_types(eeg=False, eog=False, ecg=True)

mne_ecg, mne_time = raw_ecg[:]
mne_ecg = np.squeeze(-mne_ecg)

# b, a = signal.butter(2, [0.5, 150], 'bandpass', output= 'ba', fs=fs)
# filtered = signal.filtfilt(b,a,mne_ecg)

# mne_ecg = filtered.copy()

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


# Find the R peak locations
hr = heart_rate(mne_ecg, fs)
result = hr.find_r_peaks()
result = np.array(result)

# Clip the x locations less than 0 (Learning Phase)
result = result[result > 0]

r_peak = result.copy()
###Pre Process, Data Cleaning ECG###
a = []
a += [value for value in r_peak if 500 <= value <= 1000]
new_r_peak = [value for value in r_peak if value not in a]
new_r_peak = np.array(new_r_peak)

r_peak = new_r_peak.copy()

del(a, new_r_peak)
###Pre Process, Data Cleaning ECG###

result = r_peak.copy()
rri = np.diff(result[:])

# # Plotting whole ECG signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(0,len(mne_ecg)+1, 500))
# plt.plot(mne_ecg[:])
# plt.axvline(x = 300000, color = 'r')
# plt.axvline(x = 600000, color = 'r')
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Raw Signal")

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


# rri = r_peak.copy()
# rri[1:] = rri[1:] - rri[:-1]

# Calculate the heart rate
heartRate = (60*fs)/np.average(rri)
print("Heart Rate",heartRate, "BPM")


exp_duration = (len(mne_ecg) - r_peak[0] - (len(mne_ecg) - r_peak[-1]))/fs  #Tot duration - First Peak - (Tot Duration - Last peak) 

#Interpolated Data
x_vals = np.linspace(r_peak[1],r_peak[-1],int(exp_duration*fs))
splines = scipy.interpolate.splrep(r_peak[1:],rri)
y_vals = scipy.interpolate.splev(x_vals,splines)

plt.figure(figsize = (20,4),dpi = 100)
plt.plot(x_vals/fs,y_vals)
# plt.axvline(x = newseg1, color = 'r')
# plt.axvline(x = newseg2, color = 'r')

plt.title('Interpolated RRI')
plt.ylabel('RRI [ms]')
plt.xlabel('Time [sec]')

#Data grouping
y_firstrest = y_vals[0:292490]
y_stress = y_vals[292490:591245]
y_secondrest = y_vals[591245:891245]

#Spectrogram RRI
nperseg = 100000
noverlap = nperseg*2/3
f1,t1,Sxx = scipy.signal.spectrogram(y_vals, fs=fs, nperseg = nperseg, noverlap = noverlap, scaling='spectrum')
plt.figure(figsize = (20,4), dpi = 100)
plt.pcolormesh(t1, f1, np.abs(Sxx), shading='gouraud', snap= True)
plt.title('Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.axis([t1[0], t1[-1], 0, 0.4])
plt.axvline(x = 300, color = 'r')
plt.axvline(x = 600, color = 'r')
plt.colorbar()
plt.show()

#Measure SDNN
sdnn_rest_1 = np.std(y_firstrest)
sdnn_test = np.std(y_stress)
sdnn_rest_2 = np.std(y_secondrest)
print("Standard Deviation during First Rest: ", sdnn_rest_1)
print("Standard Deviation during CSWT: ", sdnn_test)
print("Standard Deviation during Second Rest: ", sdnn_rest_2)

#Applying Fourier Transform
time_step = np.average(x_vals[1:] -  x_vals[0:-1])

sample_freq = fftfreq(y_vals.size, d=time_step)
y = fft(y_vals)
A = np.abs(y)

plt.figure(figsize = (20,4),dpi = 100)
plt.plot(sample_freq,y)

#