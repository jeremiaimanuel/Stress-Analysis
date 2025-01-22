# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 14:21:11 2025

@author: Jeremi
"""

import mne
import os
import pandas as pd
import numpy as np
from scipy import signal as sg
import math
import seaborn as sns
from matplotlib import pyplot as plt

#################################### DEFINE ###################################
second_rest = False
#################################### DEFINE ###################################

################################## LOAD FILE ##################################

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

# fpath = 'filtered_data'
# fpath = 'filtered_data_ica_all'
# fpath = 'filtered_data_asr'
fpath = 'ecg_data'

filtered_data = [os.path.join(fpath,i) for i in os.listdir(fpath)]

files = {number: filtered_data[number] for number in range(len(filtered_data))}

def load_fif_file(fdata):
    raw = mne.io.read_raw_fif(files[fdata], preload=True)
    return (raw)

print(files)
file_number = int(input("Choose File: "))
raw = load_fif_file(file_number)
raw.load_data()

events, event_ids = mne.events_from_annotations(raw)

fs = 1000


#### Full Experiment Plot

fig, axes = plt.subplots(4, 1, figsize=(3, 12))  # Adjust figsize as needed
plt.rcParams.update({
    'font.size': 20,           # Default font size for all elements
    'axes.titlesize': 18,      # Font size for axes titles
    'axes.labelsize': 16      # Font size for axes labels
})

eeg_psd = raw.compute_psd(method='welch',fmin=4,fmax=45).plot_topomap(
    bands = {'Theta (4-8 Hz)': (4, 8),'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30), 'Gamma (30-40 Hz)': (30, 40)}, 
    normalize = True, 
    axes = axes,
    ch_type='eeg')


############ Visualizing R-Peak Correction

import neurokit2 as nk
from wfdb import processing

fs=1000

mne_ecg,_ = raw[:]
mne_ecg = np.squeeze(mne_ecg)

_, info = nk.ecg_process(mne_ecg, sampling_rate = fs) #Getting the R-Peak Location

uncorr = info['ECG_R_Peaks']

corr = processing.correct_peaks(mne_ecg, info['ECG_R_Peaks'], 36, 50, 'up')

start_plot = 0
stop_plot = len(mne_ecg)
f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, sharey = True)

ax1.plot(mne_ecg[start_plot:stop_plot])
ax1.scatter(uncorr, mne_ecg[uncorr], color='red', s=50, marker='*')
ax1.set_title("Uncorrected R-Peak", fontsize=16)
ax1.set_ylabel('Amplitude (mV)', fontsize=14)
ax1.legend()
ax1.grid(False)

ax2.plot(mne_ecg[start_plot:stop_plot])
ax2.scatter(corr, mne_ecg[corr], color='red', s=50, marker='*')
ax2.set_title("Corrected R-Peak", fontsize=16)
ax2.set_xlabel('Samples', fontsize=14)
ax2.set_ylabel('Amplitude (mV)', fontsize=14)
ax2.legend()
ax2.grid(False)

plt.show()