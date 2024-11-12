# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 20:42:15 2024

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
fpath = 'filtered_data_asr'

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

trg0 = events[0,0] #Experiment Begin 
trg1 = events[1,0] #Task Begin
if files[file_number].find('B98_jikken_0001')>=1:
    trg2 = events[-1,0]
    trg3 = trg0 + 900000
else:
    trg2 = events[-2,0] #Task End
    trg3 = events[-1,0] #Experiment End

tmin = trg0/fs
tmax = trg3/fs

#Segment in Samples
eeg_seg1 = trg1 - trg0
eeg_seg2 = trg2 - trg0
eeg_seg3 = trg3 - trg0

#Segment in ms
eeg_newseg1 = int(eeg_seg1/fs)
eeg_newseg2 = int(eeg_seg2/fs)
eeg_newseg3 = int(eeg_seg3/fs)


freq_plot = [{'Theta (4-8 Hz)': (4, 8)},
             {'Alpha (8-12 Hz)': (8, 12)},
             {'Beta (12-30 Hz)': (12, 30)},
             {'Gamma (30-45 Hz)': (30, 45)}]

# bands = {'Theta (4-8 Hz)': (4, 8),
#          'Alpha (8-12 Hz)': (8, 12), 
#          'Beta (12-30 Hz)': (12, 30),
#          'Gamma (30-45 Hz)': (30, 45)}

# eeg_psd = raw.compute_psd(method='welch',fmin=4,fmax=45,tmin = 0, tmax= 10)
# eeg_psd.plot_topomap(bands = freq_plot, normalize = True, ch_type='eeg')

# epochs = mne.make_fixed_length_epochs(raw, duration=900)
# evoked = epochs.average()
# times_rest = np.arange(0, 300, 10)
# times_stress = np.arange(300, 600, 10)
# evoked.plot_topomap(times_rest, ch_type="eeg", ncols=8, nrows="auto", vlim = (-20,20))
# evoked.plot_topomap(times_stress, ch_type="eeg", ncols=8, nrows="auto", vlim = (-20,20))


psd_epochs = mne.make_fixed_length_epochs(raw, duration = 10, overlap = 9)
psd_results = psd_epochs.compute_psd(method='welch', fmin=4, fmax=8, n_fft = 256, window = 'hann')
abs_arr_psd = np.sum(np.abs(psd_results.get_data()), axis = 2) #Calculate the Absolute PSD Welch

rest1 = raw.copy().crop(tmin = 0, tmax = eeg_newseg1)
rest1_epo = mne.make_fixed_length_epochs(rest1, duration = 10, overlap = 9)
rest_psd_results = rest1_epo.compute_psd(method='welch', fmin=4, fmax=8, n_fft = 256, window ='hann')
abs_rest_arr_psd = np.sum(np.abs(rest_psd_results.get_data()),axis = 2)

stress = raw.copy().crop(tmin = eeg_newseg1, tmax = eeg_newseg2)
stress_epo = mne.make_fixed_length_epochs(stress, duration = 10, overlap = 9)
stress_psd_results = stress_epo.compute_psd(method='welch', fmin=4, fmax=8)


arr_psd = psd_results.get_data() # Convert PSD data into array
abs_arr_psd = np.sum(np.abs(arr_psd[:]), axis = 2) #Calculate the Absolute PSD Welch

# fig, ax = plt.subplots()
# psd_results.plot_topomap(bands = {'Theta (4-8 Hz)': (4, 8)}, axes = ax)

n_epochs = len(psd_results[:10])

# Set up a grid with enough rows and columns
n_cols = 10  # Adjust this depending on how many subplots you want per row
n_rows = int(np.ceil(n_epochs / n_cols))

# Create a figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 3))
axes = axes.flatten()  # Flatten to easily index each axis

# Plot each topomap in a different subplot
for i, (ax, epoch) in enumerate(zip(axes, rest_psd_results[:10])):
    rest_psd_results[i].plot_topomap(bands={'Theta (4-8 Hz)': (4, 8)}, axes=ax, show=False, vlim = (0,3))
    ax.set_title(f'Epoch {i+1}')

# Hide any unused axes
for ax in axes[n_epochs:]:
    ax.axis('off')

plt.tight_layout()
plt.show()


