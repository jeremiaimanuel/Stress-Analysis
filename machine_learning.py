# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:31:47 2024

@author: Jeremi
"""

import mne
import numpy as np
import os
# import math
import scipy
from scipy import signal as sg
from matplotlib import pyplot as plt
# from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs, read_ica
import pandas as pd
# from pan_tompkins import pan_tompkins_qrs
# from pan_tompkins import heart_rate


###### Load Data ######

directory_path = "D:/EEG RESEARCH DATA/epoch_data"
os.chdir(directory_path)

epoch_rest = mne.read_epochs('20240418_B98_jikken_0003-epoch_first_rest-epo.fif')
epoch_stress = mne.read_epochs('20240418_B98_jikken_0003-epoch_stress-epo.fif')
epoch_rest2 = mne.read_epochs('20240418_B98_jikken_0003-epoch_second_rest-epo.fif')

array_rest = epoch_rest.get_data()
array_stress = epoch_stress.get_data()
array_rest2 = epoch_rest2.get_data()

segment_length = array_rest.shape[2] // 10
ch_length = array_rest.shape[1]
# Initialize an empty list to store the segmented arrays
segmented_arrays = np.zeros(ch_length, array_rest.shape[2]//segment_length * array_rest.shape[0])

for i in range(array_rest.shape[0]):
    data_epoch = array_rest[i]
    for j in range(len(data_epoch)):
        segment_index = 0
        for k in range(0,array_rest.shape[2], segment_length):
            segment = data_epoch[j, k:k+segment_length]
            avg_segment = np.average(segment)
            
            segmented_arrays[j,i*segment_length+segment_index] = avg_segment
            segment_index +=1
        
        
# Convert the list to a numpy array
segmented_array = np.array(segmented_arrays)

# Reshape the array to the desired shape
segmented_array = segmented_array.reshape(array_rest.shape[0], array_rest.shape[1], -1)

print(segmented_array.shape)

# raw = mne.io.read_raw_brainvision("20240418_mat5mins/20240418_B98_jikken_0003.vhdr")

# # Reconstruct the original events from our Raw object
# events, event_ids = mne.events_from_annotations(raw)

# montage = mne.channels.make_standard_montage('standard_1020')
# raw.set_channel_types({'ECG':'ecg'})
# raw.set_channel_types({'vEOG':'eog'})
# raw.set_channel_types({'hEOG':'eog'})

# raw.set_montage(montage)
# #fig = raw.plot_sensors(show_names=True)


# ###### Segment Grouping ######

# fs = 1000

# trg0 = events[3,0] #Experiment Begin 
# trg1 = events[4,0] #Task Begin
# trg2 = events[-2,0] #Task End
# trg3 = events[-1,0] #Experiment End

# tmin = trg0/fs
# tmax = trg3/fs

# #Segment in Samples
# eeg_seg1 = trg1 - trg0
# eeg_seg2 = trg2 - trg0
# eeg_seg3 = trg3 - trg0

# #Segment in ms
# eeg_newseg1 = int(eeg_seg1/fs)
# eeg_newseg2 = int(eeg_seg2/fs)
# eeg_newseg3 = int(eeg_seg3/fs)

# ###### R-Wave Detection ######

# raw_ecg = raw.copy().pick_types(eeg=False, eog=False, ecg=True).crop(tmin = tmin, tmax = tmax) #make a copy

# mne_ecg, mne_time = raw_ecg[:]
# mne_ecg = np.squeeze(-mne_ecg)

# b, a = sg.butter(2, [0.5, 150], 'bandpass', output= 'ba', fs=fs)
# filtered = sg.filtfilt(b,a,mne_ecg)

# mne_ecg = filtered.copy()

# QRS_detector = pan_tompkins_qrs()
# output = QRS_detector.solve(mne_ecg, fs)

# # Convert ecg signal to numpy array
# signal = mne_ecg.copy()

# # Find the R peak locations
# hr = heart_rate(signal,fs)
# result = hr.find_r_peaks()
# result = np.array(result)

# # Clip the x locations less than 0 (Learning Phase)
# result = result[result > 0]

# ######Pre Process, Data Cleaning ECG######

# result = result[result>=237]


# ###### Making Pan Tompkins Events ######
# r_peak_onset = []
# for i in range(len(result)):
#     ons_idx = int(fs*tmin)+result[i]
#     r_peak_onset.append(ons_idx)

# pan_tompkins_events = np.zeros((len(r_peak_onset), 3), dtype=int)

# pan_tompkins_events[:, 0] = r_peak_onset
# pan_tompkins_events[:, 1] = 0 
# pan_tompkins_events[:, 2] = 7

# ###### EEG ICA ######

# raw_temp = raw.copy().crop(tmin = tmin, tmax = tmax) #make a copy
# raw_temp.load_data()

# ica = read_ica("20240418_mat5mins/20240418_B98_jikken_0003-ica.fif")
# eog_indices, eog_scores = ica.find_bads_eog(raw_temp, ch_name=['vEOG', 'hEOG'], threshold = 0.8, measure='correlation')

# ica.exclude = eog_indices

# reconst_raw = raw_temp.copy()
# ica.apply(reconst_raw)

# ###### ERP: HEP ######

# new_filt_raw = reconst_raw.load_data().copy().filter(l_freq=1.0, h_freq=40, picks = ['eeg'])

# new_filt_firstrest = new_filt_raw.copy().crop(tmin = 0, tmax = eeg_newseg1)
# new_filt_stress = new_filt_raw.copy().crop(tmin = eeg_newseg1, tmax = eeg_newseg2)
# new_filt_secondrest = new_filt_raw.copy().crop(tmin = eeg_newseg2)

# epoch_tmin = -0.2
# epoch_tmax = 0.6
# baseline = (None,0)

# first_rest_epoch = mne.Epochs(new_filt_firstrest,events = pan_tompkins_events, tmin= epoch_tmin, tmax = epoch_tmax, baseline = baseline, picks = ['eeg'])
# stress_epoch = mne.Epochs(new_filt_stress,events = pan_tompkins_events, tmin= epoch_tmin, tmax = epoch_tmax, baseline = baseline, picks = ['eeg'])
# second_rest_epoch = mne.Epochs(new_filt_secondrest,events = pan_tompkins_events, tmin= epoch_tmin, tmax = epoch_tmax, baseline = baseline, picks = ['eeg'])