# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 16:02:18 2024

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

def abs_power_extraction(signal, fs, l_freq, h_freq, tmin, tmax, t_seg = 10, noverlap = 0.9):
    """
    Extract Absolute PSD from Data

    Parameters:
    :param signal: Signal that want to be extracted from
    :param fs: frequency sampling
    :param l_freq: Minimum range of EEG bands frequency (Theta = 4, Alpha = 8, Beta = 12, Gamma = 30)
    :param h_freq: Maxmimum range of EEG bands frequency (Theta = 8, Alpha = 12, Beta = 30, Gamma = 45)
    :param tmin: Start time where the frequency of the signal want to be extracted 
    :param tmax: End time where the frequency of the signal want to be extracted
    :param t_seg: Length of time that want to be segmented (in seconds)
    :param t_overlap: Length of Overlap time (in seconds)
    
    Returns:
    :return: Extracted absolute Power
    """

    freq_data = []

    segment_length = t_seg*fs  # seconds
    overlap = int(segment_length*noverlap)  # usually t_overlap = 9 seconds, so overlap about 90% of segment

    for i in range(len(signal)):
        data_channel = signal[i]
        freq_over_time = []
        for j in range(0, len(data_channel), segment_length - overlap):
            segment = data_channel[j:j + segment_length]

            if len(segment) < segment_length:
                nperseg = len(segment)
            else:
                nperseg = segment_length

            #Compute PSD
            f_eeg, Pxx_eeg = sg.welch(segment, fs, nperseg = nperseg)

            idx_freq = np.where((l_freq<= f_eeg) & (f_eeg<= h_freq))

            abs_power_freq = np.sum(np.abs(Pxx_eeg[idx_freq]))

            #Append
            if math.isnan(abs_power_freq) == False:
                freq_over_time.append(abs_power_freq)
        freq_data.append(freq_over_time[tmin:tmax])
    
    return np.array(freq_data)

eeg_data = raw.pick_types(eeg=True).get_data()
        
theta_data_rest = abs_power_extraction(eeg_data, fs, 4, 8, 0, eeg_newseg1)
theta_data_stress = abs_power_extraction(eeg_data, fs, 4, 8, eeg_newseg1, eeg_newseg2)



def welch_extraction_mne(raw, l_freq, h_freq, tmin, tmax, t_seg=10, t_overlap=9):
    """
    Extract Absolute PSD from Data

    Parameters:
    :param signal: Signal that want to be extracted from
    :param l_freq: Minimum range of EEG bands frequency (Theta = 4, Alpha = 8, Beta = 12, Gamma = 30)
    :param h_freq: Maxmimum range of EEG bands frequency (Theta = 8, Alpha = 12, Beta = 30, Gamma = 45)
    :param tmin: Start time where the frequency of the signal want to be extracted 
    :param tmax: End time where the frequency of the signal want to be extracted
    :param t_seg: Length of time that want to be segmented (in seconds)
    :param t_overlap: Length of Overlap time (in seconds)
    
    Returns:
    :return: Extracted absolute Power
    """
    
    signal = raw.copy().crop(tmin=tmin, tmax=tmax)
    
    psd_epochs = mne.make_fixed_length_epochs(signal, duration = t_seg, overlap = t_overlap)
    
    psd_results = psd_epochs.compute_psd(
        method='welch', 
        fmin=l_freq, 
        fmax=h_freq,
        window='hann',
        n_fft=int(len(psd_epochs.times)))
    
    abs_arr_psd = np.sum(np.abs(psd_results.get_data()), axis = 2)
    
    return abs_arr_psd.T

theta_data_rest_mne = welch_extraction_mne(raw, 4, 8, 0, eeg_newseg1, t_seg=10,t_overlap=9)
theta_data_stress_mne = welch_extraction_mne(raw, 4, 8, eeg_newseg1, eeg_newseg2)
