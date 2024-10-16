# -*- coding: utf-8 -*-
"""
Created on Sun May 26 19:30:16 2024

@author: Jeremi
"""

import os
import mne
import numpy as np
from matplotlib import pyplot as plt
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs

from mne_icalabel import label_components

from pan_tompkins import pan_tompkins_qrs
from pan_tompkins import heart_rate

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

####### List of Path ######
path = ['20231019_B68_stroop5mins/20231019_B68_stroop5mins_0001.vhdr',
        '20240129_B71_mat5mins/20240129_B71_mat5mins_0001.vhdr',
        '20240418_B98_mat2mins/20240418_B98_jikken_0004.vhdr',
        '20240418_B98_mat5mins/20240418_B98_jikken_0003.vhdr',
        '20240418_B98_stroop2mins/20240418_B98_jikken_0002.vhdr',
        '20240418_B98_stroop5mins/20240418_B98_jikken_0001.vhdr',
        '20240725_X00_mat2mins/20240725_X00_jikken_0004.vhdr',
        '20240725_X00_mat5mins/20240725_X00_jikken_0003.vhdr',
        '20240725_X00_stroop2mins/20240725_X00_jikken_0002.vhdr',
        '20240725_X00_stroop5mins/20240725_X00_jikken_0001.vhdr']
####### List of Path ######

######################################### Save ICA Function #########################################
def save_ica(directory):
        
    raw = mne.io.read_raw_brainvision(directory, preload = True)
    # raw = mne.io.read_raw_brainvision(path[file_number], preload = True)
    
    # Reconstruct the original events from our Raw object
    events, event_ids = mne.events_from_annotations(raw)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_channel_types({'ECG':'ecg'})
    raw.set_channel_types({'vEOG':'eog'})
    raw.set_channel_types({'hEOG':'eog'})
    
    # raw.set_eeg_reference(ref_channels='average')
    
    raw.set_montage(montage)
    
    fs = 1000
    tmin = events[3,0]/fs #Experiment Begin 
    tmax = events[-1,0]/fs #Task Begin
    
    if directory == "20231019_B68_stroop5mins/20231019_B68_stroop5mins_0001.vhdr":
        tmin = events[9,0]/fs
    if directory == '20240725_X00_mat5mins/20240725_X00_jikken_0003.vhdr':
        tmax = events[-2,0]/fs
    if directory == '20240725_X00_stroop5mins/20240725_X00_jikken_0001.vhdr':
        tmax = events[-2,0]/fs
    if directory == '20240418_B98_stroop5mins/20240418_B98_jikken_0001.vhdr':
        tmax = tmin + 900
        
    raw_temp = raw.copy().crop(tmin = tmin, tmax = tmax).apply_function(lambda x: -x, picks='ECG') #make a copy
    
    filt_raw = raw_temp.load_data().copy().filter(l_freq=1.0, h_freq=100)
    
    ica = ICA(n_components=15, max_iter="auto",method='infomax', fit_params=dict(extended=True), random_state = 95)
    ica.fit(filt_raw)
    ica
    # ica.plot_sources(raw_temp, show_scrollbars=True)
    # ica.plot_components()
    
    ########## Labeling IC Components ##########
    ic_labels = label_components(filt_raw, ica, method="iclabel")
    print(ic_labels["labels"])
    
    # # exclude_idx = [
    # #     idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    # # ]
    # # print(f"Excluding these ICA components: {exclude_idx}")
    ########## Labeling IC Components ##########
    
    ########## Save ICA Analysis ##########
    ica.save(fname=directory.replace(".vhdr", "-ica.fif"), overwrite = True)
    ########## Save ICA Analysis ##########


######################################### Save ICA Function #########################################

files = {number: path[number] for number in range(len(path))}

print(files)
file_number = int(input("Choose File: "))

raw = mne.io.read_raw_brainvision(path[file_number], preload = True)

# Reconstruct the original events from our Raw object
events, event_ids = mne.events_from_annotations(raw)

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_channel_types({'ECG':'ecg'})
raw.set_channel_types({'vEOG':'eog'})
raw.set_channel_types({'hEOG':'eog'})

# raw.set_eeg_reference(ref_channels='average')
raw.set_montage(montage)

fs = 1000
tmin = events[3,0]/fs #Experiment Begin 
tmax = events[-1,0]/fs #Task Begin

if path[file_number] == "20231019_B68_stroop5mins/20231019_B68_stroop5mins_0001.vhdr":
    tmin = events[9,0]/fs
if path[file_number] == '20240725_X00_mat5mins/20240725_X00_jikken_0003.vhdr':
    tmax = events[-2,0]/fs
if path[file_number] == '20240725_X00_stroop5mins/20240725_X00_jikken_0001.vhdr':
    tmax = events[-2,0]/fs
if path[file_number] == '20240418_B98_stroop5mins/20240418_B98_jikken_0001.vhdr':
    tmax = tmin + 900


########## Labeling IC Components ##########

########## Save ICA Analysis ##########
ica.save(fname=files[file_number].replace(".vhdr", "-ica.fif"), overwrite = True)
########## Save ICA Analysis ##########

sources = ica.get_sources(raw_temp)
# sources = raw_temp.copy()
heog_data = raw_temp.copy().filter(l_freq = 1, h_freq = 10).get_data(picks=['hEOG'])
veog_data = raw_temp.copy().filter(l_freq = 1, h_freq = 10).get_data(picks=['vEOG'])
ecg_data = raw_temp.copy().filter(l_freq = 8, h_freq = 16).get_data(picks='ECG')

########## EOG ICA Analysis ##########
eog_indices, eog_scores = ica.find_bads_eog(raw_temp, ch_name=['vEOG', 'hEOG'], threshold = 0.8, measure='correlation')
# ica.plot_overlay(raw_temp, exclude=eog_indices, picks="eeg")

# ica.plot_properties(raw_temp, picks=eog_indices)
# ica.plot_scores(eog_scores)

ica_choose = str(input('Enter ICA 3 digit number:'))
ica_picked = "ICA"+ica_choose

f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex = True)
eog_IC = sources.copy().filter(l_freq = 1, h_freq = 10).get_data(picks=ica_picked)
ax1.plot(raw_temp.times, np.squeeze(eog_IC), 'b')
ax2.plot(raw_temp.times, np.squeeze(heog_data), 'r')
ax3.plot(raw_temp.times, np.squeeze(veog_data), 'r')
ax1.title.set_text(ica_picked)
ax2.title.set_text('hEOG')
ax3.title.set_text('vEOG')
plt.show()

########## EOG ICA Analysis ##########

########## ECG ICA Analysis ##########
ecg_indices, ecg_scores = ica.find_bads_ecg(raw_temp, ch_name='ECG',threshold = 0.8, measure='correlation')
ica.plot_overlay(raw_temp, exclude=ecg_indices, picks="eeg")

ica.plot_properties(raw_temp, picks=ecg_indices)
ica.plot_scores(ecg_scores)

ica_choose = str(input('Enter ICA 3 digit number:'))
ica_picked = "ICA"+ica_choose

f, (ax1, ax2) = plt.subplots(2, 1, sharex = True)
ecg_IC = sources.copy().filter(l_freq = 8, h_freq = 16).get_data(picks=ica_picked)
# ecg_IC = sources.copy().get_data(picks=ica_picked)
ax1.plot(raw_temp.times, np.squeeze(ecg_IC), 'b')
ax2.plot(raw_temp.times, np.squeeze(ecg_data), 'r')
ax1.title.set_text(ica_picked)
ax2.title.set_text('ECG')
plt.show()
########## ECG ICA Analysis ##########