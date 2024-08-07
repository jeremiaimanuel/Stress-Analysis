# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 12:20:49 2024

@author: jeje_
"""

import os
import mne
import numpy as np
from matplotlib import pyplot as plt
from mne.preprocessing import ICA, create_ecg_epochs, create_eog_epochs

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

raw = mne.io.read_raw_brainvision("20240418_mat2mins/20240418_B98_jikken_0004.vhdr")

# Reconstruct the original events from our Raw object
events, event_ids = mne.events_from_annotations(raw)

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_channel_types({'ECG':'ecg'})
raw.set_channel_types({'vEOG':'eog'})
raw.set_channel_types({'hEOG':'eog'})

raw.set_montage(montage)

fs = 1000
tmin = events[3,0]/fs #Experiment Begin 
tmax = events[-1,0]/fs #Experiment End

raw_temp = raw.copy().crop(tmin = tmin, tmax = tmax) #make a copy


regexp = r"(ECG|vEOG|hEOG)"
artifact_picks = mne.pick_channels_regexp(raw_temp.ch_names, regexp=regexp)

eog_evoked = create_eog_epochs(raw_temp).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))

ecg_evoked = create_ecg_epochs(raw_temp).average()
ecg_evoked.apply_baseline(baseline=(None, -0.2))

filt_raw = raw_temp.load_data().copy().filter(l_freq=1.0, h_freq=None)

ica = ICA(n_components=32, max_iter="auto", random_state = 95)
ica.fit(filt_raw)
ica
ica.plot_sources(raw_temp, show_scrollbars=True)
ica.plot_components()

########## Save ICA Analysis ##########
ica.save(fname="20240418_mat2mins/20240418_B98_jikken_0004-ica.fif", overwrite = True)
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