# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 11:35:38 2025

@author: Jeremi
"""

import mne
import numpy as np
import os
import neurokit2 as nk
from wfdb import processing

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

folder_raw = 'raw_data'
fpath_raw=[i for i in os.listdir(folder_raw) if i.endswith('.vhdr')]

counter = 8

raw = mne.io.read_raw_brainvision(os.path.join(folder_raw, fpath_raw[counter]))

events, event_ids = mne.events_from_annotations(raw)

raw.set_channel_types({'ECG':'ecg'})
raw.set_channel_types({'vEOG':'eog'})
raw.set_channel_types({'hEOG':'eog'})

####EXTRACTING From RAW so Timing Trigger need to be adjusted####

fs = 1000
tmin = events[3,0]/fs #Experiment Begin 
tmax = events[-1,0]/fs #Task End

if fpath_raw[counter] == '20231019_B68_stroop5mins_0001.vhdr':
    tmin = events[9,0]/fs
if fpath_raw[counter] == '20240725_X00_jikken_0003.vhdr':
    tmax = events[-3,0]/fs
if fpath_raw[counter] == '20240725_X00_jikken_0001.vhdr':
    tmax = events[-2,0]/fs
if fpath_raw[counter] == '20240418_B98_jikken_0001.vhdr':
    tmax = tmin + 900

####EXTRACTING From RAW so Timing Trigger need to be adjusted####