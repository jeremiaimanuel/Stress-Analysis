# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:48:18 2024

@author: Jeremi
"""

import mne
import os

directory_path = "D:/EEG RESEARCH DATA/"
os.chdir(directory_path)

fpath = 'filtered_data_asr'

rest1_data = []
for i in os.listdir(fpath):
    if i.endswith('-first-asr.fif'):
        rest1_data.append(os.path.join(fpath,i))
        
stress_data = []
for i in os.listdir(fpath):
    if i.endswith('-stress-asr.fif'):
        stress_data.append(os.path.join(fpath,i))

raw_first_rest = mne.io.read_raw_fif(rest1_data[0], preload=True)
raw_stress = mne.io.read_raw_fif(stress_data[0], preload=True)

mne.concatenate_raws([raw_first_rest,raw_stress])
