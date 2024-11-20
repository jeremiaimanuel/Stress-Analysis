# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 12:59:06 2024

@author: Jeremi
"""

import mne
import os


fpath = ['20231019_B68_stroop5mins/20231019_B68_stroop5mins_0001.vhdr',
        '20240129_B71_mat5mins/20240129_B71_mat5mins_0001.vhdr',
        '20240418_B98_mat2mins/20240418_B98_jikken_0004.vhdr',
        '20240418_B98_mat5mins/20240418_B98_jikken_0003.vhdr',
        '20240418_B98_stroop2mins/20240418_B98_jikken_0002.vhdr',
        '20240418_B98_stroop5mins/20240418_B98_jikken_0001.vhdr',
        '20240725_X00_mat2mins/20240725_X00_jikken_0004.vhdr',
        '20240725_X00_mat5mins/20240725_X00_jikken_0003.vhdr',
        '20240725_X00_stroop2mins/20240725_X00_jikken_0002.vhdr',
        '20240725_X00_stroop5mins/20240725_X00_jikken_0001.vhdr']

folder_ecg = "ecg_data"

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

def save_ecg(directory, eog_only = False):
    raw = mne.io.read_raw_brainvision(directory, preload = True)
    # raw = mne.io.read_raw_brainvision(path[file_number], preload = True)
    
    # Reconstruct the original events from our Raw object
    events, event_ids = mne.events_from_annotations(raw)
        
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
    
    raw_ecg = raw.pick('ECG').crop(tmin = tmin, tmax = tmax).filter(l_freq=0.5, h_freq=150)
    raw_ecg.apply_function(lambda x: -x)
    
    _, file_name = os.path.split(directory)
    raw_ecg.save(os.path.join(folder_ecg,file_name.replace(".vhdr", "-ecg.fif")), overwrite=True)
    

for i in fpath:
    save_ecg(i, True) #Save data which only reject EOG ICA

