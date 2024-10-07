# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 13:48:18 2024

@author: Jeremi
"""

import mne
import os
from scipy.stats import ttest_ind
from matplotlib import pyplot as plt
import numpy as np

directory_path = "D:/EEG RESEARCH DATA/"
os.chdir(directory_path)

fpath = 'filtered_data_asr'

# rest1_data = []
# for i in os.listdir(fpath):
#     if i.endswith('-first-asr.fif'):
#         rest1_data.append(os.path.join(fpath,i))
        
# stress_data = []
# for i in os.listdir(fpath):
#     if i.endswith('-stress-asr.fif'):
#         stress_data.append(os.path.join(fpath,i))
        
data = []
for i in os.listdir(fpath):
    if i.endswith('-asr.fif'):
        data.append(os.path.join(fpath,i))

# raw_first_rest = mne.io.read_raw_fif(rest1_data[2], preload=True)
# raw_stress = mne.io.read_raw_fif(stress_data[2], preload=True)

raw_asr = mne.io.read_raw_fif(data[4], preload=True)
raw_first_rest = mne.io.read_raw_fif(data[5], preload=True)
raw_stress = mne.io.read_raw_fif(data[6], preload=True)
events, event_ids = mne.events_from_annotations(raw_asr)

raw_asr.crop(tmax = (events[-2,0]-events[0,0])/1000)

mne.concatenate_raws([raw_first_rest,raw_stress])

raw_asr.plot()
raw_first_rest.plot()

asr_data = raw_asr.get_data(start = 0, stop = events[-2,0]-events[0,0])
asr_join = raw_first_rest.get_data(start = 0, stop = events[-2,0]-events[0,0])
_,times = raw_asr.get_data(start = 0, stop = events[-2,0]-events[0,0], return_times = True)

t_stat, pvalue = ttest_ind(asr_data, asr_join, axis = 1)

# ch_name = raw_asr.ch_names
# diff = asr_data - asr_join
# time = times
# 
# for j in range(len(ch_name)):
#     a = ch_name[j]

#     plt.figure(figsize= (10,4))
#     plt.plot(time,asr_data[j] * 1e6, label = 'ASR')
#     plt.plot(time,asr_join[j]* 1e6, label = 'Joined ASR')
#     plt.plot(time,diff[j]* 1e6, label = 'HEP Diference')
#     for i in range(len(t_test.pvalue)):
#         if t_test.pvalue[j][i] <= 0.01:
#             plt.axvline(x = i/1000, ymin = 0 , ymax = 1, alpha = 0.3)
#     plt.axvline(0, linestyle = '--', c = 'black')
#     plt.axhline(0, linestyle = '--', c = 'black')
#     plt.ylabel('μV')
#     plt.xlabel('Time (s)')
#     plt.legend()
#     plt.title('ASR Comparison '+ a)
#     plt.show()