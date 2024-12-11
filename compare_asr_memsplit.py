# -*- coding: utf-8 -*-
"""
Created on Tue Dec 10 06:58:15 2024

@author: Jeremi
"""

from scipy.stats import pearsonr
import mne
import os

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)


path1 = "filtered_data_asr/20231019_B68_stroop5mins_0001-asr.fif"
path2 = "test_compare_asr5/20231019_B68_stroop5mins_0001-asr.fif"
path3 = "test_compare_asr5/20231019_B68_stroop5mins_0001-asr-memsplits45.fif"

data_1 = mne.io.read_raw_fif(path1)
data_2 = mne.io.read_raw_fif(path2)
data_3 = mne.io.read_raw_fif(path3)

data_arr_1= data_1.get_data()
data_arr_2 = data_2.get_data()
data_arr_3 = data_3.get_data()

corr = [pearsonr(data_arr_2[i],data_arr_3[i]) for i in range(len(data_arr_2))]
corr = [pearsonr(data_arr_1[i],data_arr_2[i]) for i in range(len(data_arr_2))]
corr = pearsonr(data_arr_1,data_arr_2)

arr_corr = [i[0] for i in corr]