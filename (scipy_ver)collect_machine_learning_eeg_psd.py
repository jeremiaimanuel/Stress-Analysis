# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 15:30:03 2024

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

############################## IMPORT ML LIBRARY ##############################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap.umap_ as mp
import seaborn as sns
############################## IMPORT ML LIBRARY ##############################
#################################### DEFINE ###################################
second_rest = False
#################################### DEFINE ###################################
########
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

    segment_length = (t_seg*fs)+1  # seconds
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
########
########
def load_fif_file(fdata):
    raw = mne.io.read_raw_fif(files[fdata], preload=True)
    return (raw)
########
################################## LOAD FILE ##################################

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

# fpath = 'filtered_data'
# fpath = 'filtered_data_ica_all'
fpath = 'filtered_data_asr'

filtered_data = [os.path.join(fpath,i) for i in os.listdir(fpath)]
files = {number: filtered_data[number] for number in range(len(filtered_data))}

lda_accuracy =[]
lda_roc=[]
svm_accuracy=[]
svm_roc=[]

# for file_number in range(17,22):
for file_number in range(len(files)):
    
    raw = load_fif_file(file_number)
    raw.load_data()
    
    events, event_ids = mne.events_from_annotations(raw)
    
    fs = 1000
    
    trg0 = events[0,0] #Experiment Begin 
    trg1 = events[1,0] #Task Begin
    trg2 = events[-2,0] #Task End
    trg3 = events[-1,0] #Experiment End

    if 'B98_jikken_0001' in files[file_number]:
        trg2 = events[-1, 0]
        trg3 = trg0 + 900000
    elif any(keyword in files[file_number] for keyword in ['B83', 'B74', 'B94']):
        trg2 = events[-3, 0]  # Task End
    
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
    
    ################################## LOAD FILE ##################################
    
    ############################# FEATURE EXTRACTION #############################
        
    eeg_data = raw.pick_types(eeg=True, eog=False, ecg=False).get_data()
    eeg_ch_names = raw.pick_types(eeg=True, eog=False, ecg=False).ch_names
    
    prefixes = ["ch.theta.", "ch.alpha.", "ch.beta.", "ch.gamma."]
    
    new_eeg_ch_names = []
    for i in prefixes:
        for j in eeg_ch_names:
            name = i+j
            new_eeg_ch_names.append(name)
            
    theta_data_rest = abs_power_extraction(eeg_data, fs, 4, 8, 0, eeg_newseg1)
    theta_data_stress = abs_power_extraction(eeg_data, fs, 4, 8, eeg_newseg1, eeg_newseg2)
    theta_data_rest2 = abs_power_extraction(eeg_data, fs, 4, 8, eeg_newseg2, eeg_newseg3)
    
    alpha_data_rest = abs_power_extraction(eeg_data, fs, 8, 12, 0, eeg_newseg1)
    alpha_data_stress = abs_power_extraction(eeg_data, fs, 8, 12, eeg_newseg1, eeg_newseg2)
    alpha_data_rest2 = abs_power_extraction(eeg_data, fs, 8, 12, eeg_newseg2, eeg_newseg3)
    
    beta_data_rest = abs_power_extraction(eeg_data, fs, 12, 30, 0, eeg_newseg1)
    beta_data_stress = abs_power_extraction(eeg_data, fs, 12, 30, eeg_newseg1, eeg_newseg2)
    beta_data_rest2 = abs_power_extraction(eeg_data, fs, 12, 30, eeg_newseg2, eeg_newseg3)
    
    gamma_data_rest = abs_power_extraction(eeg_data, fs, 30, 40, 0, eeg_newseg1)
    gamma_data_stress = abs_power_extraction(eeg_data, fs, 30, 40, eeg_newseg1, eeg_newseg2)
    gamma_data_rest2 = abs_power_extraction(eeg_data, fs, 30, 40, eeg_newseg2, eeg_newseg3)
    
    data_list_rest = np.concatenate((theta_data_rest,alpha_data_rest,beta_data_rest,gamma_data_rest))
    data_list_stress = np.concatenate((theta_data_stress,alpha_data_stress,beta_data_stress,gamma_data_stress))
    data_list_rest2 = np.concatenate((theta_data_rest2,alpha_data_rest2,beta_data_rest2,gamma_data_rest2))
    
    label_rest = len(data_list_rest[0]) * [0]
    label_stress = len(data_list_stress[0]) * [1]
    label_rest2 = len(data_list_rest2[0]) * [2]
    
    label_str_rest = len(data_list_rest[0]) * ['Rest 1']
    label_str_stress = len(data_list_stress[0]) * ['Stress']
    label_str_rest2 = len(data_list_rest2[0]) * ['Rest 2']
    
    if second_rest:
        feature = np.concatenate((data_list_rest,data_list_stress, data_list_rest2), axis = 1)
        label = np.concatenate((label_rest, label_stress, label_rest2))
        label_str = np.concatenate((label_str_rest, label_str_stress, label_str_rest2))
        feature = feature.T
    else:
        feature = np.concatenate((data_list_rest,data_list_stress), axis = 1)
        label = np.concatenate((label_rest, label_stress))
        label_str = np.concatenate((label_str_rest, label_str_stress))    
    
        feature = feature.T
    
    ############################# DATA FRAME FEATURE #############################
    features = pd.DataFrame(feature, columns = new_eeg_ch_names)
    labels = pd.DataFrame(label, columns= ['label'])
    labels_str = pd.DataFrame(label_str, columns=['status'])
    
    df_labels = labels.join(labels_str)
    df = df_labels.join(features)
    # display(df)
    
    X = df.filter(like='ch')
    y = np.ravel(df.filter(like='label'))
    
    ############################# DATA FRAME FEATURE #############################
    
    ################################ CLASSIFICATION 1 ################################
    
    clf = LinearDiscriminantAnalysis()
    scl = StandardScaler()
    n_splits = 5
    # gkf = KFold(n_splits = n_splits)
    skf = StratifiedKFold(n_splits = n_splits)
    pipe = Pipeline([('scaler',scl),('clf',clf)])
    pipe.fit(X,y)
    
    acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    print(acc_scores)
    lda_accuracy.append(acc_scores.mean())
    print("%0.2f LDA accuracy with a standard deviation of %0.2f from dataset %d" % (acc_scores.mean(), acc_scores.std(),file_number))
    
    roc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')
    print(roc_scores)
    lda_roc.append(roc_scores.mean())
    print("LDA ROC AUC Score:%0.2f from dataset %d" % (roc_scores.mean(),file_number))
    
    ################################ CLASSIFICATION 1 ################################
    
    ################################ CLASSIFICATION 2 ################################
    from sklearn.svm import SVC
    
    clf = SVC(kernel='linear')
    n_splits = 5
    scl = StandardScaler()
    gkf = KFold(n_splits = n_splits)
    skf = StratifiedKFold(n_splits = n_splits)
    # umap = mp.UMAP(random_state=99)
    # pipe = Pipeline([('scl',scl),('umap', umap),('clf',clf)])
    pipe = Pipeline([('scl',scl),('clf',clf)])
    pipe.fit(X, y)
    
    acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    print(acc_scores)
    svm_accuracy.append(acc_scores.mean())
    print("%0.2f SVM accuracy with a standard deviation of %0.2f from dataset %d" % (acc_scores.mean(), acc_scores.std(),file_number))
    
    roc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')
    print(roc_scores)
    svm_roc.append(roc_scores.mean())
    print("SVM ROC AUC Score:%0.2f from dataset %d" % (roc_scores.mean(),file_number))
        
    ################################ CLASSIFICATION 2 ################################