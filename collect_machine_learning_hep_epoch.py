# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 17:46:50 2024

@author: Jeremi
"""

import mne
import os
import pandas as pd
import numpy as np

############################## IMPORT ML LIBRARY ##############################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap.umap_ as mp
############################## IMPORT ML LIBRARY ##############################
###### Load Data ######

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

##################### Define What I need in here #####################
    
include_second_rest = False
segmented = True

only_twave = True #IF True, better make n_segment = 4, if False, n_segment = 8
stats = 'all'
if only_twave:
    n_segment = 4
else:
    n_segment = 8

##################### Define What I need in here #####################

# fpath = 'hep_asr'
fpath = 'hep_ica_all'

rest1_data = [os.path.join(fpath,i) for i in os.listdir(fpath) if i.endswith('first-epo.fif')]
stress_data = [os.path.join(fpath,i) for i in os.listdir(fpath) if i.endswith('stress-epo.fif')]
rest2_data = [os.path.join(fpath,i) for i in os.listdir(fpath) if i.endswith('second-epo.fif')]

file_rest1 = {number: rest1_data[number] for number in range(len(rest1_data))}
file_stress = {number: stress_data[number] for number in range(len(stress_data))}
file_rest2 = {number: rest2_data[number] for number in range(len(rest2_data))}

def load_epoch_rest1(fdata):
    epoch = mne.read_epochs(rest1_data[fdata], preload=True)
    return(epoch)
def load_epoch_stress(fdata):
    epoch = mne.read_epochs(stress_data[fdata], preload=True)
    return(epoch)
def load_epoch_rest2(fdata):
    epoch = mne.read_epochs(rest2_data[fdata], preload=True)
    return(epoch)

# print(file_rest1)
# fnum = int(input("Choose Data: "))

lda_accuracy =[]
lda_roc=[]
svm_accuracy=[]
svm_roc=[]
# dataset_used1 = []
# dataset_used2 = []

for fnum in range(len(rest1_data)):
        
    epoch_rest = load_epoch_rest1(fnum)
    epoch_stress = load_epoch_stress(fnum)
    epoch_rest2 = load_epoch_rest2(fnum)
    
    if only_twave == True:
        epoch_rest = epoch_rest.crop(tmin = 0.2, tmax = 0.6)
        epoch_stress = epoch_stress.crop(tmin = 0.2, tmax = 0.6)
        epoch_rest2 = epoch_rest2.crop(tmin = 0.2, tmax = 0.6)
    
    array_rest = epoch_rest.get_data()
    array_stress = epoch_stress.get_data()
    array_rest2 = epoch_rest2.get_data()
    
    def feature_extract(epoch_array, n_segment, segmented = True, stats ='all'):
        
        n_epoch = epoch_array.shape[0]
        n_ch = epoch_array.shape[1]
        n_signal = epoch_array.shape[2]
        
        segment_length = n_signal // n_segment
        
        if segmented:    
            segment_array_avg = np.zeros((n_ch, n_segment *n_epoch +1))
            segment_array_std = np.zeros((n_ch, n_segment *n_epoch +1))
            
            for i in range(n_epoch):
                data_epoch = epoch_array[i]
                for j in range(len(data_epoch)):
                    segment_index = 0
                    for k in range(0,n_signal,segment_length):
                        segment = data_epoch[j, k:k+segment_length]
                        avg_segment = np.average(segment)
                        std_segment = np.std(segment)
                        
                        segment_array_avg[j,segment_index + (n_segment*i)] = avg_segment
                        segment_array_std[j,segment_index + (n_segment*i)] = std_segment
                        segment_index +=1
            
            segment_array_avg = np.array(segment_array_avg).T
            segment_array_std = np.array(segment_array_std).T
            segment_feature = np.concatenate((segment_array_avg,segment_array_std), axis=1)
            
            if stats == 'all':        
                return(segment_feature)
                
            elif stats == 'avg':
                return(segment_array_avg)
                
            elif stats == 'std':
                return(segment_array_std)
        
        else:
            avg_feature = np.mean(epoch_array, axis=-1)
            std_feature = np.std(epoch_array, axis = -1)
            feature = np.concatenate((avg_feature, std_feature), axis=1)
            
            if stats == 'all':
                return(feature)
            
            elif stats == 'avg':
                return(avg_feature)
            
            elif stats == 'std':
                return(std_feature)
    
    def ch_name_extract(epoch_array, stats ='all'):
        
        eeg_ch_names = epoch_array.ch_names
        ch_names = []
        
        if stats =='all':
            prefixes = ["average.ch.", "std.ch."]
      
        elif stats =='avg':
            prefixes = ["average.ch."]
        
        elif stats =='std':
            prefixes = ["std.ch."]
    
        for i in prefixes:
            for j in eeg_ch_names:
                name = i+j
                ch_names.append(name)
        
        return(ch_names)
    
    feature_rest = feature_extract(array_rest, n_segment, segmented, stats)
    feature_stress = feature_extract(array_stress, n_segment, segmented, stats)
    feature_rest2 = feature_extract(array_rest2, n_segment, segmented, stats)
    
    label_rest = len(feature_rest) * [0]
    label_stress = len(feature_stress) * [1]
    label_rest2 = len(feature_rest2) * [2]
    
    label_str_rest = len(feature_rest) * ['Rest 1']
    label_str_stress = len(feature_stress) * ['Stress']
    label_str_rest2 = len(feature_rest2) * ['Rest 2']
    
    if include_second_rest == True:
        feature = np.concatenate((feature_rest, feature_stress, feature_rest2))
        label = np.concatenate((label_rest, label_stress, label_rest2))
        label_str = np.concatenate((label_str_rest, label_str_stress, label_str_rest2))
    else:
        feature = np.concatenate((feature_rest, feature_stress))
        label = np.concatenate((label_rest, label_stress))
        label_str = np.concatenate((label_str_rest, label_str_stress))    
    
    ch_names = ch_name_extract(epoch_rest, stats)
    
    features = pd.DataFrame(feature, columns = ch_names)
    labels = pd.DataFrame(label, columns= ['label'])
    labels_str = pd.DataFrame(label_str, columns=['status'])
    
    df_labels = labels.join(labels_str)
    df = df_labels.join(features)
    
    X = df.filter(like='ch')
    y = np.ravel(df.filter(like='label'))
    
    ################################ CLASSIFICATION 1 ################################
    
    clf = LinearDiscriminantAnalysis()
    scl = StandardScaler()
    n_splits = 5
    # gkf = KFold(n_splits = n_splits)
    skf = StratifiedKFold(n_splits = n_splits)
    # umap = mp.UMAP(random_state=99)
    # pipe = Pipeline([('scl',scl),('umap', umap),('clf',clf)])
    # tscv = TimeSeriesSplit(n_splits = n_splits)
    pipe = Pipeline([('scaler',StandardScaler()),('clf',clf)])
    pipe.fit_transform(X,y)
    
    acc_scores_lda = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    # print(acc_scores_lda)
    lda_accuracy.append(acc_scores_lda.mean())
    # print("%0.2f LDA accuracy with a standard deviation of %0.2f from dataset %d" % (acc_scores_lda.mean(), acc_scores_lda.std(),fnum))
    
    roc_scores_lda = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')
    # print(roc_scores_lda)
    lda_roc.append(roc_scores_lda.mean())
    # print("LDA ROC AUC Score:%0.2f from dataset %d" % (roc_scores_lda.mean(),fnum))
    
    ################################ CLASSIFICATION 1 ################################
    
    # ################################ CLASSIFICATION 2 ################################
    
    # clf = SVC(kernel='linear')
    # n_splits = 5
    # scl = StandardScaler()
    # skf = StratifiedKFold(n_splits = n_splits)
    # umap = mp.UMAP(random_state=99)
    # pipe = Pipeline([('scl',scl),('umap', umap),('clf',clf)])
    # # pipe = Pipeline([('scl',StandardScaler()),('clf',clf)])
    # # param_grid={'clf__C':[0.25,0.5,0.75, 1]}
    # # gscv = GridSearchCV(pipe, param_grid)
    # # gscv.fit(X, y)
    # pipe.fit(X, y)
    
    # acc_scores_svm = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
    # # print(acc_scores_svm)
    # svm_accuracy.append(acc_scores_svm.mean())
    # # print("%0.2f SVM accuracy with a standard deviation of %0.2f from dataset %d" % (acc_scores_svm.mean(), acc_scores_svm.std(),fnum))
    
    # roc_scores_svm = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')
    # # print(roc_scores_svm)
    # svm_roc.append(roc_scores_svm.mean())
    # # print("SVM ROC AUC Score:%0.2f from dataset %d" % (roc_scores_svm.mean(),fnum))
        
    # ################################ CLASSIFICATION 2 ################################
    
    # dataset_used1.append(epoch_rest)
    # dataset_used2.append(epoch_stress)