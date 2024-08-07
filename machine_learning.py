# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 15:31:47 2024

@author: Jeremi
"""

import mne
import numpy as np
import os
import scipy
from scipy import signal as sg
from matplotlib import pyplot as plt
import pandas as pd

###### Load Data ######

directory_path = "D:/EEG RESEARCH DATA/epoch_data"
os.chdir(directory_path)

##################### Define What I need in here #####################
    
include_second_rest = True
segmented = True

only_twave = True #IF True, better make n_segment = 2, if Flase, n_segment = 8
stats = 'all'
if only_twave:
    n_segment = 2
else:
    n_segment = 8

##################### Define What I need in here #####################

epoch_rest = mne.read_epochs('20240418_B98_jikken_0004-epoch_first_rest-epo.fif')
epoch_stress = mne.read_epochs('20240418_B98_jikken_0004-epoch_stress-epo.fif')
epoch_rest2 = mne.read_epochs('20240418_B98_jikken_0004-epoch_second_rest-epo.fif')

if only_twave == True:
    epoch_rest = epoch_rest.crop(tmin = 0.2, tmax = 0.4)
    epoch_stress = epoch_stress.crop(tmin = 0.2, tmax = 0.4)
    epoch_rest2 = epoch_rest2.crop(tmin = 0.2, tmax = 0.4)

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
        feature = np.concatenate((avg_feature, std_feature))
        
        if stats == 'all':
            return(feature)
        
        elif stats == 'avg':
            return(avg_feature)
        
        elif stats == 'std':
            return(feature)
    

    
    eeg_ch_names = epoch_array.ch_names
    ch_names = []
    
    if stats =='all':
        prefixes = ["average.ch.", "std.ch"]
  
    elif stats =='avg':
        prefixes = ["average.ch."]
    
    elif stats =='std':
        prefixes = ["std.ch"]

    for i in prefixes:
        for j in eeg_ch_names:
            name = i+j
            ch_names.append(name)
    
    return(ch_names)
    
    eeg_ch_names = epoch_array.ch_names
    ch_names = []
    
    if stats =='all':
        prefixes = ["average.ch.", "std.ch"]

        for i in prefixes:
            for j in eeg_ch_names:
                name = i+j
                ch_names.append(name)
    
    elif stats =='avg':
        prefixes = ["average.ch."]

        for i in prefixes:
            for j in eeg_ch_names:
                name = i+j
                ch_names.append(name)
    
    elif stats =='std':
        prefixes = ["std.ch"]

        for i in prefixes:
            for j in eeg_ch_names:
                name = i+j
                ch_names.append(name)
    
    return(ch_names)

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

if include_second_rest == True:
    feature = np.concatenate((feature_rest, feature_stress, feature_rest2))
    label = np.concatenate((label_rest, label_stress, label_rest2))
else:
    feature = np.concatenate((feature_rest, feature_stress))
    label = np.concatenate((label_rest, label_stress))    

ch_names = ch_name_extract(epoch_rest, stats)

features = pd.DataFrame(feature, columns = ch_names)
labels = pd.DataFrame(label, columns= ['label'])

df = labels.join(features)
display(df)

n_features = 5
X_5 = df.sample(n_features, axis = 1, random_state=99)
df_5features = labels.join(X_5)

X = df.filter(like='ch')
y = np.ravel(df.filter(like='label'))

###############################################################################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
###############################################################################

import seaborn as sns
%matplotlib inline

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

sns.pairplot(df_5features, hue = 'label', palette= "tab10")

###############################################################################

import umap.umap_ as mp
import seaborn as sns

reducer = mp.UMAP()

# clf = LinearDiscriminantAnalysis()
# pipe = Pipeline([('scaler',StandardScaler()),('clf',clf)])

scaled_data = StandardScaler().fit_transform(X)
# scaled_data = pipe.fit_transform(X_5,y)

embedding = reducer.fit_transform(scaled_data)
embedding.shape

unique_label = df.label.unique()

fig, ax = plt.subplots()
scatter = ax.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=[sns.color_palette()[x] for x in df.label])

# Create a custom legend with unique colors and labels
legend_handles = []
for i, label in enumerate(unique_label):
    color = sns.color_palette()[i]
    handle = plt.Line2D([0], [0], marker='o', color='w', label=label,
                        markerfacecolor=color, markersize=10)
    legend_handles.append(handle)

legend = ax.legend(handles=legend_handles, loc="lower left", title="Classes")
ax.add_artist(legend)

plt.gca().set_aspect('equal', 'datalim')
plt.title('UMAP projection of the Dataset', fontsize=24)

###############################################################################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
###############################################################################

clf = LinearDiscriminantAnalysis()
gkf = KFold(5)
pipe = Pipeline([('scaler',StandardScaler()),('clf',clf)])
param_grid={}
gscv = GridSearchCV(pipe, param_grid, refit=True)
gscv.fit(X, y)

scores = cross_val_score(gscv, X, y, cv=gkf)
print(scores)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

y_pred = gscv.predict(X)
print(classification_report(y,y_pred))

###############################################################################

clf = LinearSVC(dual='auto')
gkf = KFold(5)
pipe = Pipeline([('scaler',StandardScaler()),('clf',clf)])
param_grid={'clf__C':[0.25,0.5,0.75, 1]}
# param_grid={}
# gscv = GridSearchCV(pipe, param_grid,cv = gkf,n_jobs = 4)

gscv = GridSearchCV(pipe, param_grid)
gscv.fit(X, y)
# gkf.get_n_splits(feature,label)
scores = cross_val_score(gscv, X, y, cv=gkf)
print(scores)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

y_pred = gscv.predict(X)
print(classification_report(y,y_pred))

###############################################################################

n_features = 5
X_5 = df.sample(n_features, axis = 1, random_state=99)

df_5features = labels.join(X_5)

clf = LinearSVC(dual='auto')
gkf = KFold(5)
pipe = Pipeline([('scaler',StandardScaler()),('clf',clf)])
# param_grid={'clf__C':[0.25,0.5,0.75, 1]}
param_grid={}
# gscv = GridSearchCV(pipe, param_grid,cv = gkf,n_jobs = 4)

gscv = GridSearchCV(pipe, param_grid, refit=True)
gscv.fit(X_5, y)
# gkf.get_n_splits(feature,label)
scores = cross_val_score(gscv, X_5, y, cv=gkf)
print(scores)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

y_pred = gscv.predict(X_5)
print(classification_report(y,y_pred))

###############################################################################

clf = LinearDiscriminantAnalysis()
gkf = KFold(5)
pipe = Pipeline([('scaler',StandardScaler()),('clf',clf)])
param_grid={}
gscv = GridSearchCV(pipe, param_grid, refit=True)
pipe.fit(X, y)

scores = cross_val_score(pipe, X, y, cv=gkf)
print(scores)

print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

y_pred = pipe.predict(X)
print(classification_report(y,y_pred))

###############################################################################

# ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

# raw = mne.io.read_raw_brainvision("20240418_mat5mins/20240418_B98_jikken_0003.vhdr")

# # Reconstruct the original events from our Raw object
# events, event_ids = mne.events_from_annotations(raw)

# montage = mne.channels.make_standard_montage('standard_1020')
# raw.set_channel_types({'ECG':'ecg'})
# raw.set_channel_types({'vEOG':'eog'})
# raw.set_channel_types({'hEOG':'eog'})

# raw.set_montage(montage)
# #fig = raw.plot_sensors(show_names=True)


# ###### Segment Grouping ######

# fs = 1000

# trg0 = events[3,0] #Experiment Begin 
# trg1 = events[4,0] #Task Begin
# trg2 = events[-2,0] #Task End
# trg3 = events[-1,0] #Experiment End

# tmin = trg0/fs
# tmax = trg3/fs

# #Segment in Samples
# eeg_seg1 = trg1 - trg0
# eeg_seg2 = trg2 - trg0
# eeg_seg3 = trg3 - trg0

# #Segment in ms
# eeg_newseg1 = int(eeg_seg1/fs)
# eeg_newseg2 = int(eeg_seg2/fs)
# eeg_newseg3 = int(eeg_seg3/fs)

# ###### R-Wave Detection ######

# raw_ecg = raw.copy().pick_types(eeg=False, eog=False, ecg=True).crop(tmin = tmin, tmax = tmax) #make a copy

# mne_ecg, mne_time = raw_ecg[:]
# mne_ecg = np.squeeze(-mne_ecg)

# b, a = sg.butter(2, [0.5, 150], 'bandpass', output= 'ba', fs=fs)
# filtered = sg.filtfilt(b,a,mne_ecg)

# mne_ecg = filtered.copy()

# QRS_detector = pan_tompkins_qrs()
# output = QRS_detector.solve(mne_ecg, fs)

# # Convert ecg signal to numpy array
# signal = mne_ecg.copy()

# # Find the R peak locations
# hr = heart_rate(signal,fs)
# result = hr.find_r_peaks()
# result = np.array(result)

# # Clip the x locations less than 0 (Learning Phase)
# result = result[result > 0]

# ######Pre Process, Data Cleaning ECG######

# result = result[result>=237]


# ###### Making Pan Tompkins Events ######
# r_peak_onset = []
# for i in range(len(result)):
#     ons_idx = int(fs*tmin)+result[i]
#     r_peak_onset.append(ons_idx)

# pan_tompkins_events = np.zeros((len(r_peak_onset), 3), dtype=int)

# pan_tompkins_events[:, 0] = r_peak_onset
# pan_tompkins_events[:, 1] = 0 
# pan_tompkins_events[:, 2] = 7

# ###### EEG ICA ######

# raw_temp = raw.copy().crop(tmin = tmin, tmax = tmax) #make a copy
# raw_temp.load_data()

# ica = read_ica("20240418_mat5mins/20240418_B98_jikken_0003-ica.fif")
# eog_indices, eog_scores = ica.find_bads_eog(raw_temp, ch_name=['vEOG', 'hEOG'], threshold = 0.8, measure='correlation')

# ica.exclude = eog_indices

# reconst_raw = raw_temp.copy()
# ica.apply(reconst_raw)

# ###### ERP: HEP ######

# new_filt_raw = reconst_raw.load_data().copy().filter(l_freq=1.0, h_freq=40, picks = ['eeg'])

# new_filt_firstrest = new_filt_raw.copy().crop(tmin = 0, tmax = eeg_newseg1)
# new_filt_stress = new_filt_raw.copy().crop(tmin = eeg_newseg1, tmax = eeg_newseg2)
# new_filt_secondrest = new_filt_raw.copy().crop(tmin = eeg_newseg2)

# epoch_tmin = -0.2
# epoch_tmax = 0.6
# baseline = (None,0)

# first_rest_epoch = mne.Epochs(new_filt_firstrest,events = pan_tompkins_events, tmin= epoch_tmin, tmax = epoch_tmax, baseline = baseline, picks = ['eeg'])
# stress_epoch = mne.Epochs(new_filt_stress,events = pan_tompkins_events, tmin= epoch_tmin, tmax = epoch_tmax, baseline = baseline, picks = ['eeg'])
# second_rest_epoch = mne.Epochs(new_filt_secondrest,events = pan_tompkins_events, tmin= epoch_tmin, tmax = epoch_tmax, baseline = baseline, picks = ['eeg'])