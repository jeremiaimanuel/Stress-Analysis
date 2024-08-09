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

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

##################### Define What I need in here #####################
    
include_second_rest = False
segmented = False

only_twave = False #IF True, better make n_segment = 2, if Flase, n_segment = 8
stats = 'all'
if only_twave:
    n_segment = 2
else:
    n_segment = 8

##################### Define What I need in here #####################

fpath = 'epoch_data'

rest1_data = []
for i in os.listdir(fpath):
    if i.endswith('epoch_first_rest-epo.fif'):
        rest1_data.append(os.path.join(fpath,i))
        
stress_data = []
for i in os.listdir(fpath):
    if i.endswith('epoch_stress-epo.fif'):
        stress_data.append(os.path.join(fpath,i))
        
rest2_data = []
for i in os.listdir(fpath):
    if i.endswith('epoch_second_rest-epo.fif'):
        rest2_data.append(os.path.join(fpath,i))

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

print(file_rest1)
fnum = int(input("Choose Data: "))
epoch_rest = load_epoch_rest1(fnum)
epoch_stress = load_epoch_stress(fnum)
epoch_rest2 = load_epoch_rest2(fnum)

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

# ###############################################################################

# clf = LinearSVC(dual='auto')
# gkf = KFold(5)
# pipe = Pipeline([('scaler',StandardScaler()),('clf',clf)])
# param_grid={'clf__C':[0.25,0.5,0.75, 1]}
# # param_grid={}
# # gscv = GridSearchCV(pipe, param_grid,cv = gkf,n_jobs = 4)

# gscv = GridSearchCV(pipe, param_grid)
# gscv.fit(X, y)
# # gkf.get_n_splits(feature,label)
# scores = cross_val_score(gscv, X, y, cv=gkf)
# print(scores)

# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# y_pred = gscv.predict(X)
# print(classification_report(y,y_pred))

# ###############################################################################

# n_features = 5
# X_5 = df.sample(n_features, axis = 1, random_state=99)

# df_5features = labels.join(X_5)

# clf = LinearSVC(dual='auto')
# gkf = KFold(5)
# pipe = Pipeline([('scaler',StandardScaler()),('clf',clf)])
# # param_grid={'clf__C':[0.25,0.5,0.75, 1]}
# param_grid={}
# # gscv = GridSearchCV(pipe, param_grid,cv = gkf,n_jobs = 4)

# gscv = GridSearchCV(pipe, param_grid, refit=True)
# gscv.fit(X_5, y)
# # gkf.get_n_splits(feature,label)
# scores = cross_val_score(gscv, X_5, y, cv=gkf)
# print(scores)

# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# y_pred = gscv.predict(X_5)
# print(classification_report(y,y_pred))

# ###############################################################################

# clf = LinearDiscriminantAnalysis()
# gkf = KFold(5)
# pipe = Pipeline([('scaler',StandardScaler()),('clf',clf)])
# param_grid={}
# gscv = GridSearchCV(pipe, param_grid, refit=True)
# pipe.fit(X, y)

# scores = cross_val_score(pipe, X, y, cv=gkf)
# print(scores)

# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

# y_pred = pipe.predict(X)
# print(classification_report(y,y_pred))

# ###############################################################################

