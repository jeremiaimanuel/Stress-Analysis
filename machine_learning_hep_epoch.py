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
segmented = True

only_twave = True #IF True, better make n_segment = 2, if Flase, n_segment = 8
stats = 'all' #option is avg, std, all
filt_30 = True
if only_twave:
    n_segment = 4
else:
    n_segment = 8

##################### Define What I need in here #####################

# fpath = 'epoch_data'
fpath = 'hep_asr'
# fpath = 'hep_ica_all'

rest1_data = [os.path.join(fpath,i) for i in os.listdir(fpath) if i.endswith('first-epo.fif')]
stress_data = [os.path.join(fpath,i) for i in os.listdir(fpath) if i.endswith('stress-epo.fif')]
rest2_data = [os.path.join(fpath,i) for i in os.listdir(fpath) if i.endswith('second-epo.fif')]

file_rest1 = {number: rest1_data[number] for number in range(len(rest1_data))}
file_stress = {number: stress_data[number] for number in range(len(stress_data))}
file_rest2 = {number: rest2_data[number] for number in range(len(rest2_data))}

if filt_30 == True:
    def load_epoch_rest1(fdata):
        epoch = mne.read_epochs(rest1_data[fdata], preload=True).filter(1,30)
        return(epoch)
    def load_epoch_stress(fdata):
        epoch = mne.read_epochs(stress_data[fdata], preload=True).filter(1,30)
        return(epoch)
    def load_epoch_rest2(fdata):
        epoch = mne.read_epochs(rest2_data[fdata], preload=True).filter(1,30)
        return(epoch)
else:
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
display(df)

# n_features = 5
# # X_5 = df.sample(n_features, axis = 1, random_state=99)
# X_5 = df[['status','std.ch.F3','std.ch.F1','std.ch.Fz','std.ch.F2','std.ch.F6']] #For data num 0
# df_5features = labels_str.join(X_5)

X = df.filter(like='ch')
y = np.ravel(df.filter(like='label'))
# y = df.filter(like='label')

###############################################################################

# from scipy.stats import ttest_ind

# t_test_data = ttest_ind(feature_rest,feature_stress)
# print(t_test_data.pvalue)

# p_value = t_test_data.pvalue

# p_value_data = np.vstack(np.array(([ch_names, np.squeeze(p_value)]))).T

###############################################################################
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV, StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict,permutation_test_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import umap.umap_ as mp
import seaborn as sns
###############################################################################

# %matplotlib inline

# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

# # sns.pairplot(df_5features, hue = 'label', palette= "tab10")
# sns.pairplot(df_5features, hue = 'status', palette= "tab10")

# ###############################################################################

# reducer = mp.UMAP()
# # reducer = mp.UMAP(n_neighbors=15, min_dist=0.1)

# scaled_data = StandardScaler().fit_transform(X)

# embedding = reducer.fit_transform(scaled_data)
# embedding.shape

# unique_label = df.status.unique()

# fig, ax = plt.subplots()
# scatter = ax.scatter(
#     embedding[:, 0],
#     embedding[:, 1],
#     c=[sns.color_palette()[x] for x in df.label])

# # Create a custom legend with unique colors and labels
# legend_handles = []
# for i, label in enumerate(unique_label):
#     color = sns.color_palette()[i]
#     handle = plt.Line2D([0], [0], marker='o', color='w', label=label,
#                         markerfacecolor=color, markersize=10)
#     legend_handles.append(handle)

# legend = ax.legend(handles=legend_handles, loc="lower left", title="Classes")
# ax.add_artist(legend)

# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the Dataset', fontsize=24)

################################ CLASSIFICATION 1 ################################

clf = LinearDiscriminantAnalysis()
scl = StandardScaler()
n_splits = 5
# gkf = KFold(n_splits = n_splits)
skf = StratifiedKFold(n_splits = n_splits)
# tscv = TimeSeriesSplit(n_splits = n_splits)
pipe = Pipeline([('scaler',StandardScaler()),('clf',clf)])
pipe.fit(X,y)

acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
print(acc_scores)

print("%0.2f LDA accuracy with a standard deviation of %0.2f" % (acc_scores.mean(), acc_scores.std()))

roc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')
print(roc_scores)

print("LDA ROC AUC Score: ", roc_scores.mean())

y_pred = cross_val_predict(pipe, X,y, cv=skf)
print(classification_report(y,y_pred))

y_permut = permutation_test_score(pipe, X,y, cv=skf, random_state=99)
print(y_permut)

#######Plot LDA on each Fold#######
# fold_idx = 1
# for train_idx, test_idx in skf.split(X, y):
#     print(len(train_idx), len(test_idx))
    
#     # Use .iloc[] for DataFrame and [] for numpy arrays
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # X is a pandas DataFrame
#     y_train, y_test = y[train_idx], y[test_idx]  # y is a numpy array
    
#     # Fit the pipeline on the training set
#     pipe.fit(X_train, y_train)
    
#     # Transform the data
#     X_train_lda = pipe.transform(X_train)
#     X_test_lda = pipe.transform(X_test)
    
#     # Plot LDA for this fold
#     plt.figure()
#     for i in np.unique(y_train):
#         plt.scatter(X_train_lda[y_train == i], np.zeros_like(X_train_lda[y_train == i]), alpha=.8, label=f'Class {i} (train)')
#     for i in np.unique(y_test):
#         plt.scatter(X_test_lda[y_test == i], np.zeros_like(X_test_lda[y_test == i])-0.05, alpha=.8, label=f'Class {i} (test)')
    
#     plt.legend(loc='best', shadow=False)
#     plt.yticks([])  # No need for y-axis ticks in 1D plot
#     plt.title(f'LDA of dataset for fold {fold_idx}')
#     plt.xlabel('Linear Discriminant')
#     plt.show()

#     fold_idx += 1
#######Plot LDA on each Fold#######

################################ CLASSIFICATION 1 ################################

################################ CLASSIFICATION 2 ################################
from sklearn.svm import SVC

clf = SVC(kernel='linear')
n_splits = 5
scl = StandardScaler()
gkf = KFold(n_splits = n_splits)
skf = StratifiedKFold(n_splits = n_splits)
umap = mp.UMAP(random_state=99)
pipe = Pipeline([('scl',scl),('umap', umap),('clf',clf)])
# pipe = Pipeline([('scl',StandardScaler()),('clf',clf)])
# param_grid={'clf__C':[0.25,0.5,0.75, 1]}
# gscv = GridSearchCV(pipe, param_grid)
# gscv.fit(X, y)
pipe.fit(X, y)

acc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='accuracy')
print(acc_scores)

print("%0.2f SVM accuracy with a standard deviation of %0.2f" % (acc_scores.mean(), acc_scores.std()))

roc_scores = cross_val_score(pipe, X, y, cv=skf, scoring='roc_auc')
print(roc_scores)

print("SVM ROC AUC Score: ", roc_scores.mean())

y_pred = cross_val_predict(pipe, X,y, cv=skf)
print(classification_report(y,y_pred))

y_permut = permutation_test_score(pipe, X,y, cv=skf, random_state=99)
print(y_permut)

# cm = confusion_matrix(y,y_pred)
# print(cm)

#######Plot SVM#######
# from matplotlib.colors import ListedColormap

# fold_idx = 1
# for train_idx, test_idx in skf.split(X, y):
#     print(len(train_idx), len(test_idx))
#     # Use .iloc[] for DataFrame and [] for numpy arrays
#     X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # X is a pandas DataFrame
#     y_train, y_test = y[train_idx], y[test_idx]  # y is a numpy array
    
#     #### Scaled and Classified 1 by 1 ####
#     X_train_scaled = scl.fit_transform(X_train,y_train)
#     X_test_scaled = scl.transform(X_test)
    
#     X_train_reduced = umap.fit_transform(X_train_scaled,y_train)
#     X_test_reduced = umap.transform(X_test_scaled)
    
#     clf.fit(X_train_reduced, y_train)

#     x_min, x_max = X_test_reduced[:, 0].min() - 1, X_test_reduced[:, 0].max() + 1
#     y_min, y_max = X_test_reduced[:, 1].min() - 1, X_test_reduced[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
#                           np.arange(y_min, y_max, 0.02))

#     grid = np.c_[xx.ravel(), yy.ravel()]
#     Z = np.reshape(clf.predict(grid), xx.shape)

#     plt.figure()
#     plt.contourf(xx, yy, Z, alpha=0.3,cmap = ListedColormap(('green','red')))
#     plt.xlim(xx.min(),xx.max())
#     plt.ylim(yy.min(),yy.max())

#     for i,j in enumerate(np.unique(y_train)):
#         if j == 0:
#             label = 'Rest_train'
#         else:
#             label = 'Stress_train'
#         plt.scatter(X_train_reduced[y_train ==j,0],X_train_reduced[y_train==j,1],
#                     c=ListedColormap(('limegreen','lightcoral'))(i),label=label)
    
#     for i,j in enumerate(np.unique(y_test)):
#         if j == 0:
#             label = 'Rest_test'
#         else:
#             label = 'Stress_test'
#         plt.scatter(X_test_reduced[y_test ==j,0],X_test_reduced[y_test==j,1],
#                     c=ListedColormap(('green','red'))(i),label=label)
    
#     plt.legend()
#     plt.title(f'Decision Boundary for Linear SVC fold {fold_idx}')
#     plt.show()

#     fold_idx+=1
#######Plot SVM#######

################################ CLASSIFICATION 2 ################################





