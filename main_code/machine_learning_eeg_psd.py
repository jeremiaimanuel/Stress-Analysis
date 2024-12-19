# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 13:58:34 2024

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

#################################### DEFINE ###################################
second_rest = False
#################################### DEFINE ###################################

################################## LOAD FILE ##################################

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

# fpath = 'filtered_data'
# fpath = 'filtered_data_ica_all'
fpath = 'filtered_data_asr'

filtered_data = [os.path.join(fpath,i) for i in os.listdir(fpath)]

files = {number: filtered_data[number] for number in range(len(filtered_data))}

def load_fif_file(fdata):
    raw = mne.io.read_raw_fif(files[fdata], preload=True)
    return (raw)

print(files)
file_number = int(input("Choose File: "))
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
eeg_ch_names = raw.pick_types(eeg=True).ch_names
prefixes = ["ch.theta.", "ch.alpha.", "ch.beta.", "ch.gamma."]
new_eeg_ch_names = [i+j for i in prefixes for j in eeg_ch_names]

def welch_extraction_mne(raw, l_freq, h_freq, tmin, tmax, t_seg=10, t_overlap=9):
    """
    Extract Absolute PSD from Data

    Parameters:
    :param signal: Signal that want to be extracted from
    :param l_freq: Minimum range of EEG bands frequency (Theta = 4, Alpha = 8, Beta = 12, Gamma = 30)
    :param h_freq: Maxmimum range of EEG bands frequency (Theta = 8, Alpha = 12, Beta = 30, Gamma = 45)
    :param tmin: Start time where the frequency of the signal want to be extracted 
    :param tmax: End time where the frequency of the signal want to be extracted
    :param t_seg: Length of time that want to be segmented (in seconds)
    :param t_overlap: Length of Overlap time (in seconds)
    
    Returns:
    :return: Extracted absolute Power
    """
    
    signal = raw.copy().crop(tmin=tmin, tmax=tmax)
    
    psd_epochs = mne.make_fixed_length_epochs(signal, duration = t_seg, overlap = t_overlap)
    
    psd_results = psd_epochs.compute_psd(
        method='welch', 
        fmin=l_freq, 
        fmax=h_freq,
        window='hann',
        n_fft=int(len(psd_epochs.times)))
    
    abs_arr_psd = np.sum(np.abs(psd_results.get_data()), axis = 2) #Absolute PSD Calculated here
    
    return abs_arr_psd.T
        
theta_data_rest = welch_extraction_mne(raw, 4, 8, 0, eeg_newseg1)
theta_data_stress = welch_extraction_mne(raw, 4, 8, eeg_newseg1, eeg_newseg2)
theta_data_rest2 = welch_extraction_mne(raw, 4, 8, eeg_newseg2, eeg_newseg3)

alpha_data_rest = welch_extraction_mne(raw, 8, 12, 0, eeg_newseg1)
alpha_data_stress = welch_extraction_mne(raw, 8, 12, eeg_newseg1, eeg_newseg2)
alpha_data_rest2 = welch_extraction_mne(raw, 8, 12, eeg_newseg2, eeg_newseg3)

beta_data_rest = welch_extraction_mne(raw, 12, 30, 0, eeg_newseg1)
beta_data_stress = welch_extraction_mne(raw, 12, 30, eeg_newseg1, eeg_newseg2)
beta_data_rest2 = welch_extraction_mne(raw, 12, 30, eeg_newseg2, eeg_newseg3)

gamma_data_rest = welch_extraction_mne(raw, 30, 40, 0, eeg_newseg1)
gamma_data_stress = welch_extraction_mne(raw, 30, 40, eeg_newseg1, eeg_newseg2)
gamma_data_rest2 = welch_extraction_mne(raw, 30, 40, eeg_newseg2, eeg_newseg3)

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
    feature_arr = np.concatenate((data_list_rest,data_list_stress, data_list_rest2), axis = 1)
    label = np.concatenate((label_rest, label_stress, label_rest2))
    label_str = np.concatenate((label_str_rest, label_str_stress, label_str_rest2))
    feature_arr = feature_arr.T
else:
    feature_arr = np.concatenate((data_list_rest,data_list_stress), axis = 1)
    label = np.concatenate((label_rest, label_stress))
    label_str = np.concatenate((label_str_rest, label_str_stress))    
    feature_arr = feature_arr.T
###############################################################################

from scipy.stats import ttest_ind

t_test_data = ttest_ind(data_list_rest.T,data_list_stress.T)
print(t_test_data.pvalue)

p_value = t_test_data.pvalue

p_value_data = np.column_stack((new_eeg_ch_names, p_value))

for i in range(len(p_value)):
    if p_value[i] >= 0.05:
        print(i)

###############################################################################
############################# DATA FRAME FEATURE #############################
features = pd.DataFrame(feature_arr, columns = new_eeg_ch_names)
labels = pd.DataFrame(label, columns= ['label'])
labels_str = pd.DataFrame(label_str, columns=['status'])

df_labels = labels.join(labels_str)
df = df_labels.join(features)
display(df)

X = df.filter(like='ch')
y = np.ravel(df.filter(like='label'))

# X_theta = df.filter(like='ch.theta')
# X_alpha = df.filter(like='ch.alpha')
# X_beta = df.filter(like='ch.beta')
# X_gamma = df.filter(like='ch.gamma')

# df_theta = labels.join(X_theta)
# df_alpha = labels.join(X_alpha)
# df_beta = labels.join(X_beta)
# df_gamma = labels.join(X_gamma)

# n_features = 5
# X_5 = df.sample(n_features, axis = 1, random_state=99)
# df_5features = labels_str.join(X_5)
############################# DATA FRAME FEATURE #############################

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

# ############################## DATA VISUALIZATION ##############################

%matplotlib inline

sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})
# sns.set(style='white', context='notebook', rc={'figure.figsize':(54,50)})

# sns.pairplot(df_5features, hue = 'status',palette= "tab10")
# sns.pairplot(df_theta, hue = 'status',palette= "tab10")
# sns.pairplot(df_alpha, hue = 'status',palette= "tab10")
# sns.pairplot(df_beta, hue = 'status',palette= "tab10")
# sns.pairplot(df_gamma, hue = 'status',palette= "tab10")

df_tp10 = X.filter(like='ch.gamma.TP10')
df_tp10 = labels_str.join(df_tp10)
sns.pairplot(df_tp10, hue = 'status',palette= "tab10")

# reducer = mp.UMAP()
# scaled_data = StandardScaler().fit_transform(X)
# embedding = reducer.fit_transform(scaled_data)
# embedding.shape

# unique_label = df.status.unique()

# sns.set(style='white', context='notebook', rc={'figure.figsize':(14,10)})

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

# plt.gca().set_aspect('equal','datalim')
# plt.title('UMAP projection of the Dataset', fontsize=24)

# ############################## DATA VISUALIZATION ##############################

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

#######Plot LDA on each Fold#######
fold_idx = 1
for train_idx, test_idx in skf.split(X, y):
    print(len(train_idx), len(test_idx))
    
    # Use .iloc[] for DataFrame and [] for numpy arrays
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # X is a pandas DataFrame
    y_train, y_test = y[train_idx], y[test_idx]  # y is a numpy array
    
    # Fit the pipeline on the training set
    pipe.fit(X_train, y_train)
    
    # Transform the data
    X_train_lda = pipe.transform(X_train)
    X_test_lda = pipe.transform(X_test)
    
    # Plot LDA for this fold
    plt.figure()
    for i in np.unique(y_train):
        plt.scatter(X_train_lda[y_train == i], np.zeros_like(X_train_lda[y_train == i]), alpha=.8, label=f'Class {i} (train)')
    for i in np.unique(y_test):
        plt.scatter(X_test_lda[y_test == i], np.zeros_like(X_test_lda[y_test == i])-0.05, alpha=.8, label=f'Class {i} (test)')
    
    plt.legend(loc='best', shadow=False)
    plt.yticks([])  # No need for y-axis ticks in 1D plot
    plt.title(f'LDA of dataset for fold {fold_idx}')
    plt.xlabel('Linear Discriminant')
    plt.show()

    fold_idx += 1
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
# pipe = Pipeline([('scl',scl),('umap', umap),('clf',clf)])
pipe = Pipeline([('scl',scl()),('clf',clf)])
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

# cm = confusion_matrix(y,y_pred)
# print(cm)

#######Plot SVM#######
from matplotlib.colors import ListedColormap

fold_idx = 1
for train_idx, test_idx in skf.split(X, y):
    print(len(train_idx), len(test_idx))
    # Use .iloc[] for DataFrame and [] for numpy arrays
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]  # X is a pandas DataFrame
    y_train, y_test = y[train_idx], y[test_idx]  # y is a numpy array
    
    #### Scaled and Classified 1 by 1 ####

    X_train_scaled = scl.fit_transform(X_train,y_train)
    X_test_scaled = scl.transform(X_test)

    X_train_reduced = umap.fit_transform(X_train_scaled,y_train)
    X_test_reduced = umap.transform(X_test_scaled)
    
    clf.fit(X_train_reduced, y_train)
    y_pred = clf.predict(X_test_reduced)

    x_min, x_max = X_test_reduced[:, 0].min() - 1, X_test_reduced[:, 0].max() + 1
    y_min, y_max = X_test_reduced[:, 1].min() - 1, X_test_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                          np.arange(y_min, y_max, 0.02))

    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = np.reshape(clf.predict(grid), xx.shape)

    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3,cmap = ListedColormap(('green','red')))
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())

    for i,j in enumerate(np.unique(y_train)):
        if j == 0:
            label = 'Rest_train'
        else:
            label = 'Stress_train'
        plt.scatter(X_train_reduced[y_train ==j,0],X_train_reduced[y_train==j,1],
                    c=ListedColormap(('limegreen','lightcoral'))(i),label=label)
    
    for i,j in enumerate(np.unique(y_test)):
        if j == 0:
            label = 'Rest'
        else:
            label = 'Stress'
        plt.scatter(X_test_reduced[y_test ==j,0],X_test_reduced[y_test==j,1],
                    c=ListedColormap(('green','red'))(i),label=label)
    
    plt.legend()
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Decision Boundary for Linear SVC fold {fold_idx}')
    plt.show()

    fold_idx+=1
#######Plot SVM#######

################################ CLASSIFICATION 2 ################################