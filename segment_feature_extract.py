# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 14:21:38 2024

@author: Jeremi
"""

import numpy as np

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