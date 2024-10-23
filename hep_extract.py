# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 11:35:09 2024

@author: Jeremi
"""

import mne
import numpy as np
import os
import neurokit2 as nk
from wfdb import processing

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

folder_raw = 'raw_data'
fpath_raw=[i for i in os.listdir(folder_raw) if i.endswith('.vhdr')]

folder_asr = 'filtered_data_asr'
fpath_asr=[i for i in os.listdir(folder_asr)]

folder_ica = 'filtered_data_ica_all'
fpath_ica=[i for i in os.listdir(folder_ica)]

folder_epoch_asr = "hep_asr"
folder_epoch_ica = "hep_ica_all"

#####################################################

counter = 0

raw = mne.io.read_raw_brainvision(os.path.join(folder_raw, fpath_raw[counter]))

events, event_ids = mne.events_from_annotations(raw)

raw.set_channel_types({'ECG':'ecg'})
raw.set_channel_types({'vEOG':'eog'})
raw.set_channel_types({'hEOG':'eog'})

fs = 1000
tmin = events[3,0]/fs #Experiment Begin 
tmax = events[-1,0]/fs #Task Begin

if fpath_raw[counter] == '20231019_B68_stroop5mins_0001.vhdr':
    tmin = events[9,0]/fs
if fpath_raw[counter] == '20240725_X00_jikken_0003.vhdr':
    tmax = events[-2,0]/fs
if fpath_raw[counter] == '20240725_X00_jikken_0001.vhdr':
    tmax = events[-2,0]/fs
if fpath_raw[counter] == '20240418_B98_jikken_0001.vhdr':
    tmax = tmin + 900

#####################################################

def make_r_peaks_events(raw=raw):
    raw_ecg = raw.copy().pick_types(ecg=True).crop(tmin = tmin, tmax = tmax) #make a copy

    mne_ecg,_ = raw_ecg[:]
    mne_ecg = np.squeeze(-mne_ecg)

    _, info = nk.ecg_process(mne_ecg, sampling_rate = fs)

    results_corr = processing.correct_peaks(mne_ecg, info['ECG_R_Peaks'], 36, 50, 'up')

    r_peak_onset = []
    for i in range(len(results_corr)):
        ons_idx = int(fs*tmin)+results_corr[i]
        r_peak_onset.append(ons_idx)

    r_peaks_events = np.zeros((len(r_peak_onset), 3), dtype=int)

    r_peaks_events[:, 0] = r_peak_onset
    r_peaks_events[:, 1] = 0 
    r_peaks_events[:, 2] = 7

    return(r_peaks_events,raw_ecg)

#####################################################
#####################################################

r_peak_events,raw_ecg = make_r_peaks_events(raw)

#####################################################     
#####################################################

def save_hep(folder_dir,file_dir,folder_save,epoch_events=r_peak_events):
    
    raw = mne.io.read_raw_fif(os.path.join(folder_dir,file_dir)).pick_types(['eeg'])
    fs = 1000
    
    events, event_ids = mne.events_from_annotations(raw)

    trg0 = events[0,0] #Experiment Begin 
    trg1 = events[1,0] #Task Begin
    if file_dir.find('B98_jikken_0001')>=1:
        trg2 = events[-1,0]
        trg3 = trg0 + 900000
    else:
        trg2 = events[-2,0] #Task End
        trg3 = events[-1,0] #Experiment End
    
    #Segment in Samples
    eeg_seg1 = trg1-trg0
    eeg_seg2 = trg2-trg0
    eeg_seg3 = trg3-trg0

    #Segment in ms
    eeg_newseg1 = float(eeg_seg1/fs)
    eeg_newseg2 = float(eeg_seg2/fs)
    # eeg_newseg3 = int(eeg_seg3/fs)
    
    epoch_tmin = -0.2
    epoch_tmax = 0.6
    baseline = (None,0)
    
    firstrest = raw.copy().crop(tmin = 0, tmax = eeg_newseg1)
    stress = raw.copy().crop(tmin = eeg_newseg1, tmax = eeg_newseg2)
    secondrest = raw.copy().crop(tmin = eeg_newseg2)
    
    # Filter events for each segment
    firstrest_events = epoch_events[(epoch_events[:, 0] >= trg0) & (epoch_events[:, 0] <= trg1)]
    stress_events = epoch_events[(epoch_events[:, 0] >= trg1) & (epoch_events[:, 0] <= trg2)]
    secondrest_events = epoch_events[(epoch_events[:, 0] >= trg2)]
      
    # Creating epochs
    hep_epoch = mne.Epochs(raw,
                           events=epoch_events,
                           tmin=epoch_tmin, 
                           tmax=epoch_tmax, 
                           baseline=baseline
                           )
    
    firstrest_epoch = mne.Epochs(firstrest,
                                 events=firstrest_events, 
                                 tmin=epoch_tmin, 
                                 tmax=epoch_tmax, 
                                 baseline=baseline
                                 )
    stress_epoch = mne.Epochs(stress,
                              events=stress_events, 
                              tmin=epoch_tmin, 
                              tmax=epoch_tmax, 
                              baseline=baseline
                              )
    secondrest_epoch = mne.Epochs(secondrest,
                                  events=secondrest_events, 
                                  tmin=epoch_tmin, 
                                  tmax=epoch_tmax, 
                                  baseline=baseline
                                  )

    firstrest_epoch.save(os.path.join(folder_save,file_dir.replace(".fif", "-first-epo.fif")), overwrite=True)
    stress_epoch.save(os.path.join(folder_save,file_dir.replace(".fif", "-stress-epo.fif")), overwrite=True)
    secondrest_epoch.save(os.path.join(folder_save,file_dir.replace(".fif", "-second-epo.fif")), overwrite=True)
    
#####################################################
#####################################################

# save_hep(folder_asr,fpath_asr[counter],folder_epoch_asr,r_peak_events)
save_hep(folder_ica,fpath_ica[counter],folder_epoch_ica,r_peak_events)

for i in range(len(fpath_raw)):
    counter = i

    raw = mne.io.read_raw_brainvision(os.path.join(folder_raw, fpath_raw[counter]))

    events, event_ids = mne.events_from_annotations(raw)

    raw.set_channel_types({'ECG':'ecg'})
    raw.set_channel_types({'vEOG':'eog'})
    raw.set_channel_types({'hEOG':'eog'})

    fs = 1000
    tmin = events[3,0]/fs #Experiment Begin 
    tmax = events[-1,0]/fs #Task Begin

    if fpath_raw[counter] == '20231019_B68_stroop5mins_0001.vhdr':
        tmin = events[9,0]/fs
    if fpath_raw[counter] == '20240725_X00_jikken_0003.vhdr':
        tmax = events[-2,0]/fs
    if fpath_raw[counter] == '20240725_X00_jikken_0001.vhdr':
        tmax = events[-2,0]/fs
    if fpath_raw[counter] == '20240418_B98_jikken_0001.vhdr':
        tmax = tmin + 900

    #####################################################

    def make_r_peaks_events(raw=raw):
        raw_ecg = raw.copy().pick_types(ecg=True).crop(tmin = tmin, tmax = tmax) #make a copy

        mne_ecg,_ = raw_ecg[:]
        mne_ecg = np.squeeze(-mne_ecg)

        _, info = nk.ecg_process(mne_ecg, sampling_rate = fs)

        results_corr = processing.correct_peaks(mne_ecg, info['ECG_R_Peaks'], 36, 50, 'up')

        r_peak_onset = []
        for i in range(len(results_corr)):
            ons_idx = int(fs*tmin)+results_corr[i]
            r_peak_onset.append(ons_idx)

        r_peaks_events = np.zeros((len(r_peak_onset), 3), dtype=int)

        r_peaks_events[:, 0] = r_peak_onset
        r_peaks_events[:, 1] = 0 
        r_peaks_events[:, 2] = 7

        return(r_peaks_events,raw_ecg)

    #####################################################
    #####################################################

    r_peak_events,raw_ecg = make_r_peaks_events(raw)

    #####################################################     
    #####################################################

    def save_hep(folder_dir,file_dir,folder_save,epoch_events=r_peak_events):
        
        raw = mne.io.read_raw_fif(os.path.join(folder_dir,file_dir)).pick_types(eeg=True)
        fs = 1000
        
        events, event_ids = mne.events_from_annotations(raw)

        trg0 = events[0,0] #Experiment Begin 
        trg1 = events[1,0] #Task Begin
        if file_dir.find('B98_jikken_0001')>=1:
            trg2 = events[-1,0]
            trg3 = trg0 + 900000
        else:
            trg2 = events[-2,0] #Task End
            trg3 = events[-1,0] #Experiment End
        
        #Segment in Samples
        eeg_seg1 = trg1-trg0
        eeg_seg2 = trg2-trg0
        # eeg_seg3 = trg3-trg0

        #Segment in ms
        eeg_newseg1 = float(eeg_seg1/fs)
        eeg_newseg2 = float(eeg_seg2/fs)
        # eeg_newseg3 = int(eeg_seg3/fs)
        
        epoch_tmin = -0.2
        epoch_tmax = 0.6
        baseline = (None,0)
        
        firstrest = raw.copy().crop(tmin = 0, tmax = eeg_newseg1)
        stress = raw.copy().crop(tmin = eeg_newseg1, tmax = eeg_newseg2)
        secondrest = raw.copy().crop(tmin = eeg_newseg2)
        
        # Filter events for each segment
        firstrest_events = epoch_events[(epoch_events[:, 0] >= trg0) & (epoch_events[:, 0] <= trg1)]
        stress_events = epoch_events[(epoch_events[:, 0] >= trg1) & (epoch_events[:, 0] <= trg2)]
        secondrest_events = epoch_events[(epoch_events[:, 0] >= trg2)]
          
        # Creating epochs
        # hep_epoch = mne.Epochs(raw,
        #                        events=epoch_events,
        #                        tmin=epoch_tmin, 
        #                        tmax=epoch_tmax, 
        #                        baseline=baseline
        #                        )
        drop = dict(eeg = 100e-6)
        
        firstrest_epoch = mne.Epochs(firstrest,
                                     events=firstrest_events, 
                                     tmin=epoch_tmin, 
                                     tmax=epoch_tmax, 
                                     baseline=baseline,
                                     reject=drop
                                     )
        stress_epoch = mne.Epochs(stress,
                                  events=stress_events, 
                                  tmin=epoch_tmin, 
                                  tmax=epoch_tmax, 
                                  baseline=baseline,
                                  reject=drop
                                  )
        secondrest_epoch = mne.Epochs(secondrest,
                                      events=secondrest_events, 
                                      tmin=epoch_tmin, 
                                      tmax=epoch_tmax, 
                                      baseline=baseline,
                                      reject=drop
                                      )

        firstrest_epoch.save(os.path.join(folder_save,file_dir.replace(".fif", "-first-epo.fif")), overwrite=True)
        stress_epoch.save(os.path.join(folder_save,file_dir.replace(".fif", "-stress-epo.fif")), overwrite=True)
        secondrest_epoch.save(os.path.join(folder_save,file_dir.replace(".fif", "-second-epo.fif")), overwrite=True)
        
    #####################################################
    #####################################################

    save_hep(folder_asr,fpath_asr[counter],folder_epoch_asr,r_peak_events)
    save_hep(folder_ica,fpath_ica[counter],folder_epoch_ica,r_peak_events)
    
    
