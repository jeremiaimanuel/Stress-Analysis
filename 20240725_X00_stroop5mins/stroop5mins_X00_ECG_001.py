# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 12:43:19 2023

@author: Jeremi
"""

import mne
import os
import numpy as np
from scipy import signal
from matplotlib import pyplot as plt
from pan_tompkins import pan_tompkins_qrs
from pan_tompkins import heart_rate

fs = 1000
directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

raw = mne.io.read_raw_brainvision("20240725_X00_stroop5mins/20240725_X00_jikken_0001.vhdr")
raw.load_data()

# Reconstruct the original events from our Raw object
events, event_ids = mne.events_from_annotations(raw)

raw.set_channel_types({'ECG':'ecg'})
raw.set_channel_types({'vEOG':'eog'})
raw.set_channel_types({'hEOG':'eog'})

tmin = events[3,0]/fs
tmax = events[-2,0]/fs

# iir_params = dict(order=2, ftype='butter', output = 'sos')
# raw_ecg = raw.copy().crop(tmin = tmin, tmax = tmax).pick_types(eeg=False, eog=False, ecg=True).filter(0.5, 150, picks = 'ecg', method ='iir', iir_params = iir_params)
raw_ecg = raw.copy().crop(tmin = tmin, tmax = tmax).pick_types(eeg=False, eog=False, ecg=True)

mne_ecg, mne_time = raw_ecg[:]
mne_ecg = np.squeeze(-mne_ecg)

b, a = signal.butter(2, [0.5, 150], 'bandpass', output= 'ba', fs=fs)
filtered = signal.filtfilt(b,a,mne_ecg)

# b, a = signal.butter(2, [2, 40], 'bandpass', output= 'ba', fs=fs) ## only need to remove the very first noise peak
# filtered = signal.filtfilt(b,a,mne_ecg)

mne_ecg = filtered.copy()

QRS = pan_tompkins_qrs()
output = QRS.solve(mne_ecg, fs)
bpass = QRS.bpf(mne_ecg)
der = QRS.derivative(bpass,fs)
sqr = QRS.squaring(der)
mwin = QRS.moving_window_integration(sqr, fs)

start_plot = 300
stop_plot = 3300

f, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex = True)
plt.xticks(np.arange(start_plot,stop_plot, 250))
ax1.plot(mne_ecg[start_plot:stop_plot])
ax2.plot(bpass[start_plot:stop_plot])
ax3.plot(der[start_plot:stop_plot])
ax4.plot(sqr[start_plot:stop_plot])
ax5.plot(mwin[start_plot:stop_plot])
ax1.title.set_text("Raw Signal")
ax2.title.set_text('Bandpassed Signal')
ax3.title.set_text('Derivative Signal')
ax4.title.set_text('Squared Signal')
ax5.title.set_text('Moving Window Integrated Signal')
plt.xlabel('Samples')
plt.ylabel('mV')
plt.show()

# # Plotting raw signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(start_plot,stop_plot, 250))
# plt.plot(mne_ecg[start_plot:stop_plot])
# # plt.axvline(x = 300000, color = 'r')
# # plt.axvline(x = 600000, color = 'r')
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Raw Signal")

# # Plotting bandpassed signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(start_plot,stop_plot, 250))
# plt.plot(bpass[start_plot:stop_plot])
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Bandpassed Signal")

# # Plotting derived signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(start_plot,stop_plot, 250))
# plt.plot(der[start_plot:stop_plot])
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Derivative Signal")

# # Plotting squared signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(start_plot,stop_plot, 250))
# plt.plot(sqr[start_plot:stop_plot])
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Squared Signal")

# # Plotting moving window integrated signal
# plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(start_plot,stop_plot, 250))
# plt.plot(mwin[start_plot:stop_plot])
# plt.xlabel('Samples')
# plt.ylabel('mV')
# plt.title("Moving Window Integrated Signal")


# Find the R peak locations
hr = heart_rate(mne_ecg, fs)
result = hr.find_r_peaks()
result = np.array(result)

# Clip the x locations less than 0 (Learning Phase)
result = result[result > 0]

# Plotting the R peak locations in ECG signal
plt.figure(figsize = (20,4), dpi = 100)
# plt.xticks(np.arange(0, len(mne_ecg)+1, 500))
plt.plot(mne_time*fs, mne_ecg, color = 'blue')        
plt.scatter(result, mne_ecg[result], color = 'red', s = 50, marker= '*')
# plt.axvline(x = 300000, color = 'r')
# plt.axvline(x = 600000, color = 'r')
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("R Peak Locations")

r_peak = np.unique(result)

###Pre Process, Data Cleaning ECG###

r_peak = r_peak[(r_peak>=250)
                & ~((r_peak >= 614500) & (r_peak <= 615000)) 
                & ~((r_peak >= 615250) & (r_peak <= 615500)) 
                & ~((r_peak >= 616000) & (r_peak <= 616300)) 
                & ~((r_peak >= 620000) & (r_peak <= 620300)) 
                & ~((r_peak >= 620500) & (r_peak <= 621000)) 
                & ~((r_peak >= 626000) & (r_peak <= 626100)) 
                & ~((r_peak >= 630000) & (r_peak <= 630100)) 
                & ~((r_peak >= 648600) & (r_peak <= 649000)) 
                & ~((r_peak >= 649300) & (r_peak <= 649600)) 
                & ~((r_peak >= 650000) & (r_peak <= 650300)) 
                & ~((r_peak >= 650800) & (r_peak <= 651000)) 
                & ~((r_peak >= 656800) & (r_peak <= 657000)) 
                & ~((r_peak >= 657700) & (r_peak <= 658000)) 
                & ~((r_peak >= 660750) & (r_peak <= 660775)) 
                & ~((r_peak >= 712000) & (r_peak <= 712300)) 
                & ~((r_peak >= 713400) & (r_peak <= 713600)) 
                & ~((r_peak >= 714000) & (r_peak <= 714400)) 
                & ~((r_peak >= 714800) & (r_peak <= 715000))]

###Pre Process, Data Cleaning ECG###

# ###Pre Process, Data Cleaning ECG###

# r_peak = r_peak[(r_peak>=250)]

# ###Pre Process, Data Cleaning ECG###

result = r_peak.copy()
rri = r_peak.copy()
rri[1:] = rri[1:] - rri[:-1]



####################Bandpassed Signal####################
b, a = signal.butter(2, [0.5, 150], 'bandpass', fs = fs)
filtered = signal.filtfilt(b,a,np.squeeze(mne_ecg))
# filtered = filtered.T
out_filt = QRS.solve(filtered,fs)
hr = heart_rate(filtered, fs)
result_filt = hr.find_r_peaks()
result_filt = np.array(result_filt)

# Clip the x locations less than 0 (Learning Phase)
result_filt = result_filt[result_filt > 0]

plt.figure(figsize = (20,4), dpi = 100)
plt.xticks(np.arange(0, len(mne_ecg)+1, 500))
plt.plot(filtered, color = 'blue')        
plt.scatter(result_filt, filtered[result_filt], color = 'red', s = 50, marker= '*')
# plt.axvline(x = 300000, color = 'r')
# plt.axvline(x = 600000, color = 'r')
plt.xlabel('Samples')
plt.ylabel('mV')
plt.title("R Peak Locations")


######################

raw_ecg_events,_,_ = mne.preprocessing.find_ecg_events(raw_ecg)
raw_ecg.plot(events = raw_ecg_events)

#################### adding event from the r peak pantompkins
r_peak_onset = []
for i in range(len(result)):
    ons_idx = int(fs*tmin)+result[i]
    r_peak_onset.append(ons_idx)

r_peak_anot = mne.Annotations(r_peak_onset, 0.1, 'R Peak')

raw_ecg.set_annotations(r_peak_anot)


################# Testing Different Detector (NOT WORKING) #############
def pan_tompkins_detector(unfiltered_ecg):
    """
    Jiapu Pan and Willis J. Tompkins.
    A Real-Time QRS Detection Algorithm. 
    In: IEEE Transactions on Biomedical Engineering 
    BME-32.3 (1985), pp. 230–236.
    """
    
    maxQRSduration = 0.150 #sec
    f1 = 5/fs
    f2 = 15/fs

    b, a = signal.butter(1, [f1*2, f2*2], btype='bandpass')

    filtered_ecg = signal.lfilter(b, a, unfiltered_ecg)        

    diff = np.diff(filtered_ecg) 

    squared = diff*diff

    N = int(maxQRSduration*fs)
    mwa = MWA_cumulative(squared, N)
    mwa[:int(maxQRSduration*fs*2)] = 0

    mwa_peaks = panPeakDetect(mwa, fs)

    return mwa_peaks

#Fast implementation of moving window average with numpy's cumsum function 
def MWA_cumulative(input_array, window_size):
    
    ret = np.cumsum(input_array, dtype=float)
    ret[window_size:] = ret[window_size:] - ret[:-window_size]
    
    for i in range(1,window_size):
        ret[i-1] = ret[i-1] / i
    ret[window_size - 1:]  = ret[window_size - 1:] / window_size
    
    return ret

def panPeakDetect(detection, fs):    

    min_distance = int(0.25*fs)

    signal_peaks = [0]
    noise_peaks = []

    SPKI = 0.0
    NPKI = 0.0

    threshold_I1 = 0.0
    threshold_I2 = 0.0

    RR_missed = 0
    index = 0
    indexes = []

    missed_peaks = []
    peaks = []

    for i in range(1,len(detection)-1):
        if detection[i-1]<detection[i] and detection[i+1]<detection[i]:
            peak = i
            peaks.append(i)

            if detection[peak]>threshold_I1 and (peak-signal_peaks[-1])>0.3*fs:
                    
                signal_peaks.append(peak)
                indexes.append(index)
                SPKI = 0.125*detection[signal_peaks[-1]] + 0.875*SPKI
                if RR_missed!=0:
                    if signal_peaks[-1]-signal_peaks[-2]>RR_missed:
                        missed_section_peaks = peaks[indexes[-2]+1:indexes[-1]]
                        missed_section_peaks2 = []
                        for missed_peak in missed_section_peaks:
                            if missed_peak-signal_peaks[-2]>min_distance and signal_peaks[-1]-missed_peak>min_distance and detection[missed_peak]>threshold_I2:
                                missed_section_peaks2.append(missed_peak)

                        if len(missed_section_peaks2)>0:
                            signal_missed = [detection[i] for i in missed_section_peaks2]
                            index_max = np.argmax(signal_missed)
                            missed_peak = missed_section_peaks2[index_max]
                            missed_peaks.append(missed_peak)
                            signal_peaks.append(signal_peaks[-1])
                            signal_peaks[-2] = missed_peak   

            else:
                noise_peaks.append(peak)
                NPKI = 0.125*detection[noise_peaks[-1]] + 0.875*NPKI

            threshold_I1 = NPKI + 0.25*(SPKI-NPKI)
            threshold_I2 = 0.5*threshold_I1

            if len(signal_peaks)>8:
                RR = np.diff(signal_peaks[-9:])
                RR_ave = int(np.mean(RR))
                RR_missed = int(1.66*RR_ave)

            index = index+1      
    
    signal_peaks.pop(0)

    return signal_peaks