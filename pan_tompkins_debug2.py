# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:36:02 2024

@author: Jeremi
"""
import numpy as np
from scipy import signal as sg

fs = 1000

RR1, RR2, probable_peaks, r_locs, peaks, result = ([] for i in range(6))
SPKI, NPKI, Threshold_I1, Threshold_I2, SPKF, NPKF, Threshold_F1, Threshold_F2 = (0 for i in range(8))

T_wave = False
m_win = mwin
b_pass = bpass
fs = fs
# signal = np.squeeze(x)
signal = mne_ecg
win_150ms = round(0.15*fs)

RR_Low_Limit = 0
RR_High_Limit = 0
RR_Missed_Limit = 0
RR_Average1 = 0

slopes = sg.fftconvolve(m_win, np.full((25,),1)/25, mode='same')
# slopes = sg.fftconvolve(m_win, np.full((25,1),1)/25, mode='same')
# for i in range(round(0.5*fs) + 1,len(slopes)-1):
for i in range(0,len(slopes)-1):
    if (slopes[i] > slopes[i-1]) and (slopes[i+1] <slopes[i]):
        peaks.append(i)
        
for ind in range(len(peaks)):

    # Initialize the search window for peak detection
    peak_val = peaks[ind]
    win_300ms = np.arange(max(0, peaks[ind] - win_150ms), min(peaks[ind] + win_150ms, len(b_pass)-1), 1)
    max_val = max(b_pass[win_300ms], default = 0)
    
    # Find the x location of the max peak value
    if (max_val != 0):        
        x_coord = np.asarray(b_pass == max_val).nonzero()
        probable_peaks.append(x_coord[0][0])
    # if ind ==0:
    #     print('x_coord', x_coord)
    #     print('probable_peaks', probable_peaks)
        
    if (ind < len(probable_peaks) and ind != 0):
        # Adjust RR interval and limits
        ################################################# adjust_rr_interval(ind)
        RR1 = np.diff(peaks[max(0,ind - 8) : ind + 1])/fs  
        
        # print('RR1:',  RR1)
        
        # Calculating RR Averages
        RR_Average1 = np.mean(RR1)
        RR_Average2 = RR_Average1
        
        # print('RR2_1:',  RR_Average2)
        
        # Finding the eight most recent RR intervals lying between RR Low Limit and RR High Limit  
        if (ind >= 8):
            for i in range(0, 8):
                if (RR_Low_Limit < RR1[i] < RR_High_Limit): 
                    RR2.append(RR1[i])

                    if (len(RR2) > 8):
                        RR2.remove(RR2[0])
                        RR_Average2 = np.mean(RR2)    
        
        # print('RR2_2:',  RR_Average2)
        
        # Adjusting the RR Low Limit and RR High Limit
        if (len(RR2) > 7 or ind < 8):
            RR_Low_Limit = 0.92 * RR_Average2        
            RR_High_Limit = 1.16 * RR_Average2
            RR_Missed_Limit = 1.66 * RR_Average2
        
        # Adjust thresholds in case of irregular beats
        if (RR_Average1 < RR_Low_Limit or RR_Average1 > RR_Missed_Limit): 
            Threshold_I1 /= 2
            Threshold_F1 /= 2

        RRn = RR1[-1]

        # Searchback
        ################################################# searchback(peak_val,RRn,round(RRn*fs))
        if (RRn > RR_Missed_Limit):
            # Initialize a window to searchback  
            win_rr = m_win[peak_val - round(RRn*fs) + 1 : peak_val + 1] 

            # Find the x locations inside the window having y values greater than Threshold I1             
            coord = np.asarray(win_rr > Threshold_I1).nonzero()[0]

            # Find the x location of the max peak value in the search window
            if (len(coord) > 0):
                for pos in coord:
                    if (win_rr[pos] == max(win_rr[coord])):
                        x_max = pos
                        break
            else:
                x_max = None
    
            # If the max peak value is found
            if (x_max is not None):   
                # Update the thresholds corresponding to moving window integration
                SPKI = 0.25 * m_win[x_max] + 0.75 * SPKI                         
                Threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
                Threshold_I2 = 0.5 * Threshold_I1         

                # Initialize a window to searchback 
                win_rr = b_pass[x_max - win_150ms: min(len(b_pass) -1, x_max)]  

                # Find the x locations inside the window having y values greater than Threshold F1                   
                coord = np.asarray(win_rr > Threshold_F1).nonzero()[0]

                # Find the x location of the max peak value in the search window
                if (len(coord) > 0):
                    for pos in coord:
                        if (win_rr[pos] == max(win_rr[coord])):
                            r_max = pos
                            break
                else:
                    r_max = None

                # If the max peak value is found
                if (r_max is not None):
                # Update the thresholds corresponding to bandpass filter
                    if b_pass[r_max] > Threshold_F2:                                                        
                        SPKF = 0.25 * b_pass[r_max] + 0.75 * SPKF                            
                        Threshold_F1 = NPKF + 0.25 * (SPKF - NPKF)
                        Threshold_F2 = 0.5 * Threshold_F1      

                        # Append the probable R peak location                      
                        r_locs.append(r_max)
        # T Wave Identification
        ################################################# find_t_wave(peak_val,RRn,ind,ind-1)
        if (m_win[peak_val] >= Threshold_I1): 
            if (ind > 0 and 0.20 < RRn < 0.36):
                # Find the slope of current and last waveform detected        
                curr_slope = max(np.diff(m_win[peak_val - round(win_150ms/2) : peak_val + 1]))
                last_slope = max(np.diff(m_win[peaks[ind-1] - round(win_150ms/2) : peaks[ind-1] + 1]))
            
                # If current waveform slope is less than half of last waveform slope
                if (curr_slope < 0.5*last_slope):  
                    # T Wave is found and update noise threshold                      
                    T_wave = True                             
                    NPKI = 0.125 * m_win[peak_val] + 0.875 * NPKI 

            if (not T_wave):
                # T Wave is not found and update signal thresholds
                if (probable_peaks[ind] > Threshold_F1):   
                    SPKI = 0.125 * m_win[peak_val]  + 0.875 * SPKI                                         
                    SPKF = 0.125 * b_pass[ind] + 0.875 * SPKF 

                    # Append the probable R peak location
                    r_locs.append(probable_peaks[ind])  

                else:
                    SPKI = 0.125 * m_win[peak_val]  + 0.875 * SPKI
                    NPKF = 0.125 * b_pass[ind] + 0.875 * NPKF                   

        # Update noise thresholds
        elif (m_win[peak_val] < Threshold_I1) or (Threshold_I1 < m_win[peak_val] < Threshold_I2):
            NPKI = 0.125 * m_win[peak_val]  + 0.875 * NPKI  
            NPKF = 0.125 * b_pass[ind] + 0.875 * NPKF

    else:
        # Adjust threholds
        if (m_win[peak_val] >= Threshold_I1): 
            # Update signal threshold
            SPKI = 0.125 * m_win[peak_val]  + 0.875 * SPKI

            if (probable_peaks[ind] > Threshold_F1):                                            
                SPKF = 0.125 * b_pass[ind] + 0.875 * SPKF 

                # Append the probable R peak location
                r_locs.append(probable_peaks[ind])  

            else:
                # Update noise threshold
                NPKF = 0.125 * b_pass[ind] + 0.875 * NPKF                                    
            
        # Update noise thresholds    
        elif (m_win[peak_val] < Threshold_I2) or (Threshold_I2 < m_win[peak_val] < Threshold_I1):
            NPKI = 0.125 * m_win[peak_val]  + 0.875 * NPKI  
            NPKF = 0.125 * b_pass[ind] + 0.875 * NPKF
    # Update threholds for next iteration
    # update_thresholds()
    Threshold_I1 = NPKI + 0.25 * (SPKI - NPKI)
    Threshold_F1 = NPKF + 0.25 * (SPKF - NPKF)
    Threshold_I2 = 0.5 * Threshold_I1 
    Threshold_F2 = 0.5 * Threshold_F1
    T_wave = False 

# Searchback in ECG signal 
r_locs = np.unique(np.array(r_locs).astype(int))

# Initialize a window to searchback
win_200ms = round(0.2*fs)



for r_val in r_locs:
    
    coord = np.arange(r_val - win_200ms, min(len(signal), r_val + win_200ms + 1), 1)
    
    # Find the x location of the max peak value
    if (len(coord) > 0):
        for pos in coord:               
            if (signal[pos] == max(signal[coord])):
                x_max = pos
                break
    else:
        x_max = None

    # Append the peak location
    if (x_max is not None):   
        result.append(x_max)