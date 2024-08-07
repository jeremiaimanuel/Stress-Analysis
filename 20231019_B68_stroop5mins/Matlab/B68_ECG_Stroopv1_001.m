clear all;
close all;
clc;

%%Load File
M1 = readtable("B68_CroppedECG_Stroopv1_001.csv");
% M2 = readtable("B68_CroppedECG_Stroopv1_001_resampled.csv");
fs1 = 1000;

%Pan-Topmkins Filter
rr = pan_tompkin(M1.ECG, fs1);  

% %STFT
% stft(rr,fs1,Window=kaiser(256,5),OverlapLength=220,FFTLength=512)