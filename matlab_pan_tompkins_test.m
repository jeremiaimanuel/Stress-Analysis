ecg_data = EEG.data;
ecg_data = -ecg_data;
time = EEG.times;

fs = 1000;
des_fs = 250;
[p,q] = rat(des_fs / fs);

ds_ecg_data = resample(ecg_data, p,q);
ds_ecg_time = resample(time, p,q);
y = pan_tompkin(ecg_data, 250, 0);

plot(ds_ecg_time, ds_ecg_data, y, 'o');