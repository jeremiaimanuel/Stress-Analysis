import mne
import os
import asrpy
from mne.preprocessing import read_ica
from mne_icalabel import label_components


fpath = ['20231019_B68_stroop5mins/20231019_B68_stroop5mins_0001.vhdr',
        '20240129_B71_mat5mins/20240129_B71_mat5mins_0001.vhdr',
        '20240418_B98_mat2mins/20240418_B98_jikken_0004.vhdr',
        '20240418_B98_mat5mins/20240418_B98_jikken_0003.vhdr',
        '20240418_B98_stroop2mins/20240418_B98_jikken_0002.vhdr',
        '20240418_B98_stroop5mins/20240418_B98_jikken_0001.vhdr',
        '20240725_X00_mat2mins/20240725_X00_jikken_0004.vhdr',
        '20240725_X00_mat5mins/20240725_X00_jikken_0003.vhdr',
        '20240725_X00_stroop2mins/20240725_X00_jikken_0002.vhdr',
        '20240725_X00_stroop5mins/20240725_X00_jikken_0001.vhdr']


folder_filtered_asr = "filtered_data_asr"
folder_filtered_asr20 = "filtered_data_asr20"
folder_filtered_ica = "filtered_data_ica_all"

directory_path = "D:/EEG RESEARCH DATA"
os.chdir(directory_path)

def save_eeg_asr(directory, cutoff= 5):
    raw = mne.io.read_raw_brainvision(directory, preload = True)
    # raw = mne.io.read_raw_brainvision(path[file_number], preload = True)
    
    # Reconstruct the original events from our Raw object
    events, event_ids = mne.events_from_annotations(raw)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_channel_types({'ECG':'ecg'})
    raw.set_channel_types({'vEOG':'eog'})
    raw.set_channel_types({'hEOG':'eog'})
    
    # raw.set_eeg_reference(ref_channels='average')
    
    raw.set_montage(montage)
    
    fs = 1000
    tmin = events[3,0]/fs #Experiment Begin 
    tmax = events[-1,0]/fs #Task Begin
    
    if directory == "20231019_B68_stroop5mins/20231019_B68_stroop5mins_0001.vhdr":
        tmin = events[9,0]/fs
    if directory == '20240725_X00_mat5mins/20240725_X00_jikken_0003.vhdr':
        tmax = events[-2,0]/fs
    if directory == '20240725_X00_stroop5mins/20240725_X00_jikken_0001.vhdr':
        tmax = events[-2,0]/fs
    if directory == '20240418_B98_stroop5mins/20240418_B98_jikken_0001.vhdr':
        tmax = tmin + 900
    

    raw_asr = raw.copy().crop(tmin = tmin, tmax = tmax).pick_types(eeg=True, ecg=False, eog=False).filter(l_freq=1.0, h_freq=40, picks = ['eeg'])#make a copy
    print('raw_asr: copied')
    asr = asrpy.ASR(sfreq=raw_asr.info["sfreq"], cutoff=cutoff)
    print('asr: defined')
    asr.fit(raw_asr)
    print('asr: fited')
    raw_asr = asr.transform(raw_asr)
    print('asr: transformed')

    _, file_name = os.path.split(directory)
    if cutoff == 5:
        raw_asr.save(os.path.join(folder_filtered_asr,file_name.replace(".vhdr", "-asr.fif")), overwrite=True)
    elif cutoff == 20:
        raw_asr.save(os.path.join(folder_filtered_asr20,file_name.replace(".vhdr", "-asr.fif")), overwrite=True)

for i in fpath:
    save_eeg_asr(i)

def save_eeg_ica(directory):
    raw = mne.io.read_raw_brainvision(directory, preload = True)
    # raw = mne.io.read_raw_brainvision(path[file_number], preload = True)
    
    # Reconstruct the original events from our Raw object
    events, event_ids = mne.events_from_annotations(raw)
    
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_channel_types({'ECG':'ecg'})
    raw.set_channel_types({'vEOG':'eog'})
    raw.set_channel_types({'hEOG':'eog'})
    
    # raw.set_eeg_reference(ref_channels='average')
    raw.set_montage(montage)
    
    fs = 1000
    tmin = events[3,0]/fs #Experiment Begin 
    tmax = events[-1,0]/fs #Task Begin
    
    if directory == "20231019_B68_stroop5mins/20231019_B68_stroop5mins_0001.vhdr":
        tmin = events[9,0]/fs
    if directory == '20240725_X00_mat5mins/20240725_X00_jikken_0003.vhdr':
        tmax = events[-2,0]/fs
    if directory == '20240725_X00_stroop5mins/20240725_X00_jikken_0001.vhdr':
        tmax = events[-2,0]/fs
    if directory == '20240418_B98_stroop5mins/20240418_B98_jikken_0001.vhdr':
        tmax = tmin + 900
    
    filt_raw = raw.copy().crop(tmin = tmin, tmax = tmax).filter(l_freq=1.0, h_freq=100) #make a copy
    filt_raw.load_data()

    ica = read_ica(fname=directory.replace(".vhdr", "-ica.fif"))

    ic_labels = label_components(filt_raw, ica, method="iclabel")
    print(ic_labels["labels"])

    labels = ic_labels["labels"]
    
    exclude_idx = [
        idx for idx, label in enumerate(labels) if label not in ["brain", "other"]
    ]
    print(f"Excluding these ICA components: {exclude_idx}")
    
    # eog_indices = [idx for idx, label in enumerate(labels) if label in ["eye blink"]]
    # eog_cfa_indices = [idx for idx, label in enumerate(labels) if label in ["eye blink","heart beat"]]

    # if "heart beat" not in labels:
    #     print("No CFA Detected in %s" % file_name)

    # ica.exclude = eog_indices
    # reconst_raw_eog = filt_raw.copy()
    # ica.apply(reconst_raw_eog)

    # ica.exclude = eog_cfa_indices
    # reconst_raw_cfa = filt_raw.copy()
    # ica.apply(reconst_raw_cfa)
    
    ica.exclude = exclude_idx
    reconst_raw = filt_raw.copy()
    ica.apply(reconst_raw)
    
    _, file_name = os.path.split(directory)
    # reconst_raw_eog.save(os.path.join(folder_filtered_eog,file_name.replace(".vhdr", "-ica.fif")), overwrite=True)
    # reconst_raw_cfa.save(os.path.join(folder_filtered_cfa,file_name.replace(".vhdr", "-ica.fif")), overwrite=True)
    reconst_raw.save(os.path.join(folder_filtered_ica,file_name.replace(".vhdr", "-ica.fif")), overwrite=True)
    del(raw,events,event_ids,montage,fs,tmin,tmax,filt_raw,ica,ic_labels,labels,
        exclude_idx,reconst_raw,file_name)
    
for i in fpath:
    save_eeg_ica(i)

