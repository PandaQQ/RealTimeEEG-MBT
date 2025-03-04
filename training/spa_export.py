"""
This script processes EEG data using MNE and prepares it for analysis. It includes:
- EEG preprocessing steps like resampling, filtering, and converting units.
- Second-by-second signal cleaning using PCA.
- Segmentation of relaxation and calculation data.
- Exporting the processed data and labels into a .mat file.
"""

import mne
import numpy as np
import scipy.io
from sklearn.decomposition import PCA

# Path to the EEGLAB .set file (make sure the corresponding .fdt file is in the same location)
file_path = "./data/wireless_raw.set"
# Load the data
raw = mne.io.read_raw_eeglab(file_path, preload=True)

# Add standard electrode locations
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

# Resample data to 100 Hz
raw.resample(100)

# Band-pass filter between 1 and 45 Hz
raw.filter(1., 45., fir_design='firwin')

# # Convert >  be >16 bit
eeg_data_in_volts = raw.get_data() * 1e6
# If you want to replace the data in the MNE object itself:
raw._data = eeg_data_in_volts

# SPA: Second-by-second cleaning using PCA
cutoff = 50  # No need to change (50 - 200)
seg_length = int(raw.info['sfreq'])  # 1-second segments
n_segments = int(raw.n_times / seg_length)
data_clean = raw.get_data()

for j in range(n_segments):
    start = j * seg_length
    end = start + seg_length
    segment = data_clean[:, start:end]  # 转置为 (样本数, 通道数)
    # 使用PCA（自动中心化）
    pca = PCA()
    com = pca.fit_transform(np.transpose(segment))  # 主成分得分
    # 应用方差阈值 (MATLAB等价逻辑)
    high_var_mask = pca.explained_variance_ > (cutoff ** 2)
    com[:, high_var_mask] = 0  # 置零高方差成分
    # 重构数据并转置回原始维度
    reconstructed = pca.inverse_transform(com)  # (n_channels, seg_length)
    data_clean[:, start:end] = np.transpose(reconstructed)

raw._data = data_clean  # Update raw data with cleaned data

# Extract latencies for specific events
events, event_id = mne.events_from_annotations(raw)
lats = [event[0] for event in events if event[2] in [2, 3]]

# get events of S 33 and S 22
# print(events)
# exit()


# Prepare data_relax and data_calcu
data_relax = []
data_calcu = []

for j in range(60):
    # Relaxation segment
    start = lats[j * 2]
    end = start + int(30 * seg_length)
    relax_segment = raw.get_data(start=start, stop=end)
    relax_segment = relax_segment.reshape((-1, raw.info['nchan'], seg_length))
    data_relax.append(relax_segment)

    # Calculation segment
    start = lats[j * 2 + 1]
    end = lats[j * 2 + 2]
    calcu_segment = raw.get_data(start=start, stop=end)
    n_samples = calcu_segment.shape[1] // seg_length * seg_length
    calcu_segment = calcu_segment[:, :n_samples]
    calcu_segment = calcu_segment.reshape((-1, raw.info['nchan'], seg_length))
    data_calcu.append(calcu_segment)

# Combine and label data
data_relax = np.concatenate(data_relax, axis=0)
data_calcu = np.concatenate(data_calcu, axis=0)

data = np.concatenate((data_relax, data_calcu), axis=0)
# save labels data as (3121, 1)
labels = np.concatenate((np.zeros(data_relax.shape[0]), np.ones(data_calcu.shape[0]))).reshape(-1, 1)
# Save to MAT file
scipy.io.savemat('./training/EEG_SPA_001.mat', {'data': data, 'labels': labels})