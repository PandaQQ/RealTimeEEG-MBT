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
from mne.export import export_raw

# Path to the EEGLAB .set file (make sure the corresponding .fdt file is in the same location)
file_path = "../data/wireless_raw.set"
# Load the data
raw = mne.io.read_raw_eeglab(file_path, preload=True)

# Add standard electrode locations
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

# Resample data to 125 Hz
raw.resample(125)

# Band-pass filter between 1 and 45 Hz
# raw.filter(1., 45., fir_design='firwin')

seg_length = int(raw.info['sfreq'])  # 1-second segments
n_segments = int(raw.n_times / seg_length)
data_clean = raw.get_data()

# Extract latencies for specific events
events, event_id = mne.events_from_annotations(raw)
lats = [event[0] for event in events if event[2] in [2, 3]]


# Prepare data_relax and data_calcu
data_relax = []
data_calcu = []

end_of_data = 0
# for j in range(60):
for j in range(60):
    # Relaxation segment
    start = lats[j * 2]
    end = start + int(30 * seg_length)
    # show current start and end
    print(start, end)

    # Calculation segment
    start = lats[j * 2 + 1]
    end = lats[j * 2 + 2]
    # show current start and end
    print(start, end)
    if j == 48:
        end_of_data = end
        print("End of data: ", end_of_data)


print("End of data: ", end_of_data)

'''
So I want to split the data into two parts, from 0 to 283842 is first part, from 283842 to the end is the second part
and then save these 2 files as new set files
'''


# Suppose we have the split index in samples
split_idx = end_of_data + 1

# Convert that split index to seconds
split_time_sec = split_idx / raw.info['sfreq']

# 1) Create the first segment from t = 0 to t = split_time_sec
raw_part1 = raw.copy().crop(tmin=0,
                            tmax=(split_idx - 1) / raw.info['sfreq'])  # -1 to ensure we stop exactly before split_time_sec if needed

# 2) Create the second segment from t = split_time_sec to end
raw_part2 = raw.copy().crop(tmin=split_time_sec, tmax=None)

# 3) Save each part as an EEGLAB .set file
raw_part1.export('../data/wireless_raw_8.set', fmt='eeglab', overwrite=True)
raw_part2.export('../data/wireless_raw_2.set', fmt='eeglab', overwrite=True)