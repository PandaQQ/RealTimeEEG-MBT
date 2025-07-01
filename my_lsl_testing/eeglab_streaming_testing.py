import time
from pylsl import StreamInfo, StreamOutlet
from scipy.io import loadmat
import mne
from pylsl import StreamInfo, StreamOutlet
import time
import sys
import os


# data_temp = loadmat('sample_data_24ch.mat')
# data = data_temp['data']
# info = StreamInfo('EEG_stream', 'EEG', 24, 0, 'float32', 'myuid34234')
# info1 = StreamInfo('MyMarkerStream', 'Markers', 1, 0, 'string', 'myuidw43536')
#
# # next make an outlet
# outlet = StreamOutlet(info)
# outlet1 = StreamOutlet(info1)
#
# j = 0
# while True:
#     mysample = data[:, j].tolist()
#     outlet.push_sample(mysample)
#     time.sleep(0.004)
#     j = j + 1
#     if j % 500 == 0:
#         outlet1.push_sample('1')
#         print('1')

# Load EEG data from WM_Oct8_testing.vhdr file
eeglab_file = '../data/wireless_raw.set'
raw = mne.io.read_raw_eeglab(eeglab_file, preload=True)

# Add standard electrode locations
# montage = mne.channels.make_standard_montage('standard_1005')
# raw.set_montage(montage)

# Resample data to 100 Hz
raw.resample(250)
# Convert from microvolts to volts
eeg_data_in_volts = raw.get_data() * 1e6
# # If you want to replace the data in the MNE object itself:
raw._data = eeg_data_in_volts

# Extract data and metadata
data, times = raw.get_data(return_times=True)
n_channels = raw.info['nchan']
sfreq = raw.info['sfreq']
channel_names = raw.info['ch_names']

# Define LSL stream info
info = StreamInfo(name='EEG_Stream',
                  type='EEG',
                  channel_count=n_channels,
                  nominal_srate=sfreq,
                  channel_format='float32',
                  source_id='eeglab_stream')

info1 = StreamInfo('MyMarkerStream',
                   'Markers',
                   1,
                   0,
                   'string',
                   'myuidw43536')

# Add channel names to stream description
ch_names = info.desc().append_child("channels")
for ch in channel_names:
    ch_names.append_child("channel").append_child_value("label", ch)

# Create outlet to stream data
outlet = StreamOutlet(info)
# Calculate the duration between samples in seconds
sample_interval = 1.0 / sfreq
# Stream data at the same frequency as the original recording
for i in range(data.shape[1]):
    outlet.push_sample(data[:, i])
    temp_data = data[:, i]
    # Pause for the duration of one sample to maintain the original frequency
    time.sleep(sample_interval)
    if i % sfreq == 0:
        outlet1 = StreamOutlet(info1)
        outlet1.push_sample('1')
        print('1')