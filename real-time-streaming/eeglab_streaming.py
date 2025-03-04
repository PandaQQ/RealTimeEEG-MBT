import mne
from pylsl import StreamInfo, StreamOutlet
import time
import sys
import os

# Load EEG data from WM_Oct8_testing.vhdr file
eeglab_file = './data/wireless_raw_2.set'
raw = mne.io.read_raw_eeglab(eeglab_file, preload=True)

# Add standard electrode locations
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

# Resample data to 100 Hz
raw.resample(250)

# Band-pass filter between 1 and 45 Hz
raw.filter(1., 45., fir_design='firwin')

# Convert from microvolts to volts
eeg_data_in_volts = raw.get_data() * 1e6
# If you want to replace the data in the MNE object itself:
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