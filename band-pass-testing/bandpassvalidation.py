
import mne

# Path to the EEGLAB .set file (make sure the corresponding .fdt file is in the same location)
file_path = "../data/wireless_raw_8.set"
# Load the data
raw = mne.io.read_raw_eeglab(file_path, preload=True)

# Add standard electrode locations
montage = mne.channels.make_standard_montage('standard_1005')
raw.set_montage(montage)

# Resample data to 100 Hz
raw.resample(125)

# # Convert >  be >16 bit
eeg_data_in_volts = raw.get_data() * 1e6
# If you want to replace the data in the MNE object itself:
raw._data = eeg_data_in_volts

# draw raw data with first 1 second
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(12, 6))
# raw.plot(duration=1.0, n_channels=raw.info['nchan'],
#          scalings='auto', title='Raw EEG data (first 1 second)',
#          show=True, block=True)

# You can also try a different visualization with specific channels
plt.figure(figsize=(12, 6))
data, times = raw[:, :int(raw.info['sfreq'])]  # Get first second of data
plt.plot(times, data.T)
plt.title('Raw EEG data (first 1 second)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV)')
plt.show()


# Band-pass filter between 1 and 45 Hz
# raw.filter(1., 45., fir_design='firwin')


# Apply a band-pass filter from 1 to 45 Hz
filtered_data = mne.filter.filter_data(
    data,
    sfreq=125.0,
    l_freq=1.,
    h_freq=45.,
    fir_design='firwin',
    pad='reflect_limited'
)


# show filtered data after band-pass
plt.figure(figsize=(12, 6))
data = filtered_data  # Get first second of data
plt.plot(times, data.T)
plt.title('Raw EEG data (first 1 second) - after band-pass')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (μV)')
plt.show()
