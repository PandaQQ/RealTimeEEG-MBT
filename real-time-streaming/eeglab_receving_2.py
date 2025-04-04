import sys
import os
import time

import numpy as np
from pylsl import StreamInlet, resolve_streams
import mne
import matplotlib.pyplot as plt
from pylsl.resolve import resolve_stream

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Project root
sys.path.append(parent_dir)

from lib.preprocessing import spa_cleaning_by_second, cwt_transform_pywt, freq_adjust, process_eeg_chunk
from lib.eeg_cnn_predictor import EEGCNNPredictor
# import lib.light_control as light


# Load model
model_path = './models/eeg_cnn_model.pth'
predictor = EEGCNNPredictor(model_path, device='cpu')

# After defining the desired channels and before the while loop
print("Looking for an EEG stream...")
streams = resolve_stream('type', 'EEG')
print(streams)

# if len(streams) == 0: and then wait until the stream is found
while len(streams) == 0:
    print("No EEG streams found. Make sure your EEG device is streaming.")
    streams = resolve_stream('type', 'EEG')

# if len(streams) == 0:
#     raise RuntimeError("No EEG streams found. Make sure your EEG device is streaming.")


# Select the first stream (or modify the selection logic as needed)
inlet = StreamInlet(streams[0])
print("EEG stream resolved. Receiving data...")

# ---------------------------------------------
# Configuration
# ---------------------------------------------
srate = 250  # Sampling rate in Hz
block_size = 250  # 1 second of data = 250 samples
refresh_rate = 1.0  # seconds between processing steps

# Rolling buffers (we will NOT delete from these)
all_data_buffer = []  # Will hold samples
all_timestamps = []  # Will hold timestamps

predictions = []

# We’ll base the “once per second” refresh on LSL timestamps
# or you can use local wall-clock time with time.time().
last_process_time = None

print("Starting data acquisition...")
while True:
    # Pull any available chunk
    chunk, ts_chunk = inlet.pull_chunk()

    if chunk:
        # 'chunk' is a list of [sample1, sample2, ...], each sample is [ch1, ch2, ...]
        # Extend our buffer with the new chunk
        # all_data_buffer.extend(chunk)
        all_data_buffer.extend(chunk)  # shape from LSL: list of [ch1, ch2, ...]
        all_timestamps.extend(ts_chunk)  # list of float timestamps

        # if chunk:
        # Extend our buffers
        # all_data_buffer.extend(chunk)  # shape from LSL: list of [ch1, ch2, ...]
        # all_timestamps.extend(ts_chunk)  # list of float timestamps

        if all_timestamps:
            # Current LSL time is the timestamp of the latest sample
            current_lsl_time = all_timestamps[-1]

            # Initialize last_process_time the first time we have data
            if last_process_time is None:
                last_process_time = current_lsl_time

            # Check if at least 1 second has passed since last processing
            # AND we have at least 1 second of data in the buffer
            if (current_lsl_time - last_process_time) >= refresh_rate and len(all_data_buffer) >= block_size:
                # -----------------------------------------------------
                # 1) Extract the LAST 250 samples (latest 1 second)
                # -----------------------------------------------------
                # Each element of all_data_buffer is [channel_1, channel_2, ...].
                # Turn it into a NumPy array of shape (num_samples, num_channels).
                recent_samples = np.array(all_data_buffer[-block_size:])  # shape = (250, n_channels)

                # 1) Frequency adjustments( if needed)
                # shape = (250, n_channels) to shape = (125, n_channels)
                recent_samples = recent_samples[::2]  # Downsample by a factor of 2

                # 2) Band-pass filter
                # handle 1-45 Hz band-pass filter on microvolts
                # Convert from microvolts to volts
                band_pass_data = mne.filter.filter_data(
                    recent_samples.T,
                    sfreq=125,
                    l_freq=1.,
                    h_freq=45.,
                    fir_design='firwin',
                    pad='reflect_limited'
                ).T

                # 2) SPA cleaning
                clean_data = spa_cleaning_by_second(band_pass_data)
                # 3) CWT transform
                power_features = cwt_transform_pywt(clean_data)
                # 4) Predict
                prediction = predictor.predict(power_features)
                print(f"Prediction for segment {prediction['predicted_class_label']}")
                # append the prediction to the list
                predictions.append(prediction['predicted_class_label'])

                if prediction['predicted_class_label'] == 'Active':
                    # light.red_control()
                    # print with timestamp with 2 decimal places
                    print("Active channel detected" + " at " + str(round(current_lsl_time, 2)))
                else:
                    # light.green_control()
                     print("Resting channel detected" + " at " + str(round(current_lsl_time, 2)))
                # Option A: “Hop” exactly 1 second ahead
                last_process_time += refresh_rate

                if len(predictions) == 360:
                    print(predictions)

                # print the len of all_data_buffer
                # print(f"Buffer length: {len(all_data_buffer)}")
    # Brief sleep to avoid busy-wait
    time.sleep(0.01)