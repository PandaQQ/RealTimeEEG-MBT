import sys
import os
import numpy as np
from pylsl import StreamInlet, resolve_streams
import mne
import matplotlib.pyplot as plt
from pylsl.resolve import resolve_stream

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Project root
sys.path.append(parent_dir)

from lib.preprocessing import spa_cleaning_by_second, cwt_transform_pywt, freq_adjust
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

# Get stream info to extract sampling frequency and channel information
# info = inlet.info()
sfreq = 250  # info.nominal_srate()
n_channels = 24  # info.channel_count()

# Define chunk duration in seconds and calculate the number of samples per chunk
chunk_duration = 1.0  # seconds
chunk_samples = int(sfreq * chunk_duration)
predictions = []

while True:
    # Initialize lists to store samples and timestamps
    samples = []
    timestamps = []
    samples_collected = 0

    # Collect data until we have the desired number of samples
    while samples_collected < chunk_samples:
        # Pull a chunk of data from the inlet
        chunk, ts = inlet.pull_chunk()
        if ts:
            samples.append(chunk)
            timestamps.extend(ts)
            samples_collected += len(chunk)

    # Check if any data was collected
    if not samples:
        continue

    # Concatenate samples and transpose to get shape (n_total_channels, n_times)
    data_chunk_full = np.concatenate(samples, axis=0).T

    # Ensure the data chunk has the correct number of samples
    if data_chunk_full.shape[1] != chunk_samples:
        print(f"Data chunk has {data_chunk_full.shape[1]} samples, expected {chunk_samples}. Skipping this chunk.")
        continue

    # Select only the desired channels
    data_chunk = data_chunk_full

    # Print the shape of the data chunk
    print(f"Data chunk shape: {data_chunk.shape}")

    adjust_data_chunk = freq_adjust(data_chunk, from_freq=250, to_freq=125)
    # Perform SPA Cleaning
    clean_data = spa_cleaning_by_second(adjust_data_chunk)
    # Perform time-frequency decomposition using Morlet wavelets
    power = cwt_transform_pywt(clean_data)

    # Make a prediction
    prediction = predictor.predict(power)
    print(f"Prediction for segment {prediction['predicted_class_label']}")
    predictions.append(prediction['predicted_class_label'])

    if prediction['predicted_class_label'] == 'Active':
        # light.red_control()
        print("Active channel detected")
    else:
        # light.green_control()
        print("Resting channel detected")

    if len(predictions) > 48*30:
        print(predictions)

print(predictions)
