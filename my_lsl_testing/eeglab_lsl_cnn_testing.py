"""
Created on Wed July 1st 11:00:00 2025
@author: PandaQQ
"""

import pylsl
import numpy as np
import matplotlib
import sys
import os
import mne

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time

# Add the project root directory to the Python path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # Project root
sys.path.append(parent_dir)

from lib.preprocessing import spa_cleaning_by_second, cwt_transform_pywt
from lib.eeg_cnn_predictor import EEGCNNPredictor

# Load model
model_path = '../models/eeg_cnn_model.pth'
predictor = EEGCNNPredictor(model_path, device='cpu')

# Parameters
SAMPLING_RATE = 250  # Hz
BUFFER_DURATION = 10.0  # seconds
NUM_CHANNELS = 24
BUFFER_SIZE = int(SAMPLING_RATE * BUFFER_DURATION)
PREDICTION_INTERVAL = 1.0  # seconds between predictions

# Initialize data buffers
buffer = deque(maxlen=BUFFER_SIZE)
timestamps_buffer = deque(maxlen=BUFFER_SIZE)
prediction_history = deque(maxlen=60)  # Keep last 60 predictions
prediction_binary = deque(maxlen=60)  # 1 for Active, 0 for Rest
cwt_data = None

# Processing timing
last_prediction_time = None

# Set up the plot with 4 subplots
plt.ion()  # Interactive mode on
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Real-time EEG Visualization with CWT and Predictions')

# Flatten axes for easier indexing
ax_eeg = axes[0, 0]
ax_fft = axes[0, 1]
ax_cwt = axes[1, 0]
ax_pred = axes[1, 1]

# Initialize lines for EEG channels
lines_eeg = []
colors = plt.cm.viridis(np.linspace(0, 1, NUM_CHANNELS))

for i in range(NUM_CHANNELS):
    line, = ax_eeg.plot([], [], color=colors[i], lw=1, alpha=0.7)
    lines_eeg.append(line)

ax_eeg.set_ylim(-500, 500)
ax_eeg.set_xlim(-BUFFER_DURATION, 0)
ax_eeg.grid(True)
ax_eeg.set_xlabel('Time (s)')
ax_eeg.set_ylabel('Amplitude (µV)')
ax_eeg.set_title('EEG Channels (Last 10s)')

# Initialize FFT plot
line_fft, = ax_fft.plot([], [], lw=1)
ax_fft.set_ylim(-10, 20)
ax_fft.set_xlim(0, 20)
ax_fft.set_xlabel('Frequency (Hz)')
ax_fft.set_ylabel('Power (dB)')
ax_fft.set_title('FFT - Channel 10')
ax_fft.grid(True)

# Initialize CWT plot
cwt_im = None
ax_cwt.set_title('CWT Transform (Average Power Features)')
ax_cwt.set_xlabel('Time Bins')
ax_cwt.set_ylabel('Frequency Bins')

# Initialize prediction plot
line_pred, = ax_pred.plot([], [], 'o-', linewidth=2, markersize=6)
ax_pred.set_ylim(-0.1, 1.1)
ax_pred.set_xlabel('Time (predictions)')
ax_pred.set_ylabel('State (1=Active, 0=Rest)')
ax_pred.set_title('Prediction Results')
ax_pred.grid(True)

plt.tight_layout()

# Find EEG stream
print("Looking for an EEG stream...")
streams = pylsl.resolve.resolve_stream('type', 'EEG')
if not streams:
    raise RuntimeError("No EEG stream found")

inlet = pylsl.StreamInlet(streams[0])
print(f"Connected to stream: {streams[0].name()}")

# Get stream info
info = inlet.info()
fs = info.nominal_srate()
print(f"Sampling rate: {fs} Hz")
print(f"Number of channels: {info.channel_count()}")

# Initialize time axis
time_axis = np.linspace(-BUFFER_DURATION, 0, BUFFER_SIZE)


def process_eeg_prediction():
    """Process EEG data for CWT and prediction"""
    global cwt_data, last_prediction_time

    if len(buffer) < 250:  # Need at least 1 second of data
        return

    # Extract last 250 samples (1 second)
    recent_samples = np.array(list(buffer)[-250:])

    # Downsample by factor of 2 (250Hz -> 125Hz)
    recent_samples = recent_samples[::2]  # Shape: (125, n_channels)

    # Band-pass filter with shorter filter length
    try:
        band_pass_data = mne.filter.filter_data(
            recent_samples.T,
            sfreq=125,
            l_freq=1.,
            h_freq=45.,
            fir_design='firwin',
            pad='reflect_limited',
            verbose=False
        ).T
    except Exception as e:
        print(f"Filtering error: {e}")
        return

    # SPA cleaning
    try:
        clean_data = spa_cleaning_by_second(band_pass_data)
        print(f"Clean data shape: {clean_data.shape}")
    except Exception as e:
        print(f"SPA cleaning error: {e}")
        return

    # CWT transform
    try:
        power_features = cwt_transform_pywt(clean_data)
        power_features_for_prediction = power_features.T

        # Ensure correct shape for model: (125, 20, 24)
        if power_features.shape != (125, 20, 24):
            print(f"Warning: CWT shape {power_features.shape} != expected (125, 20, 24)")
            # Try to fix common shape issues
            if len(power_features.shape) == 3:
                # Transpose if dimensions are in wrong order
                if power_features.shape == (24, 20, 125):
                    power_features = power_features.transpose(2, 1, 0)
                elif power_features.shape == (20, 125, 24):
                    power_features = power_features.transpose(1, 0, 2)

            print(f"Corrected CWT shape: {power_features.shape}")

        # Store for visualization (average across channels for 2D display)
        cwt_data = np.mean(power_features, axis=2)  # Average across channels -> (125, 20)

    except Exception as e:
        print(f"CWT transform error: {e}")
        return

    # Prediction - Use the raw CWT output directly
    try:
        print("Making prediction...")
        print(f"Power features shape for prediction: {power_features_for_prediction.shape}")

        # Power
        # features
        # shape
        # for prediction: (1, 125, 20, 24)
        #  (n_channels, n_frequencies, n_times).

        # (24, 20, 125) = power_features_for_prediction.shape[2:5]

        prediction = predictor.predict(power_features_for_prediction)
        pred_label = prediction['predicted_class_label']

        # Store prediction
        prediction_history.append(pred_label)
        prediction_binary.append(1 if pred_label == 'Active' else 0)

        print(f"Prediction: {pred_label}")

    except Exception as e:
        print(f"Prediction error: {e}")
        return

    last_prediction_time = time.time()


def update_plot(frame):
    global last_prediction_time, cwt_im

    # Pull new samples
    samples, timestamps = inlet.pull_chunk()

    if samples:
        # Add new samples to buffer
        buffer.extend(samples)
        timestamps_buffer.extend(timestamps)

        # Check if it's time for prediction
        current_time = time.time()
        if (last_prediction_time is None or
                current_time - last_prediction_time >= PREDICTION_INTERVAL):
            try:
                process_eeg_prediction()
            except Exception as e:
                print(f"Processing error: {e}")

        # Update EEG plot for each channel
        for i in range(NUM_CHANNELS):
            channel_data = [sample[i] for sample in buffer]

            if len(channel_data) < BUFFER_SIZE:
                padded_data = np.zeros(BUFFER_SIZE)
                padded_data[-len(channel_data):] = channel_data
                channel_data = padded_data
            else:
                channel_data = channel_data[-BUFFER_SIZE:]

            # Update EEG line data
            lines_eeg[i].set_data(time_axis, np.array(channel_data) - np.mean(channel_data))

            # Update FFT for channel 10
            if i == 9:
                time_series = np.array(channel_data)
                fft_data = np.abs(np.fft.fft(time_series))
                fft_data = fft_data[1:int(BUFFER_DURATION * 60)] * 2 / len(fft_data)
                freq_axis = np.linspace(0, 60, len(fft_data))
                line_fft.set_data(freq_axis, fft_data)

        # Update CWT plot
        if cwt_data is not None:
            ax_cwt.clear()
            im = ax_cwt.imshow(cwt_data.T, aspect='auto', origin='lower', cmap='viridis')
            ax_cwt.set_title('CWT Transform (Average Power Features)')
            ax_cwt.set_xlabel('Time Bins')
            ax_cwt.set_ylabel('Frequency Bins')

            # Add colorbar if not exists
            if not hasattr(ax_cwt, 'colorbar_added'):
                plt.colorbar(im, ax=ax_cwt, fraction=0.046, pad=0.04)
                ax_cwt.colorbar_added = True

        # Update prediction plot
        if len(prediction_binary) > 0:
            pred_time_axis = np.arange(len(prediction_binary))
            line_pred.set_data(pred_time_axis, list(prediction_binary))

            # Adjust x-axis to show last 30 predictions
            if len(prediction_binary) > 30:
                ax_pred.set_xlim(len(prediction_binary) - 30, len(prediction_binary))
            else:
                ax_pred.set_xlim(0, max(30, len(prediction_binary)))

    return lines_eeg + [line_fft, line_pred]


# Create animation
ani = FuncAnimation(fig, update_plot, interval=50, blit=False)
plt.show(block=True)