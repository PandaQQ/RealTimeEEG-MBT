# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 14:33:22 2025

@author: guang
"""

import pylsl
import numpy as np
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import time

# Parameters
SAMPLING_RATE = 250  # Hz
BUFFER_DURATION = 10.0  # seconds
NUM_CHANNELS = 24
BUFFER_SIZE = int(SAMPLING_RATE * BUFFER_DURATION)

# Initialize data buffer
buffer = deque(maxlen=BUFFER_SIZE)
timestamps = deque(maxlen=BUFFER_SIZE)

# Set up the plot
plt.ion()  # Interactive mode on
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('Real-time EEG Visualization (Last 1 Second)')

# Initialize lines for each channel
lines = []
colors = plt.cm.viridis(np.linspace(0, 1, 24))  # Using viridis colormap for distinct colors

for i in range(NUM_CHANNELS):
    line, = axes[0].plot([], [], color=colors[i], lw=1)
    lines.append(line)


axes[0].set_ylim(-500, 500)  # Adjust based on your EEG data range
axes[0].set_xlim(-BUFFER_DURATION, 0)  # Adjust based on your EEG data range
axes[0].grid(True)
axes[0].set_xlabel('Time (s)')

line_fft, = axes[1].plot([],[],lw=1)
lines.append(line_fft)
axes[1].set_ylim(-10, 20)  # Adjust based on your EEG data range
axes[1].set_xlim(0, 20)  # Adjust based on your EEG data range
axes[1].set_xlabel('Frequency (Hz)')


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

def update_plot(frame):
    # Pull new samples
    samples, timestamps = inlet.pull_chunk()
    
    if samples:
        # Add new samples to buffer
        buffer.extend(samples)
        
        # Update plot for each channel
        for i in range(NUM_CHANNELS):
            # Get channel data (last BUFFER_SIZE samples)
            channel_data = [sample[i] for sample in buffer]
            
            # If we don't have enough data yet, pad with zeros
            if len(channel_data) < BUFFER_SIZE:
                padded_data = np.zeros(BUFFER_SIZE)
                padded_data[-len(channel_data):] = channel_data
                channel_data = padded_data
            else:
                channel_data = channel_data[-BUFFER_SIZE:]
                
            if i == 9:
                time_series = np.array(channel_data)
                temp = np.abs(np.fft.fft(time_series))
                temp = temp[1:int(BUFFER_DURATION*60)]*2/len(temp)
                temp_t = np.linspace(0,60,len(temp))
                lines[-1].set_data(temp_t,temp)
            
            # Update line data
            lines[i].set_data(time_axis, channel_data-np.mean(channel_data))
            #
    
    return lines

# Create animation
ani = FuncAnimation(fig, update_plot, interval=50, blit=True)

plt.show(block=True)