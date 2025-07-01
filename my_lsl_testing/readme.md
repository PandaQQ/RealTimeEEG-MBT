# Real-Time EEG Streaming and CNN Prediction System

## Overview

This project implements a real-time EEG data streaming and analysis system using Lab Streaming Layer (LSL) with a CNN-based classifier for brain state prediction. The system can stream EEG data from EEGLAB files and perform real-time predictions to classify brain activity states as "Active" or "Resting".

## Project Structure

```
RealTimeEEG-mBot/
├── my_lsl_testing/                    # Main LSL testing scripts
│   ├── eeglab_lsl_cnn_testing.py     # Real-time EEG visualization with CNN predictions
│   └── eeglab_streaming_testing.py   # EEG data streaming from EEGLAB files
├── lib/                               # Core library modules
│   ├── eeg_cnn_predictor.py          # CNN model predictor class
│   ├── preprocessing.py              # EEG preprocessing functions
│   └── light_control.py              # External device control
├── models/                            # Trained models
│   └── eeg_cnn_model.pth            # Pre-trained CNN model
├── data/                             # EEG data files
│   ├── wireless_raw.set             # Main EEGLAB dataset
│   └── wireless_raw_*.set           # Additional datasets
└── training/                         # Training related files
    └── traning_cwt.py               # CNN training script
```

## Features

### Real-Time EEG Processing
- **Multi-channel EEG streaming** (24 channels at 250 Hz)
- **Real-time visualization** with 4 synchronized plots:
  - Raw EEG signals from all channels
  - FFT power spectrum analysis
  - Continuous Wavelet Transform (CWT) visualization
  - Prediction results timeline

### Signal Processing Pipeline
1. **Band-pass filtering** (1-45 Hz)
2. **SPA (Second-by-second PCA) cleaning** for artifact removal
3. **Continuous Wavelet Transform** for time-frequency analysis
4. **CNN-based classification** for brain state prediction

### CNN Architecture
- **Input**: CWT power features (125 time points × 20 frequencies × 24 channels)
- **Architecture**: 
  - Conv2D layer (24→50 channels)
  - Batch normalization
  - Max pooling
  - Fully connected layers (32 neurons)
  - Output: 2 classes (Active/Resting)

## Installation

### Prerequisites
- Python 3.8+
- PyTorch
- MNE-Python
- Lab Streaming Layer (LSL)

### Required Dependencies
```bash
pip install torch torchvision
pip install mne
pip install pylsl
pip install matplotlib
pip install numpy scipy
pip install scikit-learn
pip install pywavelets
pip install eeglabio
```

## Usage

### 1. Streaming EEG Data

To stream EEG data from an EEGLAB file:

```python
python eeglab_streaming_testing.py
```

This script:
- Loads EEG data from `../data/wireless_raw.set`
- Resamples to 250 Hz
- Creates an LSL stream named 'EEG_Stream'
- Streams data in real-time with proper timing

### 2. Real-Time Analysis and Prediction

To run the complete real-time analysis system:

```python
python eeglab_lsl_cnn_testing.py
```

This script:
- Connects to the EEG LSL stream
- Performs real-time signal processing
- Makes predictions every 1 second
- Displays live visualization with 4 subplots

## Key Components

### EEGCNNPredictor Class
```python
from lib.eeg_cnn_predictor import EEGCNNPredictor

# Initialize predictor
predictor = EEGCNNPredictor('../models/eeg_cnn_model.pth', device='cpu')

# Make prediction
result = predictor.predict(cwt_features)
print(f"Predicted state: {result['predicted_class_label']}")
```

### Preprocessing Functions
```python
from lib.preprocessing import spa_cleaning_by_second, cwt_transform_pywt

# Clean EEG data
clean_data = spa_cleaning_by_second(raw_eeg_segment)

# Extract CWT features
cwt_features = cwt_transform_pywt(clean_data)
```

## Configuration Parameters

### Data Acquisition
- **Sampling Rate**: 250 Hz
- **Channels**: 24 EEG channels
- **Buffer Duration**: 10 seconds
- **Prediction Interval**: 1 second

### Signal Processing
- **Band-pass Filter**: 1-45 Hz
- **SPA Cutoff**: 50 (eigenvalue threshold)
- **CWT Frequencies**: 20 logarithmically spaced frequencies (1-69 Hz)
- **Downsampling**: 250 Hz → 125 Hz for CNN input

### CNN Model
- **Input Shape**: (125, 20, 24) - (time, frequency, channels)
- **Classes**: 2 (Active=1, Resting=0)
- **Device**: CPU (configurable to GPU)

## Real-Time Visualization

The system provides four synchronized real-time plots:

1. **EEG Channels Plot**: Raw signals from all 24 channels (last 10 seconds)
2. **FFT Plot**: Power spectrum of channel 10 (0-20 Hz)
3. **CWT Plot**: Time-frequency representation (averaged across channels)
4. **Prediction Plot**: Binary prediction timeline (last 60 predictions)

## Performance Considerations

### Processing Pipeline Timing
- **Data Buffer**: 10-second sliding window
- **Prediction Window**: 1-second segments
- **Update Rate**: 50ms for visualization
- **Processing Latency**: ~100-200ms per prediction

### Memory Management
- Circular buffers for efficient memory usage
- Automatic buffer size management
- Real-time garbage collection friendly

## Troubleshooting

### Common Issues

1. **No EEG Stream Found**
   - Ensure `eeglab_streaming_testing.py` is running first
   - Check LSL stream name matches

2. **Shape Mismatch Errors**
   - Verify CWT output shape is (125, 20, 24)
   - Check downsampling from 250Hz to 125Hz

3. **Model Loading Errors**
   - Ensure model path `../models/eeg_cnn_model.pth` exists
   - Check PyTorch version compatibility

4. **Filtering Errors**
   - Reduce filter length for short data segments
   - Ensure minimum 1 second of data before processing

## File Descriptions

### Core Scripts
- **`eeglab_lsl_cnn_testing.py`**: Main real-time analysis application
- **`eeglab_streaming_testing.py`**: EEG data streaming utility

### Library Modules
- **`eeg_cnn_predictor.py`**: CNN model wrapper for predictions
- **`preprocessing.py`**: Signal processing and feature extraction
- **`light_control.py`**: External device control interface

## Future Enhancements

- [ ] GPU acceleration for real-time processing
- [ ] Multi-model ensemble predictions
- [ ] Advanced artifact rejection algorithms
- [ ] Real-time parameter tuning interface
- [ ] Integration with external devices (mBot, lighting systems)
- [ ] Data logging and analysis tools

## Authors

**PandaQQ** - Initial development (July 2025)

## Dependencies Version Info

The project requires specific versions of key packages for optimal performance:
- PyTorch (for CNN inference)
- MNE-Python (for EEG processing)
- PyLSL (for real-time streaming)
- PyWavelets (for CWT analysis)
- Matplotlib (for visualization)

See `requirements.txt` for complete dependency list.
