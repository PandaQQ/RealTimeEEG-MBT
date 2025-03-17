## Enhancement for V2 receiving data from LSL
> Below is a minimal working example showing how you can:
>	1.	Keep the entire data buffer (i.e. not delete processed samples).
>	2.	Process once every second (the “refresh check”), using only the latest 1 second (250 samples) each time.

This example assumes your device is sampling at 250 Hz. Adapt as necessary for your real sampling rate and channel count.

```
import time
import numpy as np
from pylsl import StreamInlet, resolve_stream

# ----------------------------------------------------------------
# Example placeholders for your own functions/classes
# ----------------------------------------------------------------
def freq_adjust(data_array):
    """Placeholder for frequency adjustments."""
    return data_array

def cwt_transform_pywt(data_array):
    """Placeholder for wavelet transform."""
    # data_array shape = (num_samples, num_channels) if you have multiple channels
    return np.mean(data_array)  # Dummy example

class Predictor:
    def __init__(self):
        pass

    def predict(self, features):
        """Placeholder for ML model prediction."""
        return float(features)  # Dummy example

# ----------------------------------------------------------------
def main():
    # ---------------------------------------------
    # Configuration
    # ---------------------------------------------
    srate = 250        # Sampling rate in Hz
    block_size = 250   # 1 second of data = 250 samples
    refresh_rate = 1.0 # seconds between processing steps

    # Resolve the EEG stream (adjust 'type','EEG' or 'name','YourStreamName')
    print("Resolving LSL stream...")
    streams = resolve_stream('type', 'EEG')
    inlet = StreamInlet(streams[0])

    # Rolling buffers (we will NOT delete from these)
    all_data_buffer = []      # Will hold samples
    all_timestamps = []       # Will hold timestamps

    # Predictor
    predictor = Predictor()

    # We’ll base the “once per second” refresh on LSL timestamps
    # or you can use local wall-clock time with time.time().
    last_process_time = None

    print("Starting data acquisition...")
    while True:
        # Pull any available chunk
        chunk, ts_chunk = inlet.pull_chunk()
        if chunk:
            # Extend our buffers
            all_data_buffer.extend(chunk)       # shape from LSL: list of [ch1, ch2, ...]
            all_timestamps.extend(ts_chunk)     # list of float timestamps

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

                # -----------------------------------------------------
                # 2) Frequency adjustments (if needed)
                # -----------------------------------------------------
                adjusted_data = freq_adjust(recent_samples)

                # -----------------------------------------------------
                # 3) Wavelet transform
                # -----------------------------------------------------
                features = cwt_transform_pywt(adjusted_data)

                # -----------------------------------------------------
                # 4) Prediction
                # -----------------------------------------------------
                prediction = predictor.predict(features)
                print(f"[INFO] Prediction = {prediction:.4f}")

                # -----------------------------------------------------
                # 5) Update last_process_time
                # -----------------------------------------------------
                # Option A: “Hop” exactly 1 second ahead
                last_process_time += refresh_rate

                # Option B: “Sync” to the actual latest sample time
                # last_process_time = current_lsl_time

        # Brief sleep to avoid busy-wait
        time.sleep(0.01)

# ----------------------------------------------------------------
if __name__ == "__main__":
    main()
```

```
How It Works
	1.	No Deletion of Old Samples
	•	Notice we never call del all_data_buffer[:block_size]. We keep all the samples in all_data_buffer and simply look at the last 250 samples each time we process.
	2.	Refresh Check with 1s
	•	We track the time of the last processing event in last_process_time.
	•	Every loop, if (current_lsl_time - last_process_time) >= refresh_rate, we do our 1-second block processing.
	•	Then we either:
	•	“Hop” forward by 1 second (last_process_time += refresh_rate), or
	•	Match the current LSL timestamp (last_process_time = current_lsl_time).
	3.	Collecting Data via pull_chunk()
	•	We continuously read chunks of samples. Each sample is a list [chan1, chan2, ...], appended to all_data_buffer. Timestamps go into all_timestamps.
	•	Since we never remove old data, all_data_buffer will grow over time. For long experiments, consider a more memory-efficient approach (like trimming older data if you only ever need the last few seconds).
	4.	Processing the Last 1 Second
	•	recent_samples = np.array(all_data_buffer[-block_size:]) pulls the most recent 250 samples (one second at 250 Hz).
	5.	Frequency Adjustment
	•	freq_adjust() is a placeholder function where you might apply filtering or resampling.
	6.	Wavelet Transform
	•	cwt_transform_pywt() is a placeholder for your PyWavelets transform.
	7.	Prediction
	•	predictor.predict(...) is where you call your ML model or custom classification logic.
	8.	Sleeping
	•	time.sleep(0.01) prevents this loop from using 100% CPU. Adjust as desired.
```
-----

> If You Need to Prevent Infinite Buffer Growth
> If you truly never need older data, you could occasionally trim the buffer:
```
# After processing:
max_buffer_size = 10 * block_size  # e.g. keep only last 10 seconds
if len(all_data_buffer) > max_buffer_size:
    all_data_buffer = all_data_buffer[-max_buffer_size:]
    all_timestamps  = all_timestamps[-max_buffer_size:]
```

> But this is optional; if you’re only running briefly and don’t mind memory usage, you can keep everything as in the example above.
⸻

> Choosing last_process_time += 1.0 vs last_process_time = current_lsl_time
- If you do last_process_time += refresh_rate, you’ll process at a fixed 1-second cadence no matter whether the data is on time or a bit late. This keeps nice, discrete 1-second steps.
- you do last_process_time = current_lsl_time, you effectively say “whenever the data is at least 1 second newer than the last process time, process immediately and then set the last_process_time to right now.” That might create a slight drift if data arrives slightly slower or faster.
Either approach is valid; pick whichever timing method suits your needs best.