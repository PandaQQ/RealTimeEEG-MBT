import mne
import numpy as np


def spa_cleaning(raw, cutoff=50):
    """
    Perform Second-by-second cleaning using PCA.

    Parameters:
    - raw (mne.io.Raw): The raw EEG data.
    - cutoff (int): Threshold for eigenvalues.

    Returns:
    - mne.io.Raw: Cleaned raw EEG data.
    """
    seg_length = int(raw.info['sfreq'])  # 1-second segments
    n_segments = int(raw.n_times / seg_length)
    data_clean = raw.get_data()

    for j in range(n_segments):
        start = j * seg_length
        end = start + seg_length
        segment = data_clean[:, start:end]
        data_clean[:, start:end] = spa_cleaning_by_second(segment, cutoff)

    return data_clean


def spa_cleaning_by_second(segment, cutoff=50):
    """
    Perform Second-by-second cleaning using PCA.

    Parameters:
    - data (numpy.ndarray): The raw EEG data. Shape: (n_channels, n_times).
    - cutoff (int): Threshold for eigenvalues.

    Returns:
    - numpy.ndarray: Cleaned raw EEG data.
    """

    # Perform PCA
    # mean_centered = segment - np.mean(segment, axis=1, keepdims=True)
    # u, s, vh = np.linalg.svd(mean_centered, full_matrices=False)
    # s[s > cutoff ** 2] = 0  # Threshold eigenvalues
    # cleaned_segment = np.dot(u * s, vh) + np.mean(segment, axis=1, keepdims=True)
    # return cleaned_segment

    from sklearn.decomposition import PCA
    pca = PCA()
    com = pca.fit_transform(np.transpose(segment))
    high_var_mask = pca.explained_variance_ > (cutoff ** 2)
    com[:, high_var_mask] = 0
    reconstructed = pca.inverse_transform(com)
    return np.transpose(reconstructed)


def cwt_transform_pywt(data, frequencies=np.arange(1, 50)):
    """
    Perform Continuous Wavelet Transform (CWT) on the input data using PyWavelets.

    Parameters:
    - data (numpy.ndarray): Input data of shape (n_channels, n_times).
    - frequencies (numpy.ndarray): Array of frequencies for CWT.

    Returns:
    - numpy.ndarray: Transformed data of shape (n_channels, n_frequencies, n_times).
    """
    # temp_seg = data[j, k, :]
    # coef, freqs = pywt.cwt(temp_seg, np.arange(1, 50), 'cmor1-1')
    # freqs = freqs * data.shape[2]
    # data_cwt1[k, :, :, j] = abs(coef)
    frequencies = np.array([1., 1.25, 1.5625, 1.953125, 2.44140625,
                            3.05175781, 3.81469727, 4.76837158, 5.96046448, 7.4505806,
                            9.31322575, 11.64153218, 14.55191523, 18.18989404, 22.73736754,
                            28.42170943, 35.52713679, 44.40892099, 55.51115123, 69.38893904])

    import pywt
    n_channels, n_times = data.shape
    n_frequencies = len(frequencies)
    cwt_data = np.empty((n_channels, n_frequencies, n_times))

    for ch_idx in range(n_channels):
        signal = data[ch_idx, :]
        # coef, freqs = pywt.cwt(signal, frequencies, wavelet='cmor')
        coef, freqs = pywt.cwt(signal, frequencies, wavelet='cmor1-1')
        cwt_data[ch_idx, :, :] = abs(coef)
    return cwt_data


def freq_adjust(data, from_freq=250, to_freq=125):
    """
    Adjust the frequency of the input data.
    This function is useful for down sampling the data.
    """
    from scipy.signal import resample
    n_channels, n_times = data.shape
    new_times = int(n_times * to_freq / from_freq)
    new_data = np.empty((n_channels, new_times))
    for ch_idx in range(n_channels):
        new_data[ch_idx, :] = resample(data[ch_idx, :], new_times)
    return new_data


def process_eeg_chunk(chunk, srate=250):
    # Convert from microvolts to volts if needed
    data_in_volts = chunk * 1e6  # chunk is shape (250, 24) for example

    # Transpose to (n_channels, n_samples) for MNE functions
    data_in_volts = data_in_volts.T

    # Apply a band-pass filter from 1 to 45 Hz
    filtered_data = mne.filter.filter_data(
        data_in_volts,
        sfreq=srate,
        l_freq=1.,
        h_freq=45.,
        method='fir',
        fir_design='firwin'
    )

    return filtered_data.T  # Transpose back to (n_samples, n_channels)
