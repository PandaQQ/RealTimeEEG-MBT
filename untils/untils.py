"""
I want to read the file wireless_raw.fdt/ wireless_raw.set with MNE
and separate them as 80% (48) and 20%(12) with 60 segments for each.
and save as wireless_raw_8.fdt/ wireless_raw_8.set and wireless_raw_2.fdt/ wireless_raw_2.set

for j in range(60):
    # Relaxation segment
    start = lats[j * 2]
    end = start + int(30 * seg_length)
    relax_segment = raw.get_data(start=start, stop=end)
    relax_segment = relax_segment.reshape((-1, raw.info['nchan'], seg_length))
    data_relax.append(relax_segment)

    # Calculation segment
    start = lats[j * 2 + 1]
    end = lats[j * 2 + 2]
    calcu_segment = raw.get_data(start=start, stop=end)
    n_samples = calcu_segment.shape[1] // seg_length * seg_length
    calcu_segment = calcu_segment[:, :n_samples]
    calcu_segment = calcu_segment.reshape((-1, raw.info['nchan'], seg_length))
    data_calcu.append(calcu_segment)
"""

import mne
import numpy as np
import random
import os


def split_eeg_data(file_path, train_ratio=0.8, random_seed=42):
    """
    Load EEG data from a .set file, split it into training and testing sets,
    and save them as separate files.

    Parameters:
    -----------
    file_path : str
        Path to the EEGLAB .set file
    train_ratio : float
        Ratio of data to use for training (default: 0.8)
    random_seed : int
        Random seed for reproducibility

    Returns:
    --------
    train_file_path : str
        Path to the saved training data file
    test_file_path : str
        Path to the saved testing data file
    """
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load the data
    raw = mne.io.read_raw_eeglab(file_path, preload=True)

    # Extract events
    events, event_id = mne.events_from_annotations(raw)
    lats = [event[0] for event in events if event[2] in [2, 3]]

    # Calculate number of segments for training and testing
    total_segments = 60
    train_segments = int(total_segments * train_ratio)
    test_segments = total_segments - train_segments

    # Generate random indices for train/test split
    all_indices = list(range(total_segments))
    train_indices = sorted(random.sample(all_indices, train_segments))
    test_indices = sorted(list(set(all_indices) - set(train_indices)))

    # Create two separate raw objects
    train_raw = raw.copy()
    test_raw = raw.copy()

    # Get the original event descriptions
    orig_descriptions = raw.annotations.description
    orig_onsets = raw.annotations.onset

    # Create new annotations for training data
    train_onsets = []
    train_descriptions = []

    # Create new annotations for testing data
    test_onsets = []
    test_descriptions = []

    # Split annotations based on segment indices
    for i, (onset, desc) in enumerate(zip(orig_onsets, orig_descriptions)):
        event_idx = i // 2  # Each segment has 2 events
        if event_idx in train_indices and event_idx < total_segments:
            train_onsets.append(onset)
            train_descriptions.append(desc)
        elif event_idx in test_indices and event_idx < total_segments:
            test_onsets.append(onset)
            test_descriptions.append(desc)

    # Set annotations for training data
    train_raw.set_annotations(mne.Annotations(
        onset=train_onsets,
        duration=[0] * len(train_onsets),
        description=train_descriptions
    ))

    # Set annotations for testing data
    test_raw.set_annotations(mne.Annotations(
        onset=test_onsets,
        duration=[0] * len(test_onsets),
        description=test_descriptions
    ))

    # Extract file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    file_dir = os.path.dirname(file_path)

    # Define output file paths
    train_file_path = os.path.join(file_dir, f"{file_name}_8.set")
    test_file_path = os.path.join(file_dir, f"{file_name}_2.set")

    # Save the files
    train_raw.save(train_file_path, overwrite=True)
    test_raw.save(test_file_path, overwrite=True)

    print(f"Training data ({train_segments} segments) saved to: {train_file_path}")
    print(f"Testing data ({test_segments} segments) saved to: {test_file_path}")

    return train_file_path, test_file_path


# Example usage:
train_file, test_file = split_eeg_data("../data/wireless_raw.set", train_ratio=0.8, random_seed=42)
