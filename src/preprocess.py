import numpy as np


def preprocess_eeg_data(eeg_data, trail_need=0, num_channels=32, drop_time=3, sampling_rate=128):
    """
    Preprocess EEG data by selecting specific channels and dropping initial time points.

    Args:
        eeg_data (numpy.ndarray): Raw EEG data, shape (trials, channels, time).
        trail_need (int): Index of trails that need to be processed (default=0).
        num_channels (int): Number of channels to keep (default=32).
        drop_time (int): Time in seconds to drop at the beginning (default=3 seconds).
        sampling_rate (int): Sampling rate of the EEG data in Hz (default=128 Hz).

    Returns:
        numpy.ndarray: Preprocessed EEG data, shape (trials, num_channels, new_time).
    """
    # Calculate the number of time points to drop
    drop_points = drop_time * sampling_rate

    # Select the first `num_channels` and drop the first `drop_points` time points
    preprocessed_data = eeg_data[trail_need, :num_channels, drop_points:]

    return preprocessed_data
