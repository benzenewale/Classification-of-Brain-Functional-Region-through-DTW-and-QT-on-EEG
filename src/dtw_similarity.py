from dtaidistance import dtw
import numpy as np


def compute_dtw_similarity(eeg_data, normalize='minmax'):
    """
    Compute DTW similarity matrix for EEG data with optional normalization.
    Args:
        eeg_data (numpy.ndarray): EEG data of shape (channels, time).
        normalize (str): Normalization method. Options: 'minmax', 'length', None.
    Returns:
        numpy.ndarray: Similarity matrix of shape (channels, channels).
    """
    n_channels = eeg_data.shape[0]
    similarity_matrix = np.zeros((n_channels, n_channels))
    distances = []

    # Compute DTW distances
    for i in range(n_channels):
        for j in range(i, n_channels):
            dist = dtw.distance(eeg_data[i], eeg_data[j])
            distances.append(dist)  # Store distances for normalization
            similarity_matrix[i, j] = dist
            similarity_matrix[j, i] = dist

    # Apply normalization if needed
    if normalize == 'minmax':
        max_dist = np.max(distances)
        min_dist = np.min(distances)
        similarity_matrix = (similarity_matrix - min_dist) / (max_dist - min_dist)
    elif normalize == 'length':
        # Normalize by sequence length (assumes all sequences have the same length)
        seq_length = eeg_data.shape[1]
        similarity_matrix = similarity_matrix / seq_length
    elif normalize == 'softmax':
        # Apply softmax normalization
        exp_matrix = np.exp(-similarity_matrix)  # Use negative to ensure smaller distances get higher weights
        similarity_matrix = exp_matrix / np.sum(exp_matrix, axis=1, keepdims=True)
    elif normalize == 'zscore':
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        similarity_matrix = (similarity_matrix - mean_dist) / std_dist

    # Convert distances to similarity values
    similarity_matrix = 1 / (1 + similarity_matrix)

    return similarity_matrix
