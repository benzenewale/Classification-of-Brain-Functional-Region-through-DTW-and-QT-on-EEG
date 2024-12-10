from sklearn.cluster import AgglomerativeClustering


def qt_clustering(similarity_matrix, threshold):
    """
    Perform QT clustering on the similarity matrix.
    Args:
        similarity_matrix (numpy.ndarray): Similarity matrix.
        threshold (float): Threshold for cluster similarity.
    Returns:
        list: Cluster labels for each channel.
    """
    clustering = AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold, linkage='complete'
    )
    labels = clustering.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
    return labels
