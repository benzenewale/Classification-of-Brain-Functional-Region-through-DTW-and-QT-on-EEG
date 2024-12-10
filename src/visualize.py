import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import mne
import seaborn as sns  # For heat mapping
import numpy as np


def plot_clusters(labels, channel_positions):
    """
    Visualize EEG channel clusters on a 2D head map.

    Args:
        labels (list or np.ndarray): Cluster labels for each channel.
        channel_positions (dict): Dictionary of channel names and their 3D positions.
                                  Format: {channel_name: (x, y, z)}.
    """
    # Converting 3D coordinates to 2D (projection to a plane)
    pos_2d = {ch: (coords[0], coords[1]) for ch, coords in channel_positions.items()}

    # Get the channel name and corresponding coordinates
    channel_names = list(pos_2d.keys())
    xy_coords = np.array(list(pos_2d.values()))

    # Ensure labels are in the same order as the channels
    assert len(labels) == len(channel_names), "Labels and channels count mismatch."

    # Creating color maps
    unique_labels = np.unique(labels)
    cmap = get_cmap("tab10", len(unique_labels))  # 使用离散 colormap
    colors = [cmap(label) for label in labels]

    # Scalp mapping
    plt.figure(figsize=(10, 8))
    for idx, (name, coord) in enumerate(zip(channel_names, xy_coords)):
        plt.scatter(coord[0], coord[1], color=colors[idx], s=150, label=f"Cluster {labels[idx]}")
        plt.text(coord[0], coord[1], name, fontsize=10, ha='center', va='center')

    # De-emphasis display legend
    handles, _ = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), [f"Cluster {l}" for l in unique_labels], loc="upper right")

    # Graphic beautification
    plt.title("EEG Channel Clusters", fontsize=16)
    plt.axis("equal")
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    plt.grid(False)
    plt.tight_layout()
    plt.show()


def plot_similarity_matrix(similarity_matrix, channel_names=None, cmap="viridis"):
    """
    Plot the similarity matrix as a heatmap.

    Args:
        similarity_matrix (numpy.ndarray): Similarity matrix (channels x channels).
        channel_names (list or None): Optional, list of channel names to label the axes.
        cmap (str): Colormap to use for the heatmap. Default is "viridis".
    """
    plt.figure(figsize=(10, 8))

    # Using seaborn to draw heat map
    sns.heatmap(similarity_matrix,
                xticklabels=channel_names,
                yticklabels=channel_names,
                cmap=cmap,
                annot=False,
                cbar=True)

    plt.title("Similarity Matrix")
    plt.xlabel("Channels")
    plt.ylabel("Channels")
    plt.tight_layout()
    plt.show()