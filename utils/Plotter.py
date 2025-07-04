import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
class Plotter:
    """
    A class for visualizing clustering results and normalization effects in fuzzy recommender systems.
    """
    def __init__(self, output_dir="output/images", show_plots=False):
        """
        Initialize the Plotter.

        Parameters:
            output_dir (str): Directory to save output images.
            show_plots (bool): Whether to display plots interactively.
        """
        self.output_dir = output_dir
        self.show_plots = show_plots
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_clusters(self, R_scaled, membership, prefix=None):
        """
        Plot PCA-reduced user clusters, membership histogram, and heatmap for most uncertain users.

        Parameters:
            R_scaled (np.ndarray): Scaled user-item rating matrix (users x items).
            membership (np.ndarray): Membership matrix (clusters x users).
            prefix (str, optional): Prefix for output filenames.
        """
        os.makedirs(self.output_dir, exist_ok=True)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(R_scaled)
        cluster_labels = np.argmax(membership, axis=0)

        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap='Set1', alpha=0.7)
        plt.title("User Clusters (Fuzzy C-Means)")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter, label='Cluster Label')
        fname = f"fuzzy_clusters_pca.png" if prefix is None else f"fuzzy_clusters_pca_{prefix}.png"
        plt.savefig(os.path.join(self.output_dir, fname))
        if self.show_plots:
            plt.show()

        plt.figure(figsize=(8, 4))
        max_membership = np.max(membership, axis=0)
        plt.hist(max_membership, bins='auto', color='skyblue', edgecolor='black')
        plt.title("Distribution of Maximum Membership Values")
        plt.xlabel("Maximum Membership Value")
        plt.ylabel("Frequency")
        fname = f"membership_histogram.png" if prefix is None else f"membership_histogram_{prefix}.png"
        plt.savefig(os.path.join(self.output_dir, fname))

        entropy = -np.sum(membership * np.log(membership + 1e-10), axis=0)
        idx_uncertain = np.argsort(entropy)[-10:]
        plt.figure(figsize=(10, 6))
        sns.heatmap(membership[:, idx_uncertain], cmap='viridis', annot=True)
        plt.title("Heatmap Membership Values for Most Uncertain Users")
        plt.xlabel("User")
        plt.ylabel("Cluster")
        fname = f"membership_heatmap.png" if prefix is None else f"membership_heatmap_{prefix}.png"
        plt.savefig(os.path.join(self.output_dir, fname))
        if self.show_plots:
            plt.show()

    def plot_single_normalization(self, R_norm, membership, norm_name):
        """
        Plot PCA, max membership histogram, and data distribution for a single normalization method.

        Parameters:
            R_norm (np.ndarray): Normalized user-item rating matrix (users x items).
            membership (np.ndarray): Membership matrix (clusters x users).
            norm_name (str): Name of the normalization method (used in plot titles and filenames).
        """
        os.makedirs(self.output_dir, exist_ok=True)
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(R_norm)
        cluster_labels = np.argmax(membership, axis=0)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap='Set1', alpha=0.7)
        plt.title(f"PCA - {norm_name}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")

        plt.subplot(1, 3, 2)
        max_membership = np.max(membership, axis=0)
        plt.hist(max_membership, bins='auto', alpha=0.7)
        plt.title(f"Max Membership - {norm_name}")
        plt.xlabel("Max Membership")
        plt.ylabel("Frequency")

        plt.subplot(1, 3, 3)
        plt.hist(R_norm.flatten(), bins='auto', alpha=0.7)
        plt.title(f"Data Distribution - {norm_name}")
        plt.yscale('log')
        plt.xlabel("Rating Value")
        plt.ylabel("Frequency")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"comparison_{norm_name}.png"), dpi=300, bbox_inches='tight')
        if self.show_plots:
            plt.show()
