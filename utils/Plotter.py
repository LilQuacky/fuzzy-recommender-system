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

    def plot_clusters(self, R_scaled, membership, prefix=None, clustering_method="fcm", normalization=None):
        """
        Plot PCA-reduced user clusters. For FCM, also plot membership histogram and heatmap for most uncertain users.

        Parameters:
            R_scaled (np.ndarray): Scaled user-item rating matrix (users x items).
            membership (np.ndarray): Membership matrix (clusters x users).
            prefix (str, optional): Prefix for output filenames.
            clustering_method (str): "fcm" or "kmeans" to determine plot types.
            normalization (str, optional): Name of the normalization method (used for directory structure).
        """
        base_dir = self.output_dir
        if normalization is not None:
            if not base_dir.rstrip(os.sep).endswith('train'):
                base_dir = os.path.join(base_dir, 'train')
            base_dir = os.path.join(base_dir, 'normalization', normalization)
        os.makedirs(base_dir, exist_ok=True)

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(R_scaled)
        cluster_labels = np.argmax(membership, axis=0)

        if clustering_method == "fcm":
            cluster_dir_name = "fuzzy_clusters"
            plot_title = "User Clusters (Fuzzy C-Means)"
        else:
            cluster_dir_name = "hard_clusters"
            plot_title = "User Clusters (K-Means)"

        clusters_dir = os.path.join(base_dir, cluster_dir_name)
        os.makedirs(clusters_dir, exist_ok=True)
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap='Set1', alpha=0.7)
        plt.title(plot_title)
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.colorbar(scatter, label='Cluster Label')
        fname = f"{cluster_dir_name}_pca.png" if prefix is None else f"{cluster_dir_name}_pca_{prefix}.png"
        plt.savefig(os.path.join(clusters_dir, fname))
        if self.show_plots:
            plt.show()
        plt.close()

        if clustering_method == "fcm":
            membership_histogram_dir = os.path.join(base_dir, "membership_histogram")
            os.makedirs(membership_histogram_dir, exist_ok=True)
            max_membership = np.max(membership, axis=0)
            if np.min(max_membership) < 0 or np.max(max_membership) > 1:
                print(f"[WARNING] Membership values out of [0, 1] range: min={np.min(max_membership)}, max={np.max(max_membership)}")
            plt.figure(figsize=(8, 4))
            plt.hist(max_membership, bins='auto', color='skyblue', edgecolor='black')
            plt.title("Distribution of Maximum Membership Values")
            plt.xlabel("Maximum Membership Value")
            plt.ylabel("Frequency")
            plt.xlim(0, 1)
            plt.xticks(np.linspace(0, 1, 11))
            fname = f"membership_histogram.png" if prefix is None else f"membership_histogram_{prefix}.png"
            plt.savefig(os.path.join(membership_histogram_dir, fname))
            plt.close()

            membership_heatmap_dir = os.path.join(base_dir, "membership_heatmap")
            os.makedirs(membership_heatmap_dir, exist_ok=True)
            entropy = -np.sum(membership * np.log(membership + 1e-10), axis=0)
            idx_uncertain = np.argsort(entropy)[-10:]
            plt.figure(figsize=(10, 6))
            sns.heatmap(membership[:, idx_uncertain], cmap='viridis', annot=True)
            plt.title("Heatmap Membership Values for Most Uncertain Users")
            plt.xlabel("User")
            plt.ylabel("Cluster")
            fname = f"membership_heatmap.png" if prefix is None else f"membership_heatmap_{prefix}.png"
            plt.savefig(os.path.join(membership_heatmap_dir, fname))
            if self.show_plots:
                plt.show()
            plt.close()
