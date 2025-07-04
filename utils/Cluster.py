import numpy as np
import skfuzzy as fuzz 
import sklearn.cluster


class Clusterer:
    """
    A class for performing fuzzy c-means clustering and making predictions using cluster centroids and memberships.
    """
    def fcm_cluster(self, X, n_clusters, m, error, max_iter, seed):
        """
        Perform fuzzy c-means clustering on the input data.

        Parameters:
            X (np.ndarray): Data matrix (users x items).
            n_clusters (int): Number of clusters.
            m (float): Fuzziness parameter.
            error (float): Stopping criterion; stop if improvement is less than this value.
            max_iter (int): Maximum number of iterations.
            seed (int): Random seed for reproducibility.

        Returns:
            cntr (np.ndarray): Cluster centers.
            u (np.ndarray): Final fuzzy partitioned matrix (membership matrix).
        """
        cntr, u, _, _, _, _, _ = fuzz.cmeans(
            data=X.T, c=n_clusters, m=m, error=error, maxiter=max_iter, seed=seed
        )
        return cntr, u
    
    def kmeans_cluster(self, X, n_clusters, seed):
        """
        Perform K-means clustering on the input data.

        Parameters:
            X (np.ndarray): Data matrix (users x items).
            n_clusters (int): Number of clusters.
            seed (int): Random seed for reproducibility.

        Returns:
            cntr (np.ndarray): Cluster centers.
            u (np.ndarray): Hard partitioned matrix (one-hot membership matrix).
        """
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=seed)
        labels = kmeans.fit_predict(X)
        cntr = kmeans.cluster_centers_
        u = np.zeros((n_clusters, X.shape[0]))
        u[labels, np.arange(X.shape[0])] = 1
        return cntr, u

    def predict(self, cntr, membership):
        """
        Predict user-item ratings using cluster centroids and membership matrix.

        Parameters:
            cntr (np.ndarray): Cluster centers (clusters x items).
            membership (np.ndarray): Membership matrix (clusters x users).

        Returns:
            pred (np.ndarray): Predicted ratings matrix (users x items).
        """
        n_clusters, n_users = membership.shape
        n_items = cntr.shape[1]
        pred = np.zeros((n_users, n_items))
        for c in range(n_clusters):
            weights = membership[c, :]
            pred += np.outer(weights, cntr[c, :])
        return pred

    def predict_test(self, R_test_scaled, cntr, m, error, max_iter):
        """
        Predict the membership matrix for test data using trained cluster centers.

        Parameters:
            R_test_scaled (np.ndarray): Scaled test data matrix (users x items).
            cntr (np.ndarray): Trained cluster centers.
            m (float): Fuzziness parameter.
            error (float): Stopping criterion for prediction.
            max_iter (int): Maximum number of iterations for prediction.

        Returns:
            u_test (np.ndarray): Predicted membership matrix for test data.
        """
        u_test, _, _, _, _, _ = fuzz.cmeans_predict(
            R_test_scaled.T, cntr, m, error=error, maxiter=max_iter)
        return u_test
