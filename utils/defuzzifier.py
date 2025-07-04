import numpy as np

def defuzzify_maximum(membership):
    """
    Defuzzification by Maximum: assign each user to the cluster with the highest membership value.
    Parameters:
        membership (np.ndarray): Membership matrix (clusters x users)
    Returns:
        cluster_assignments (np.ndarray): Array of cluster indices (len = n_users)
    """
    return np.argmax(membership, axis=0)

def defuzzify_cog(membership):
    """
    Defuzzification by Center of Gravity (COG): assign each user to a cluster index as the weighted average of cluster indices.
    Parameters:
        membership (np.ndarray): Membership matrix (clusters x users)
    Returns:
        cog_assignments (np.ndarray): Array of float cluster indices (len = n_users)
    """
    cluster_indices = np.arange(membership.shape[0]).reshape(-1, 1)
    cog = np.sum(membership * cluster_indices, axis=0) / np.sum(membership, axis=0)
    return cog 
