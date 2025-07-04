import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr

class Evaluator:
    """
    A class for evaluating predicted ratings and handling denormalization of predictions.
    """
    def denormalize(self, pred_norm, R):
        """
        Denormalize predicted ratings by adding back the user means.

        Parameters:
            pred_norm (np.ndarray or pd.DataFrame): Normalized predicted ratings (users x items).
            R (pd.DataFrame): Original user-item rating matrix (to compute user means).

        Returns:
            np.ndarray: Denormalized predicted ratings (users x items).
        """
        means = R.mean(axis=1).values.reshape(-1, 1)
        return pred_norm + means

    def evaluate(self, y_true, y_pred):
        """
        Evaluate predictions using RMSE and MAE, ignoring NaN values in the ground truth or in the prediction.

        Parameters:
            y_true (array-like): Ground truth ratings (can contain NaNs).
            y_pred (array-like): Predicted ratings.

        Returns:
            tuple: (rmse, mae) where rmse is root mean squared error and mae is mean absolute error.
        """
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        return np.sqrt(mse), mae

def select_pearson_neighbors(user_vector, candidate_matrix, threshold=0.5):
    """
    Select the users (rows of `candidate_matrix`) who have a Pearson correlation coefficient greater than the `threshold` with `user_vector`.
    
    Parameters:
        user_vector (np.ndarray): Rating vector of the target user (shape: `n_items`,)
        candidate_matrix (np.ndarray): User-candidate matrix (shape: `n_users`, `n_items`)
        threshold (float): Threshold for the Pearson correlation coefficient

    Returns:
        indices (np.ndarray): Indices of the candidate users who exceed the threshold
        pearson_values (np.ndarray): Corresponding Pearson correlation values
    """
    indices = []
    pearson_values = []

    for idx, candidate in enumerate(candidate_matrix):
        mask = ~np.isnan(user_vector) & ~np.isnan(candidate)
        if np.sum(mask) < 2:
            continue
        r_tuple = pearsonr(user_vector[mask], candidate[mask])
        r = r_tuple[0]
        if not isinstance(r, (float, int)) or np.isnan(r):
            continue
        if r > threshold:
            indices.append(idx)
            pearson_values.append(r)

    return np.array(indices), np.array(pearson_values)
