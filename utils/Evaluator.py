import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


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
        Evaluate predictions using RMSE and MAE, ignoring NaN values in the ground truth.

        Parameters:
            y_true (array-like): Ground truth ratings (can contain NaNs).
            y_pred (array-like): Predicted ratings.

        Returns:
            tuple: (rmse, mae) where rmse is root mean squared error and mae is mean absolute error.
        """
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        mask = ~np.isnan(y_true)
        mse = mean_squared_error(y_true[mask], y_pred[mask])
        mae = mean_absolute_error(y_true[mask], y_pred[mask])
        return np.sqrt(mse), mae
