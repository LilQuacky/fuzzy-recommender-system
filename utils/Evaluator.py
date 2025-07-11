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
      Evaluates predictions using RMSE and MAE metrics.

      Args:
          y_true (np.ndarray): Ground truth values.
          y_pred (np.ndarray): Predicted values.

      Returns:
          tuple: (rmse, mae)
              rmse (float or np.nan): Root Mean Squared Error.
              mae (float or np.nan): Mean Absolute Error.
              Returns np.nan for both if there are no valid samples.
      """
      y_true = np.array(y_true, dtype=np.float64)
      y_pred = np.array(y_pred, dtype=np.float64)
      mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
      if np.sum(mask) == 0:
          return np.nan, np.nan
      mse = mean_squared_error(y_true[mask], y_pred[mask])
      mae = mean_absolute_error(y_true[mask], y_pred[mask])
      return np.sqrt(mse), mae

    def evaluate_top_n(self, y_true, y_pred, n=10, threshold=4.0):
        """
        Evaluate top-N recommendations using Precision, Recall, Accuracy, and F1-score.
        
        Parameters:
            y_true (array-like): Ground truth ratings (users x items, can contain NaNs).
            y_pred (array-like): Predicted ratings (users x items).
            n (int): Number of top items to recommend.
            threshold (float): Rating threshold to consider an item as "liked".
            
        Returns:
            dict: Dictionary containing precision, recall, accuracy, f1_score for both train and test.
        """
        y_true = np.array(y_true, dtype=np.float64)
        y_pred = np.array(y_pred, dtype=np.float64)
        
        # For each user, get top-N recommendations and compare with ground truth
        precision_scores = []
        recall_scores = []
        accuracy_scores = []
        f1_scores = []
        
        n_users = y_true.shape[0]
        
        for user_idx in range(n_users):
            user_true = y_true[user_idx]
            user_pred = y_pred[user_idx]
            
            # Find items that user has rated in ground truth (not NaN)
            rated_mask = ~np.isnan(user_true)
            if np.sum(rated_mask) == 0:
                continue
                
            # Get predicted ratings for rated items only
            rated_pred = user_pred[rated_mask]
            rated_true = user_true[rated_mask]
            
            # Get top-N recommendations (highest predicted ratings)
            if len(rated_pred) <= n:
                top_n_indices = np.argsort(rated_pred)[::-1]  # All items
            else:
                top_n_indices = np.argsort(rated_pred)[::-1][:n]  # Top N items
            
            # Get ground truth for top-N items
            top_n_true = rated_true[top_n_indices]
            
            # Define "liked" items (rating >= threshold)
            liked_items = top_n_true >= threshold
            
            # Calculate metrics
            if len(liked_items) == 0:
                precision = 0.0
                recall = 0.0
                accuracy = 0.0
                f1_score = 0.0
            else:
                # Precision: fraction of recommended items that are liked
                precision = np.mean(liked_items)
                
                # For recall, we need to know how many items the user liked in total
                total_liked = np.sum(rated_true >= threshold)
                if total_liked == 0:
                    recall = 0.0
                else:
                    # Recall: fraction of liked items that were recommended
                    recommended_liked = np.sum(liked_items)
                    recall = recommended_liked / total_liked
                
                # Accuracy: fraction of correct predictions (liked items in top-N)
                accuracy = precision
                
                # F1-score: harmonic mean of precision and recall
                if precision + recall == 0:
                    f1_score = 0.0
                else:
                    f1_score = 2 * (precision * recall) / (precision + recall)
            
            precision_scores.append(precision)
            recall_scores.append(recall)
            accuracy_scores.append(accuracy)
            f1_scores.append(f1_score)
        
        # Return average metrics across all users
        return {
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'accuracy': np.mean(accuracy_scores),
            'f1_score': np.mean(f1_scores)
        }

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
