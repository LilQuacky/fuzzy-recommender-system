import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test_per_user(R, test_size, random_state):
    """
    Split the user-item rating matrix into train and test sets for each user.
    For each user, randomly select a proportion of items as the test set.

    Parameters:
        R (pd.DataFrame): User-item rating matrix.
        test_size (float): Proportion of each user's ratings to include in the test set (between 0 and 1).
        random_state (int): Random seed for reproducibility.

    Returns:
        R_train (pd.DataFrame): Training set with test ratings set to NaN.
        R_test (pd.DataFrame): Test set with only the test ratings, others set to NaN.
    """
    R_train = R.copy()
    R_test = pd.DataFrame(index=R.index, columns=R.columns)
    for user in R.index:
        user_ratings = R.loc[user].dropna()
        if len(user_ratings) < 2:
            continue
        train_items, test_items = train_test_split(user_ratings.index, test_size=test_size, random_state=random_state)
        R_test.loc[user, test_items] = R.loc[user, test_items]
        R_train.loc[user, test_items] = np.nan
    return R_train, R_test

def filter_dense(R, min_user_ratings=20, min_item_ratings=20):
    """
    Filter the user-item rating matrix to keep only dense users and items.
    Users with at least min_user_ratings and items with at least min_item_ratings are retained.

    Parameters:
        R (pd.DataFrame): User-item rating matrix.
        min_user_ratings (int): Minimum number of ratings required for a user to be kept.
        min_item_ratings (int): Minimum number of ratings required for an item to be kept.

    Returns:
        pd.DataFrame: Filtered user-item rating matrix.
    """
    user_mask = R.notna().sum(axis=1) >= min_user_ratings
    R_filtered = R.loc[user_mask]
    item_mask = R_filtered.notna().sum(axis=0) >= min_item_ratings
    return R_filtered.loc[:, item_mask]
