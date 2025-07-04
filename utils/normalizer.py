def simple_centering(R):
    """
    Normalize the user-item rating matrix by subtracting the mean rating of each user (row-wise centering).

    Parameters:
        R (pd.DataFrame): User-item rating matrix.

    Returns:
        pd.DataFrame: Centered rating matrix with user means subtracted.
    """
    user_means = R.mean(axis=1)
    return R.subtract(user_means, axis=0).astype(float)


def zscore_per_user(R):
    """
    Normalize the user-item rating matrix using z-score normalization per user (row-wise).

    Parameters:
        R (pd.DataFrame): User-item rating matrix.

    Returns:
        pd.DataFrame: Z-score normalized rating matrix per user.
    """
    R_norm = R.copy()
    for user in R_norm.index:
        user_ratings = R_norm.loc[user].dropna()
        if len(user_ratings) > 1 and user_ratings.std() > 0:
            mean_val = user_ratings.mean()
            std_val = user_ratings.std()
            R_norm.loc[user] = (R_norm.loc[user] - mean_val) / std_val
        else:
            R_norm.loc[user] = R_norm.loc[user] - user_ratings.mean()
    return R_norm.fillna(0).astype(float)


def minmax_per_user(R):
    """
    Normalize the user-item rating matrix using min-max scaling per user (row-wise).

    Parameters:
        R (pd.DataFrame): User-item rating matrix.

    Returns:
        pd.DataFrame: Min-max normalized rating matrix per user.
    """
    R_norm = R.copy()
    for user in R_norm.index:
        user_ratings = R_norm.loc[user].dropna()
        if len(user_ratings) > 1:
            min_val = user_ratings.min()
            max_val = user_ratings.max()
            if max_val > min_val:
                R_norm.loc[user] = (R_norm.loc[user] - min_val) / (max_val - min_val)
    return R_norm.astype(float).fillna(0)


def no_normalization(R):
    """
    Return the user-item rating matrix as float, filling missing values with 0 (no normalization).

    Parameters:
        R (pd.DataFrame): User-item rating matrix.

    Returns:
        pd.DataFrame: Rating matrix as float with NaNs replaced by 0.
    """
    return R.astype(float).fillna(0)

NORMALIZATION_FUNCS = {
    "simple_centering": simple_centering,
    "zscore_per_user": zscore_per_user,
    "minmax_per_user": minmax_per_user,
    "no_normalization": no_normalization
}
