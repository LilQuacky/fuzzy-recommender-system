import pandas as pd


def load_data_100k(path="dataset"):
    """
    Load the MovieLens 100k dataset from the specified path.

    Parameters:
        path (str): The base directory containing the 'ml-100k' folder. Defaults to 'dataset'.

    Returns:
        ratings (pd.DataFrame): The raw ratings data with columns [user_id, item_id, rating, timestamp].
        R (pd.DataFrame): User-item rating matrix (users as rows, items as columns).
    """
    ratings = pd.read_csv(f"{path}/ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    R = ratings.pivot(index='user_id', columns='item_id', values='rating')
    return ratings, R

def load_data_1m(path="dataset"):
    """
    Load the MovieLens 1M dataset from the specified path.

    Parameters:
        path (str): The base directory containing the 'ml-1m' folder. Defaults to 'dataset'.

    Returns:
        ratings (pd.DataFrame): The raw ratings data with columns [user_id, item_id, rating, timestamp].
        R (pd.DataFrame): User-item rating matrix (users as rows, items as columns).
    """
    ratings = pd.read_csv(f"{path}/ml-1m/ratings.dat", sep="::", engine='python', names=["user_id", "item_id", "rating", "timestamp"])
    R = ratings.pivot(index='user_id', columns='item_id', values='rating')
    return ratings, R
