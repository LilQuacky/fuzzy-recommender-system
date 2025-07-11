from utils.loader import load_data_100k, load_data_1m
from utils.preprocessor import filter_dense, split_train_test_per_user
import numpy as np

class DataManager:
    """
    DataManager handles loading, preprocessing, and normalization of user-item rating datasets
    for recommender system experiments. It supports both MovieLens 100k and 1M datasets.

    Methods:
        - load_and_preprocess(): Loads the dataset, filters for density, splits into train/test, and aligns test set.
        - normalize(R_train, R_test_aligned): Normalizes the train and test matrices using a provided normalization function, filling missing values with row or global means.
    """
    def __init__(self, config):
        self.config = config

    def load_and_preprocess(self):
        if self.config.get('dataset_name') == 'ml-1m':
            _, R = load_data_1m()
        else:
            _, R = load_data_100k()
        R_dense = filter_dense(R, self.config['min_user_ratings'], self.config['min_item_ratings'])
        R_train, R_test = split_train_test_per_user(R_dense, test_size=self.config['test_size'], random_state=self.config['random_state'])
        R_test_aligned = R_test.reindex(columns=R_train.columns, fill_value=np.nan)
        return R_train, R_test_aligned

    def normalize(self, R_train, R_test_aligned, norm_func=None):
        if norm_func is not None:
            R_train_norm = norm_func(R_train)
            R_test_norm = norm_func(R_test_aligned)
        else:
            R_train_norm = R_train.astype(float)
            R_test_norm = R_test_aligned.astype(float)
        
        global_mean = R_train_norm.stack().mean()
        R_train_filled = R_train_norm.apply(lambda row: row.fillna(row.mean() if not np.isnan(row.mean()) else global_mean), axis=1)
        R_test_filled = R_test_norm.apply(lambda row: row.fillna(row.mean() if not np.isnan(row.mean()) else global_mean), axis=1)
        return R_train_filled, R_test_filled 
