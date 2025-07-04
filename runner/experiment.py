class Experiment:
    """
    Experiment encapsulates the setup and execution of a fuzzy clustering-based recommender system experiment.
    It manages the clustering, prediction, evaluation, and visualization steps for a given dataset split and configuration.

    Methods:
        - run(): Executes the experiment pipeline, including clustering, prediction, evaluation, and plotting. Returns a dictionary of evaluation metrics and statistics.
    """
    def __init__(self, R_train_scaled, R_test_scaled, R_train, R_test_aligned, n_clusters, m, max_iter, error, seed, clusterer, evaluator, plotter):
        self.R_train_scaled = R_train_scaled
        self.R_test_scaled = R_test_scaled
        self.R_train = R_train
        self.R_test_aligned = R_test_aligned
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.seed = seed
        self.clusterer = clusterer
        self.evaluator = evaluator
        self.plotter = plotter

    def run(self):
        import time
        import numpy as np

        start_fcm = time.perf_counter()

        cntr, u = self.clusterer.fcm_cluster(self.R_train_scaled, self.n_clusters, self.m, self.error, self.max_iter, self.seed)
        u_test = self.clusterer.predict_test(self.R_test_scaled, cntr, self.m, self.error, self.max_iter)
        pred_train_norm = self.clusterer.predict(cntr, u)
        pred_test_norm = self.clusterer.predict(cntr, u_test)

        pred_train = self.evaluator.denormalize(pred_train_norm, self.R_train)
        pred_test = self.evaluator.denormalize(pred_test_norm, self.R_test_aligned)

        rmse_train, mae_train = self.evaluator.evaluate(self.R_train.values, pred_train)
        rmse_test, mae_test = self.evaluator.evaluate(self.R_test_aligned.values, pred_test)

        avg_max_membership = np.mean(np.max(u, axis=0))
        avg_entropy = np.mean(-np.sum(u * np.log(u + 1e-10), axis=0))
        fcm_time_sec = time.perf_counter() - start_fcm

        prefix = f"c{self.n_clusters}_m{self.m}"
        self.plotter.plot_clusters(self.R_train_scaled, u, prefix=prefix)
        self.plotter.plot_single_normalization(self.R_train_scaled, u, prefix)

        return {
            "train_rmse": rmse_train,
            "train_mae": mae_train,
            "test_rmse": rmse_test,
            "test_mae": mae_test,
            "avg_max_membership": avg_max_membership,
            "avg_entropy": avg_entropy,
            "fcm_time_sec": fcm_time_sec
        }
