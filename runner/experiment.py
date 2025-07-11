class Experiment:
    """
    Experiment encapsulates the setup and execution of a fuzzy clustering-based recommender system experiment.
    It manages the clustering, prediction, evaluation, and visualization steps for a given dataset split and configuration.

    Methods:
        - run(): Executes the experiment pipeline, including clustering, prediction, evaluation, and plotting. Returns a dictionary of evaluation metrics and statistics.
    """
    def __init__(self, R_train_scaled, R_test_scaled, R_train, R_test_aligned, n_clusters, m, max_iter, error, seed, clusterer, evaluator, plotter,
                 clustering_method="fcm", defuzz_method="maximum", neighbor_method="none", top_n_config=None):
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
        self.clustering_method = clustering_method
        self.defuzz_method = defuzz_method
        self.neighbor_method = neighbor_method
        self.top_n_config = top_n_config or {"n_recommendations": 10, "rating_threshold": 4.0}

    def predict_with_pearson_neighbors(self, R, cluster_assignments, threshold=0.5):
        import numpy as np
        from utils.Evaluator import select_pearson_neighbors
        n_users, n_items = R.shape
        pred = np.full((n_users, n_items), np.nan)
        for user_idx in range(n_users):
            user_vector = R[user_idx]
            user_cluster = cluster_assignments[user_idx]
            same_cluster_mask = (cluster_assignments == user_cluster)
            same_cluster_mask[user_idx] = False
            candidate_matrix = R[same_cluster_mask]
            if candidate_matrix.shape[0] == 0:
                continue
            neighbor_indices, _ = select_pearson_neighbors(user_vector, candidate_matrix, threshold=threshold)
            if len(neighbor_indices) == 0:
                continue
            neighbors = candidate_matrix[neighbor_indices]
            with np.errstate(all='ignore'):
                pred[user_idx] = np.nanmean(neighbors, axis=0)
        return pred

    def run(self):
        import time
        import numpy as np
        from utils.defuzzifier import defuzzify_maximum, defuzzify_cog

        start_time = time.perf_counter()

        if self.clustering_method == "fcm":
            cntr, u = self.clusterer.fcm_cluster(self.R_train_scaled, self.n_clusters, self.m, self.error, self.max_iter, self.seed)
            u_test = self.clusterer.predict_test(self.R_test_scaled, cntr, self.m, self.error, self.max_iter)
        elif self.clustering_method == "kmeans":
            cntr, u = self.clusterer.kmeans_cluster(self.R_train_scaled, self.n_clusters, self.seed)
            u_test = self.clusterer.kmeans_cluster(self.R_test_scaled, self.n_clusters, self.seed)[1]
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        if self.clustering_method == "kmeans":
            cluster_assignments = np.argmax(u, axis=0)
            cluster_assignments_test = np.argmax(u_test, axis=0)
        else:
            if self.defuzz_method == "maximum":
                cluster_assignments = defuzzify_maximum(u)
                cluster_assignments_test = defuzzify_maximum(u_test)
            elif self.defuzz_method == "cog":
                cluster_assignments = np.round(defuzzify_cog(u)).astype(int)
                cluster_assignments_test = np.round(defuzzify_cog(u_test)).astype(int)
            else:
                raise ValueError(f"Unknown defuzzification method: {self.defuzz_method}")

        if self.neighbor_method == "pearson":
            pred_train = self.predict_with_pearson_neighbors(self.R_train_scaled, cluster_assignments)
            pred_test = self.predict_with_pearson_neighbors(self.R_test_scaled, cluster_assignments_test)
            pred_train = self.evaluator.denormalize(pred_train, self.R_train)
            pred_test = self.evaluator.denormalize(pred_test, self.R_test_aligned)
        else:
            pred_train_norm = self.clusterer.predict(cntr, u)
            pred_test_norm = self.clusterer.predict(cntr, u_test)
            pred_train = self.evaluator.denormalize(pred_train_norm, self.R_train)
            pred_test = self.evaluator.denormalize(pred_test_norm, self.R_test_aligned)

        rmse_train, mae_train = self.evaluator.evaluate(self.R_train.values, pred_train)
        rmse_test, mae_test = self.evaluator.evaluate(self.R_test_aligned.values, pred_test)

        top_n_train = self.evaluator.evaluate_top_n(self.R_train.values, pred_train, n=self.top_n_config["n_recommendations"], threshold=self.top_n_config["rating_threshold"])
        top_n_test = self.evaluator.evaluate_top_n(self.R_test_aligned.values, pred_test, n=self.top_n_config["n_recommendations"], threshold=self.top_n_config["rating_threshold"])


        if self.clustering_method == "fcm":
            avg_max_membership = np.mean(np.max(u, axis=0))
            avg_entropy = np.mean(-np.sum(u * np.log(u + 1e-10), axis=0))
        else:
            avg_max_membership = 1.0
            avg_entropy = 0.0
            
        elapsed_sec = time.perf_counter() - start_time

        prefix = f"c{self.n_clusters}_m{self.m}_{self.clustering_method}_{self.defuzz_method}_{self.neighbor_method}"
        self.plotter.plot_clusters(self.R_train_scaled, u, prefix=prefix, clustering_method=self.clustering_method)

        return {
            "train_rmse": rmse_train,
            "train_mae": mae_train,
            "test_rmse": rmse_test,
            "test_mae": mae_test,
            "train_precision": top_n_train['precision'],
            "train_recall": top_n_train['recall'],
            "train_accuracy": top_n_train['accuracy'],
            "train_f1_score": top_n_train['f1_score'],
            "test_precision": top_n_test['precision'],
            "test_recall": top_n_test['recall'],
            "test_accuracy": top_n_test['accuracy'],
            "test_f1_score": top_n_test['f1_score'],
            "avg_max_membership": avg_max_membership,
            "avg_entropy": avg_entropy,
            "elapsed_sec": elapsed_sec
        }
