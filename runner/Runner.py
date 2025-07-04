from runner.config_manager import ConfigManager
from runner.data_manager import DataManager
from runner.experiment import Experiment
from runner.result_manager import ResultManager
from utils.normalizer import NORMALIZATION_FUNCS
from datetime import datetime
import os
import numpy as np

class Runner:
    """
    Runner orchestrates the end-to-end execution of recommender system experiments, including configuration loading,
    data preparation, experiment execution, and result saving. It manages the experiment workflow for different
    normalization strategies, cluster counts, and fuzziness parameters.

    Methods:
        - run(): Executes the full experiment pipeline for all specified configurations and saves results and config files.
    """
    def __init__(self, config_path):
        self.config = ConfigManager(config_path).load()
        timestamp_format = self.config.get("run_timestamp_format", "run_%Y_%m_%d_%H_%M_%S")
        run_timestamp = datetime.now().strftime(timestamp_format)
        base_output_dir = os.path.join(self.config.get("output_dir", "output"), run_timestamp)

        self.config["run_timestamp"] = run_timestamp
        self.config["base_output_dir"] = base_output_dir
        self.config["images_dir"] = os.path.join(base_output_dir, self.config.get("images_subdir", "images"))
        self.config["results_dir"] = os.path.join(base_output_dir, self.config.get("results_subdir", "results"))

        os.makedirs(self.config["images_dir"], exist_ok=True)
        os.makedirs(self.config["results_dir"], exist_ok=True)

        self.data_manager = DataManager(self.config)
        self.result_manager = ResultManager(self.config)

    def run(self):
      from utils.Cluster import Clusterer
      from utils.Evaluator import Evaluator
      from utils.Plotter import Plotter

      results = {}
      clusterer = Clusterer()
      evaluator = Evaluator()
      plotter = Plotter(self.config["images_dir"], show_plots=self.config.get("show_plots"))

      R_train, R_test_aligned = self.data_manager.load_and_preprocess()

      normalizations = self.config.get("normalizations", ["simple_centering"])
      cluster_values = self.config["cluster_values"]
      m_values = self.config["m_values"]
      noise_std = self.config["noise_std"]
      max_iter = self.config["max_iter"]
      error = self.config["error"]
      seed = self.config["seed"]

      clustering_methods = self.config.get("clustering_methods", ["fcm"])
      defuzz_methods = self.config.get("defuzzification_methods", ["maximum"])
      neighbor_methods = self.config.get("neighbor_selection_methods", ["none"])

      for norm_name in normalizations:
          print(f"Running experiment with normalization: {norm_name}")
          norm_func = NORMALIZATION_FUNCS[norm_name]
          R_train_filled, R_test_filled = self.data_manager.normalize(R_train, R_test_aligned, norm_func)

          noise_train = np.random.normal(0, noise_std, size=R_train_filled.shape)
          noise_test = np.random.normal(0, noise_std, size=R_test_filled.shape)
          R_train_scaled = R_train_filled.to_numpy() + noise_train
          R_test_scaled = R_test_filled.to_numpy() + noise_test

          results[norm_name] = {}
          for c in cluster_values:
              results[norm_name][str(c)] = {}
              for m in m_values:
                  results[norm_name][str(c)][str(m)] = {}
                  for clustering_method in clustering_methods:
                      results[norm_name][str(c)][str(m)][clustering_method] = {}
                      for defuzz_method in defuzz_methods:
                          results[norm_name][str(c)][str(m)][clustering_method][defuzz_method] = {}
                          for neighbor_method in neighbor_methods:
                              base_plot_dir = os.path.join(
                                  self.config["images_dir"],
                                  f"norm={norm_name}"
                              )
                              os.makedirs(base_plot_dir, exist_ok=True)
                              # Sottodirectory per ogni tipo di plot
                              comparison_dir = os.path.join(base_plot_dir, "comparison")
                              fuzzy_clusters_dir = os.path.join(base_plot_dir, "fuzzy_clusters")
                              membership_heatmap_dir = os.path.join(base_plot_dir, "membership_heatmap")
                              membership_histogram_dir = os.path.join(base_plot_dir, "membership_histogram")
                              for d in [comparison_dir, fuzzy_clusters_dir, membership_heatmap_dir, membership_histogram_dir]:
                                  os.makedirs(d, exist_ok=True)
                              # Passa la directory base al Plotter, che poi userà le subdir in base al tipo di plot
                              plotter = Plotter(base_plot_dir, show_plots=self.config.get("show_plots"))
                              print(f"Running: norm={norm_name}, c={c}, m={m}, clustering={clustering_method}, defuzz={defuzz_method}, neighbor={neighbor_method}")
                              experiment = Experiment(
                                  R_train_scaled, R_test_scaled, R_train, R_test_aligned,
                                  n_clusters=c, m=m, max_iter=max_iter, error=error, seed=seed,
                                  clusterer=clusterer, evaluator=evaluator, plotter=plotter,
                                  clustering_method=clustering_method,
                                  defuzz_method=defuzz_method,
                                  neighbor_method=neighbor_method
                              )
                              metrics = experiment.run()
                              results[norm_name][str(c)][str(m)][clustering_method][defuzz_method][neighbor_method] = metrics

      self.result_manager.save_results(results)
      self.result_manager.save_results_csv(results)
      self.result_manager.save_config(self.config)
      print("Runner completed all experiments.")
