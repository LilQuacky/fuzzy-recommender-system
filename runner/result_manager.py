import json
import os
import csv

class ResultManager:
    """
    ResultManager handles saving experiment results and configuration files to the specified results directory.

    Methods:
        - save_results(results, filename='results.json'): Saves the experiment results as a JSON file in the results directory.
        - save_config(config, filename='config.json'): Saves the configuration as a JSON file in the results directory.
        - save_results_csv(results, filename='results.csv'): Saves the experiment results in CSV format.
    """
    def __init__(self, config):
        self.config = config
        self.results_dir = config["results_dir"] 

    def save_results(self, results, filename='results.json'):
        path = os.path.join(self.results_dir, filename)
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)

    def save_config(self, config, filename='config.json'):
        path = os.path.join(self.results_dir, filename)
        with open(path, 'w') as f:
            json.dump(config, f, indent=4)

    def save_results_csv(self, results, filename='results.csv'):
        """
        Save experiment results in CSV format, flattening the nested structure.
        Each row represents a run with parameters and metrics.
        """
        path = os.path.join(self.results_dir, filename)
        flat_results = []
        # Naviga la struttura annidata e appiattisce
        for norm_name, norm_dict in results.items():
            for c, c_dict in norm_dict.items():
                for m, m_dict in c_dict.items():
                    for clustering_method, clustering_dict in m_dict.items():
                        for defuzz_method, defuzz_dict in clustering_dict.items():
                            for neighbor_method, metrics in defuzz_dict.items():
                                row = {
                                    'normalization': norm_name,
                                    'n_clusters': c,
                                    'm': m,
                                    'clustering_method': clustering_method,
                                    'defuzz_method': defuzz_method,
                                    'neighbor_method': neighbor_method
                                }
                                row.update(metrics)
                                flat_results.append(row)
        if flat_results:
            with open(path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=flat_results[0].keys())
                writer.writeheader()
                writer.writerows(flat_results) 
