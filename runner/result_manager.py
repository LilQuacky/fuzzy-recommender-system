import json
import os

class ResultManager:
    """
    ResultManager handles saving experiment results and configuration files to the specified results directory.

    Methods:
        - save_results(results, filename='results.json'): Saves the experiment results as a JSON file in the results directory.
        - save_config(config, filename='config.json'): Saves the configuration as a JSON file in the results directory.
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
