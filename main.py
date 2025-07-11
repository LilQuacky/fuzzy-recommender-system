"""
Main entry point for running fuzzy recommender system experiments.

This script loads configuration from 'config/config.json', sets up the normalization functions,
and runs the experiment using the Runner class. After completion, it automatically runs the results aggregator.
If the 'summary_only' parameter in the config is true, only the aggregator is executed on the latest results.
"""
from runner.Runner import Runner
import json
import os

if __name__ == "__main__":
    config_path = "config/kmeans.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    if config.get("summary_only"):
        print("Running only the aggregator")
        from utils.aggregate_plotter import main as aggregate_main
        output_dir = config["output_dir"]

        runs = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, d))]
        if not runs:
            print("No runs found in output directory.")
        else:
            latest_run = sorted(runs)[-1]
            results_dir = os.path.join(output_dir, latest_run, config["results_subdir"])
            csv_path = os.path.join(results_dir, "results.csv")
            aggregate_main(csv_path)
    else:
        print("Running the experiment")
        runner = Runner(config_path)
        results = runner.run()
        from utils.aggregate_plotter import main as aggregate_main
        results_dir = runner.config["results_dir"]
        csv_path = os.path.join(results_dir, "results.csv")
        aggregate_main(csv_path)
