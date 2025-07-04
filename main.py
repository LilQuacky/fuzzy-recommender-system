"""
Main entry point for running fuzzy recommender system experiments.

This script loads configuration from 'config/config.json', sets up the normalization functions,
and runs the experiment using the Runner class.
"""
from runner.Runner import Runner

if __name__ == "__main__":
    config_path = "config/config.json"
    runner = Runner(config_path)
    results = runner.run()
