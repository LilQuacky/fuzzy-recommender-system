# Fuzzy Recommender System

A Python-based framework for running experiments with fuzzy clustering-based recommender systems, using the MovieLens datasets. The system supports various normalization strategies, cluster/fuzziness parameters, and provides detailed evaluation and visualization of results.

## Table of Contents

- [Synopsis](#synopsis)
- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Datasets](#datasets)
- [Normalization Methods](#normalization-methods)
- [Algorithm Details](#algorithm-details)
- [Evaluation Metrics](#evaluation-metrics)
- [Outputs & Visualization](#outputs--visualization)
- [Experimental Results](#experimental-results)
- [Comprehensive Report](#comprehensive-report)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Synopsis

This project was developed as an in-depth analysis of Fuzzy Sets for the **Uncertain System** module of the **Complex And Uncertain System** course at the University of Milan-Bicocca.

The aim of this work is to explore the use of **fuzzy clustering** in **recommender systems**, with a particular focus on personalized recommendations for audiovisual content. This project is inspired by and extends the methodology of [**User based Collaborative Filtering using fuzzy C-means**](#https://www.sciencedirect.com/science/article/abs/pii/S0263224116302159) by Koohi and Kiani, where the Fuzzy C-Means (FCM) algorithm is used to group users into overlapping clusters, modeling the possibility that a user may simultaneously belong to multiple preference groups.

**Comparison with the Reference Study:**
- The original study compared three clustering methods (Fuzzy C-Means, K-means, and SOM), two defuzzification strategies (Maximum and Center of Gravity), and used Pearson correlation for neighbor selection, evaluating on the MovieLens 100k dataset.
- **This project** implements a modular and extensible framework that supports Fuzzy C-Means and K-means clustering, both Maximum and COG defuzzification, and optional Pearson-based neighbor selection. All choices are fully configurable via config files.
- The system automates experiments across all parameter combinations (normalization, clusters, fuzziness, clustering method, etc.), provides advanced and organized visualizations.
- All results and configurations are saved for full reproducibility, enabling robust and transparent experimentation.
- The framework includes an aggregate plotter module that automatically generates summary visualizations across all experiment runs. 

## Overview

This project implements a **fuzzy clustering approach** for **collaborative filtering** in **recommender systems**. It allows for systematic experimentation with different normalization techniques, cluster counts, and fuzziness parameters, and evaluates performance on the MovieLens 100k and 1M datasets.

## Features

- **Fuzzy C-Means Clustering** and **K-Means Clustering** for user-item rating prediction
- **Multiple normalization strategies** (centering, z-score, min-max, none)
- **Configurable experiment parameters** (clusters, fuzziness, noise, etc.)
- **Modular selection of clustering, defuzzification, and neighbor selection methods** via config
- **Automated experiment runner** with reproducible results
- **Evaluation metrics**: RMSE, MAE, precision, recall, accuracy, F1-score
- **Visualization**: PCA plots, membership histograms, heatmaps
- **Results and configuration saving** for reproducibility
- **Comprehensive LaTeX report** with theoretical foundations and experimental analysis

## Project Structure

```
fuzzy-recommender-system/
  config/                # Experiment configuration files
  dataset/               # MovieLens datasets (100k, 1M)
  output/                # Results and plots
  report/                # LaTeX files for comprehensive report (in Italian)
    Capitoli/            # Report chapters
      cap1.tex          # Introduction and motivation
      cap2.tex          # Theoretical foundations
      cap3.tex          # EDA
      cap4.tex          # System implementation
      cap5.tex          # Experimental results
      cap6.tex          # Conclusions and future work
    report.tex          # Main latex file
    report.pdf          # Final report
  runner/                # Experiment orchestration and management
  utils/                 # Clustering, evaluation, normalization,
  LICENSE                # License file
  main.py                # Main entry point
  README.md              # This file
```

## Files Overview

**Project Root**
- **main.py**: Entry point for running experiments. Loads the configuration and launches the experiment pipeline via the `Runner` class.

**config/**
- **run_fcm.json**: Configuration for FCM analysis.
- **run_kmeans.json**: Configuration for K-Means analysis.

**runner/**
- **Runner.py**: Orchestrates the entire experiment workflow: loads config, prepares data, runs all experiment combinations, saves results, and triggers aggregate plotting.
- **experiment.py**: Encapsulates a single experiment run: handles clustering, prediction, evaluation, and plotting for a specific parameter combination.
- **config_manager.py**: Loads and manages experiment configuration from JSON files.
- **data_manager.py**: Handles data loading, preprocessing, normalization, and train/test splitting.
- **result_manager.py**: Manages saving of experiment results and configuration for reproducibility.

**utils/**
- **aggregate_plotter.py**: Aggregates and visualizes results across multiple experiment runs, generating summary plots and tables for comparison.
- **Cluster.py**: Implements clustering algorithms, including Fuzzy C-Means (FCM) and K-Means, and provides prediction logic based on user cluster memberships.
- **defuzzifier.py**: Supplies defuzzification methods (Maximum, Center of Gravity) to convert fuzzy cluster memberships into crisp cluster assignments.
- **eda_movielens100k.py**: Experimental Dataset Analysis on MovieLens 100k ratings Dataset.
- **Evaluator.py**: Offers evaluation metrics (RMSE, MAE), denormalization utilities, and neighbor selection functions using Pearson correlation.
- **Plotter.py**: Manages all visualizations, including PCA cluster plots, membership histograms, heatmaps, and rating distribution plots, organized by normalization and experiment.
- **normalizer.py**: Implements normalization techniques for the user-item rating matrix, such as centering, z-score, min-max scaling, and no normalization.
- **preprocessor.py**: Provides preprocessing utilities, including filtering for dense users/items and splitting the dataset into train and test sets per user.
- **loader.py**: Loads MovieLens datasets (100k, 1M) and constructs the user-item rating matrix for downstream processing.

**report/Capitoli/**
- **cap1.tex**: Introduction to fuzzy sets, recommender systems, and project objectives
- **cap2.tex**: Theoretical foundations covering fuzzy logic, clustering, and collaborative filtering
- **cap3.tex**: Experimental Dataset Analysis executed on MovieLens 100k Dataset
- **cap4.tex**: Detailed system implementation including architecture, algorithms, and design choices
- **cap5.tex**: Comprehensive experimental results and analysis
- **cap6.tex**: Conclusions, limitations, and future development directions

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/LilQuacky/fuzzy-recommender-system
   cd fuzzy-recommender-system
   ```

2. **Install dependencies:**
   - Python 3.8+ recommended
   - Required packages:
     ```
     pip install numpy pandas scikit-learn scikit-fuzzy matplotlib seaborn
     ```

3. **Download MovieLens datasets:**
   - Place the `ml-100k` and/or `ml-1m` folders inside the `dataset/` directory.
   - [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
   - [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

## Configuration

All experiment settings are controlled via config file. The system supports different clustering methods with appropriate parameter handling:

- **FCM (Fuzzy C-Means)**: Uses all parameters including `m_values` (fuzziness) and `defuzzification_methods`
- **K-means**: Hard clustering that ignores `m_values` and `defuzzification_methods` (uses fixed values internally)

FCM configuration:

```json
{
    "dataset_name": "ml-100k",
    "normalizations": ["no_normalization", "simple_centering", "minmax_per_user", "zscore_per_user"],
    "min_user_ratings": 100,
    "min_item_ratings": 100,
    "cluster_values": [2, 3, 4, 5],
    "m_values": [1.2, 1.5, 1.8, 2.0, 2.2, 2.5],
    "noise_std": 0.05,
    "test_size": 0.2,
    "random_state": 42,
    "seed": 31,
    "max_iter": 3000,
    "error": 1e-06,
    "output_dir": "output",
    "show_plots": false,
    "images_subdir": "images",
    "results_subdir": "results",
    "run_timestamp_format": "run_fcm",
    "clustering_methods": ["fcm"],
    "defuzzification_methods": ["maximum", "cog"],
    "neighbor_selection_methods": ["pearson"],
    "summary_only": false,
    "top_n": 5,
    "top_n_evaluation": {
        "n_recommendations": 10,
        "rating_threshold": 4.0
    }
}

```
K-Means configuration:
```json
{
    "dataset_name": "ml-100k",
    "normalizations": ["no_normalization", "simple_centering", "minmax_per_user", "zscore_per_user"],
    "min_user_ratings": 100,
    "min_item_ratings": 100,
    "cluster_values": [2, 3, 4, 5],
    "m_values": [2.0],
    "noise_std": 0.05,
    "test_size": 0.2,
    "random_state": 42,
    "seed": 31,
    "max_iter": 3000,
    "error": 1e-06,
    "output_dir": "output",
    "show_plots": false,
    "images_subdir": "images",
    "results_subdir": "results",
    "run_timestamp_format": "run_kmeans",
    "clustering_methods": ["kmeans"],
    "defuzzification_methods": ["maximum"],
    "neighbor_selection_methods": ["pearson"],
    "summary_only": false,
    "top_n": 5,
    "top_n_evaluation": {
        "n_recommendations": 10,
        "rating_threshold": 4.0
    }
} 
```

### Configuration Parameters

- **normalizations**: List of normalization methods to use (options include "no_normalization", "simple_centering", "minmax_per_user", "zscore_per_user").
- **clustering_methods**: List of clustering algorithms to use (such as "fcm" or "kmeans").
- **min_user_ratings**: The minimum number of ratings a user must have provided to be included in the analysis.
- **min_item_ratings**: The minimum number of ratings a movie must have received to be included in the analysis.
- **defuzzification_methods**: List of defuzzification strategies to apply (for example, "maximum" or "cog").
- **neighbor_selection_methods**: List of neighbor selection strategies to use (such as "none" or "pearson").
- **top_n**: The number of top-performing runs to display in the summary and plots for each evaluation metric (default is 5).
- **summary_only**: If set to true, the system will generate only the summary and plots from the most recent results, without running new experiments.
- **n_recommendations**: The number of items recommended to each user for the purpose of evaluating recommendation quality.
- **rating_threshold**: Minimum predicted rating value that an item must reach to be considered recommended in the evaluation of top-N recommendation metrics. 

## Usage

Run the main experiment script:

```bash
python main.py
```

- Results and plots will be saved in a timestamped subdirectory under `output/`.
- You can adjust experiment parameters in the config files or create a new config file.
- To generate only the summary and aggregate plots from the latest results, set `"summary_only": true` in the config file and run `python main.py`.

### Running Specific Experimental Configurations

## Datasets

- **MovieLens 100k**: `dataset/ml-100k/`
- **MovieLens 1M**: `dataset/ml-1m/`
- Data is loaded and preprocessed automatically (dense user/item filtering, train/test split).

## Normalization Methods

Available normalization strategies:

- `"simple_centering"`: Subtracts each user's mean rating
- `"zscore_per_user"`: Z-score normalization per user
- `"minmax_per_user"`: Min-max scaling per user
- `"no_normalization"`: No normalization (missing values filled with 0)

## Algorithm Details

- **Fuzzy C-Means Clustering** (`skfuzzy.cmeans`):
  - Users are clustered in the normalized rating space.
  - Each user has a degree of membership to each cluster.
  - Predictions are made as weighted sums of cluster centroids.
- **Experiment Pipeline**:
  1. Load and preprocess data
  2. Normalize ratings
  3. Add Gaussian noise (optional)
  4. Run fuzzy c-means clustering for each (normalization, cluster, fuzziness) combination
  5. Predict ratings for train/test sets
  6. Evaluate and visualize results

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **Precision** (Top-N recommendation quality)
- **Recall** (Top-N recommendation coverage)
- **Accuracy** (Top-N recommendation accuracy)
- **F1-Score** (Harmonic mean of precision and recall)
- **Average Maximum Membership** (cluster certainty)
- **Average Entropy** (membership uncertainty)
- **Clustering Time** (seconds)

## Outputs & Visualization

- **Results**: Saved as JSON and CSV in `output/<run_name>/results/`
- **Config**: Saved alongside results for reproducibility
- **Plots**: Saved in `output/<run_name>/images/` organized by normalization and phase:
- **Summary**: `summary.txt` in the results directory contains the top-N runs for each main metric, ranked from best (1) to worst (N)
- **Aggregate Comparison**: All aggregate plots and summary are generated automatically after each experiment, or can be generated standalone with `summary_only` mode

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
