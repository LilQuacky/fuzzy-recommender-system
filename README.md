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
- **This project** implements a modular and extensible framework that supports Fuzzy C-Means and K-means clustering, both Maximum and COG defuzzification, and optional Pearson-based neighbor selection. All choices are fully configurable via `sample.json`.
- The system automates experiments across all parameter combinations (normalization, clusters, fuzziness, clustering method, etc.), provides advanced and organized visualizations.
- All results and configurations are saved for full reproducibility, enabling robust and transparent experimentation.
- The framework includes an aggregate plotter module that automatically generates summary visualizations across all experiment runs. 

## Overview

This project implements a **fuzzy clustering approach** for **collaborative filtering** in **recommender systems**. It allows for systematic experimentation with different normalization techniques, cluster counts, and fuzziness parameters, and evaluates performance on the MovieLens 100k and 1M datasets.

The system has been tested through three major experimental runs:
- **Run Sample**: Baseline exploration with 288 parameter combinations
- **Run FCM Deep Dive**: Focused optimization of Fuzzy C-Means with 80 configurations
- **Run K-Means**: Comprehensive benchmark of hard clustering with 108 configurations

Results show significant improvements in recommendation accuracy, with FCM achieving 84% better performance compared to baseline configurations.

## Features

- **Fuzzy C-Means Clustering** and **K-Means Clustering** for user-item rating prediction
- **Multiple normalization strategies** (centering, z-score, min-max, none)
- **Configurable experiment parameters** (clusters, fuzziness, noise, etc.)
- **Modular selection of clustering, defuzzification, and neighbor selection methods** via config
- **Automated experiment runner** with reproducible results
- **Evaluation metrics**: RMSE, MAE, membership entropy, etc.
- **Visualization**: PCA plots, membership histograms, heatmaps
- **Results and configuration saving** for reproducibility
- **Comprehensive LaTeX report** with theoretical foundations and experimental analysis

## Project Structure

```
fuzzy-recommender-system/
  config/                # Experiment configuration files
  dataset/               # MovieLens datasets (100k, 1M)
  output/                # Results and plots
  report/                # LaTeX files for comprehensive report
    Capitoli/            # Report chapters
      cap1.tex          # Introduction and motivation
      cap2.tex          # Theoretical foundations
      cap3.tex          # System implementation
      cap4.tex          # Experimental results
      cap5.tex          # Conclusions and future work
  runner/                # Experiment orchestration and management
  utils/                 # Clustering, evaluation, normalization,
  LICENSE                # License file
  main.py                # Main entry point
  README.md              # This file
  relazione.pdf          # Comprehensive report (in Italian)
```

## Files Overview

**Project Root**
- **main.py**: Entry point for running experiments. Loads the configuration and launches the experiment pipeline via the `Runner` class.

**config/**
- **sample.json**: Main configuration file. Controls all experiment parameters, methods, and plotting options.
- **run_fcm_deep_dive.json**: Optimized configuration for FCM analysis with extended parameter ranges.
- **run_kmeans.json**: Configuration for comprehensive K-Means benchmarking.

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
- **Evaluator.py**: Offers evaluation metrics (RMSE, MAE), denormalization utilities, and neighbor selection functions using Pearson correlation.
- **Plotter.py**: Manages all visualizations, including PCA cluster plots, membership histograms, heatmaps, and rating distribution plots, organized by normalization and experiment.
- **normalizer.py**: Implements normalization techniques for the user-item rating matrix, such as centering, z-score, min-max scaling, and no normalization.
- **preprocessor.py**: Provides preprocessing utilities, including filtering for dense users/items and splitting the dataset into train and test sets per user.
- **loader.py**: Loads MovieLens datasets (100k, 1M) and constructs the user-item rating matrix for downstream processing.

**report/Capitoli/**
- **cap1.tex**: Introduction to fuzzy sets, recommender systems, and project objectives
- **cap2.tex**: Theoretical foundations covering fuzzy logic, clustering, and collaborative filtering
- **cap3.tex**: Detailed system implementation including architecture, algorithms, and design choices
- **cap4.tex**: Comprehensive experimental results and analysis of three major runs
- **cap5.tex**: Conclusions, limitations, and future development directions

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

All experiment settings are controlled via `config/sample.json`. Example:

```json
{
    "dataset_name": "ml-100k",
    "normalizations": ["simple_centering", "zscore_per_user", "minmax_per_user", "no_normalization"],
    "min_user_ratings": 150,
    "min_item_ratings": 150,
    "cluster_values": [4, 6, 8],
    "m_values": [1.5, 2.0, 2.5],
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
    "run_timestamp_format": "run_%Y_%m_%d_%H_%M_%S",
    "clustering_methods": ["fcm", "kmeans"],
    "defuzzification_methods": ["maximum", "cog"],
    "neighbor_selection_methods": ["none", "pearson"],
    "top_n": 5,
    "summary_only": false
}
```

### Configuration Parameters

- **clustering_methods**: List of clustering algorithms to use ("fcm", "kmeans")
- **defuzzification_methods**: List of defuzzification strategies ("maximum", "cog")
- **neighbor_selection_methods**: List of neighbor selection strategies ("none", "pearson")
- **top_n**: Number of top runs to show in the summary and plots for each metric (default: 5)
- **summary_only**: If true, only the summary and plots are generated from the latest results, without running new experiments

## Usage

Run the main experiment script:

```bash
python main.py
```

- Results and plots will be saved in a timestamped subdirectory under `output/`.
- You can adjust experiment parameters in `config/sample.json` or create a new config file.
- To generate only the summary and aggregate plots from the latest results, set `"summary_only": true` in `sample.json` and run `python main.py`.

### Running Specific Experimental Configurations

To run the pre-configured experimental runs:

```bash
# Copy the desired configuration to sample.json
cp config/run_fcm_deep_dive.json config/sample.json
python main.py
```

## Datasets

- **MovieLens 100k**: `dataset/ml-100k/`
- **MovieLens 1M**: `dataset/ml-1m/`
- Data is loaded and preprocessed automatically (dense user/item filtering, train/test split).

## Normalization Methods

Available normalization strategies (set in `sample.json`):

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
- **Average Maximum Membership** (cluster certainty)
- **Average Entropy** (membership uncertainty)
- **Clustering Time** (seconds)

## Outputs & Visualization

- **Results**: Saved as JSON and CSV in `output/<timestamp>/results/`
- **Config**: Saved alongside results for reproducibility
- **Plots**: Saved in `output/<timestamp>/images/` organized by normalization and plot type:
  - `comparison/<metric>/`: For each main metric (`train_rmse`, `train_mae`, `test_rmse`, `test_mae`), contains:
    - `barplot_top_N_<metric>.png`: Barplot of the top-N runs for the metric
    - `heatmap_<metric>.png`: Heatmap of the metric by n_clusters and normalization
    - `boxplot_<metric>.png`: Boxplot of the metric by clustering method
  - `fuzzy_clusters/`: PCA scatter plots of user clusters
  - `membership_histogram/`: Membership histograms
  - `membership_heatmap/`: Heatmaps of most uncertain users
- **Summary**: `summary.txt` in the results directory contains the top-N runs for each main metric, ranked from best (1) to worst (N)
- **Aggregate Comparison**: All aggregate plots and summary are generated automatically after each experiment, or can be generated standalone with `summary_only` mode

### Aggregate Comparison Plots

After every experiment run, aggregate comparison plots and a summary are automatically generated for each main metric (`train_rmse`, `train_mae`, `test_rmse`, `test_mae`).

- All plots are saved in `images/comparison/<metric>/` for each metric.
- The number of top runs shown is configurable via `top_n` in `sample.json`.
- The summary and plots are always ranked from best (rank 1) to worst (rank N) for each metric.
- Metrics `avg_max_membership` and `avg_entropy` are no longer included in the summary or plots for clarity.

## Experimental Results

The system has been extensively tested through three major experimental runs, each providing unique insights into the performance of fuzzy clustering for recommender systems.

### Run Sample
- **Configurations**: 288 parameter combinations
- **Parameters**: 4 normalizations × 3 clusters × 3 fuzziness × 2 algorithms × 2 defuzzifications × 2 neighbor selections
- **Best Results**: RMSE 3.816, MAE 3.704
- **Key Findings**: 
  - FCM with 8 clusters and m=1.5 optimal for RMSE
  - K-Means with 4 clusters optimal for MAE
  - Min-max normalization most effective
  - FCM shows more balanced membership distributions

### Run FCM Deep Dive
- **Configurations**: 80 focused FCM configurations
- **Parameters**: Extended cluster range (6-12), granular fuzziness (1.2-2.2)
- **Best Results**: RMSE 0.594, MAE 0.468 (84% improvement over baseline)
- **Key Findings**:
  - 12 clusters with m=2.0 optimal configuration
  - Simple centering superior to min-max normalization
  - Pearson correlation improves performance by 15-20% but increases computation time 1000x
  - Membership maximum reduced to 0.08, indicating very uniform preference distribution

### Run K-Means Benchmark
- **Configurations**: 108 K-Means configurations
- **Parameters**: Comprehensive parameter sweep for hard clustering comparison
- **Best Results**: RMSE 0.597, MAE 0.465
- **Key Findings**:
  - 15 clusters optimal for K-Means
  - FCM slightly better for RMSE (0.594 vs 0.597)
  - K-Means slightly better for MAE (0.465 vs 0.468)
  - Binary membership (1.0) with zero entropy confirms hard clustering nature

## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
