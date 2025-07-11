# Fuzzy Recommender System

This project implements an experimental framework for recommender systems based on fuzzy clustering, with a particular focus on comparing Fuzzy C-Means (FCM) and K-Means, inspired by the methodology of Koohi & Kiani (2016). The system is designed to be modular, extensible, and fully reproducible, enabling systematic experiments on MovieLens 100k (with compatibility for 1M).

## Table of Contents

- [Synopsis](#synopsis)
- [Architecture and Pipeline](#architecture-and-pipeline)
- [Main Features](#main-features)
- [Algorithms and Methods](#algorithms-and-methods)
- [Evaluation and Metrics](#evaluation-and-metrics)
- [Experimental Results](#experimental-results)
- [Limitations and Future Work](#limitations-and-future-work)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Contact](#contact)

## Synopsis

The framework was developed to explore the application of fuzzy clustering in recommender systems, comparing FCM and K-Means on MovieLens 100k. The goal is to assess whether handling uncertainty through fuzzy memberships brings benefits over traditional clustering. The system is designed to be easily extendable to new datasets, parameters, and strategies.

## Architecture and Pipeline

The architecture is divided into four layers:
- **Configuration**: centralized management of experimental parameters via JSON files.
- **Orchestration**: the `Runner` class coordinates the experiment flow, handling parameter combinations, result saving, and reproducibility.
- **Processing**: modules for clustering, prediction, evaluation, and visualization.
- **Utilities**: support functions for loading, preprocessing, and normalization.

The experimental pipeline follows these steps:
1. Load configuration
2. Prepare and filter data (including train/test split)
3. Loop over all combinations of normalization, clusters, fuzziness, clustering method, defuzzification, neighbor selection
4. Clustering (FCM or K-Means)
5. Prediction of missing ratings
6. Performance evaluation
7. Visualization and result saving

## Main Features

- Support for Fuzzy C-Means (FCM) and K-Means (no SOM)
- Four normalization strategies: simple centering, per-user z-score, per-user min-max, no normalization
- Defuzzification: maximum and center of gravity (COG)
- Neighbor selection: none or Pearson
- Centralized and reproducible configuration
- Organized output and automatic visualization (PCA, heatmap, boxplot, membership histograms, summary)
- Option to add noise to data and filter users/items by minimum density

## Algorithms and Methods

### Fuzzy C-Means (FCM)
- Implemented via `skfuzzy`, allows each user to belong to multiple clusters with different membership degrees.
- Configurable fuzziness parameter `m`.
- Rating prediction as a weighted average of cluster centroids according to memberships.

### K-Means
- Implemented via `scikit-learn`, each user belongs to a single cluster (hard clustering).
- Used as a baseline for comparison.

### Normalization
- **Simple Centering**: subtract user mean
- **Per-user z-score**: standardization per user
- **Per-user min-max**: scaling between 0 and 1 per user
- **No normalization**: original values

### Defuzzification
- **Maximum**: cluster with maximum membership
- **COG (Center of Gravity)**: continuous index as weighted average of memberships

### Neighbor Selection
- **None**: all users in the cluster
- **Pearson**: only users with Pearson correlation above threshold

## Evaluation and Metrics

Performance is evaluated on both train and test sets using:
- **RMSE** (Root Mean Square Error)
- **MAE** (Mean Absolute Error)
- **Precision** (top-N)
- **Recall** (top-N)
- **Accuracy** (top-N)
- **F1-score** (top-N)
- **Avg Max Membership** and **Avg Entropy** (FCM only)

Visualization includes:
- Boxplot, heatmap, lineplot for all metrics
- PCA of clusters
- Membership histograms and heatmaps
- Textual summary of top-N results

## Experimental Results

Experiments were conducted on MovieLens 100k, filtering users and items with at least 150 ratings (about 300 users, 900 movies). The main findings are:
- **FCM and K-Means performances are very similar** across all metrics (RMSE, MAE, precision, recall, F1-score)
- **No substantial advantage of fuzzy clustering** over hard clustering in this context
- **Recall and F1-score are higher on test than on train**, suggesting good generalization
- **Homogeneity pattern**: clusters are not well separated, memberships are uniformly distributed, and entropy is high
- **Simple Centering** emerges as the most effective normalization for RMSE/MAE

Example comparison table:

| Metric    | FCM Train | K-Means Train | FCM Test | K-Means Test |
|-----------|-----------|--------------|----------|--------------|
| RMSE      | 0.6851    | 0.6864       | 0.6789   | 0.6817       |
| MAE       | 0.5107    | 0.5092       | 0.5448   | 0.5409       |
| Precision | 58.9%     | 58.8%        | 60.5%    | 60.5%        |
| Recall    | 13.8%     | 13.8%        | 51.2%    | 51.2%        |
| F1-Score  | 21.1%     | 21.1%        | 52.4%    | 52.4%        |

## Limitations and Future Work

- The dataset density (obtained by filtering for computability) may have hidden more interesting fuzzy structures
- The homogeneity of preferences limits cluster separability
- The framework is ready to be extended to:
  - Retain original sparsity
  - Apply to larger or different-domain datasets (e-commerce, music, news)
  - Analysis on MovieLens 1M (already compatible)
  - Systematic variation of other parameters (noise, test size, min ratings)
  - Integration of other fuzzy algorithms (Gustafson-Kessel, Possibilistic C-Means)

## Project Structure

```
fuzzy-recommender-system/
  config/                # Experiment configuration files
  dataset/               # MovieLens data (100k, 1M)
  output/                # Results and plots
  report/                # LaTeX report (in Italian)
    Capitoli/            # Report chapters
  runner/                # Experiment orchestration
  utils/                 # Clustering, evaluation, normalization, plotting
  main.py                # Entry point
  README.md              # This file
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/LilQuacky/fuzzy-recommender-system
   cd fuzzy-recommender-system
   ```
2. Install dependencies (Python 3.8+):
   ```bash
   pip install numpy pandas scikit-learn scikit-fuzzy matplotlib seaborn
   ```
3. Download the MovieLens datasets and place them in `dataset/`:
   - [MovieLens 100k](https://grouplens.org/datasets/movielens/100k/)
   - [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)

## Configuration

All parameters are managed via JSON files in `config/`. Example:
```json
{
    "dataset_name": "ml-100k",
    "normalizations": ["simple_centering", "zscore_per_user", "minmax_per_user", "no_normalization"],
    "cluster_values": [2, 3, 4, 5],
    "m_values": [1.2, 1.5, 1.8, 2.0, 2.2, 2.5],
    "clustering_methods": ["fcm", "kmeans"],
    "defuzzification_methods": ["maximum", "cog"],
    "neighbor_selection_methods": ["none", "pearson"],
    "min_user_ratings": 150,
    "min_item_ratings": 150,
    "test_size": 0.2,
    "max_iter": 3000,
    "error": 1e-06
}
```

## Usage

To run an experiment:
```bash
python main.py
```
- Results and plots will be saved in a subfolder of `output/`.
- To generate only the summary and aggregate plots from the latest results, set `"summary_only": true` in the config and run `python main.py` again.

## Contact

Andrea Falbo — [GitHub](https://github.com/LilQuacky) — a.falbo7@campus.unimib.it
