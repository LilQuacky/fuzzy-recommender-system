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
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Synopsis

This project was developed as an in-depth analysis of Fuzzy Sets for the **Uncertain System** module of the **Complex And Uncertain System** course at the University of Milan-Bicocca.

The aim of this work is to explore the use of **fuzzy clustering** in **recommender systems**, with a particular focus on personalized recommendations for audiovisual content. Specifically, this project proposes a methodological extension of the work presented in [**User based Collaborative Filtering using fuzzy C-means**](#https://www.sciencedirect.com/science/article/abs/pii/S0263224116302159) by Koohi and Kiani, where the Fuzzy C-Means (FCM) algorithm is used to group users into overlapping clusters, thus modeling the possibility that a user may simultaneously belong to multiple preference groups.

Building on this contribution, the project aims to explore the impact of different normalization methods, clustering parameters, and noise on recommendation performance. It provides a reproducible framework for evaluating fuzzy clustering in collaborative filtering through both quantitative metrics and visual analysis.


## Overview

This project implements a **fuzzy c-means clustering approach** for **collaborative filtering** in **recommender systems**. It allows for systematic experimentation with different normalization techniques, cluster counts, and fuzziness parameters, and evaluates performance on the MovieLens 100k and 1M datasets.


## Features

- **Fuzzy C-Means Clustering** for user-item rating prediction
- **Multiple normalization strategies** (centering, z-score, min-max, none)
- **Configurable experiment parameters** (clusters, fuzziness, noise, etc.)
- **Automated experiment runner** with reproducible results
- **Evaluation metrics**: RMSE, MAE, membership entropy, etc.
- **Visualization**: PCA plots, membership histograms, heatmaps
- **Results and configuration saving** for reproducibility


## Project Structure

```
fuzzy-recommender-system/
  config/                # Experiment configuration files
  dataset/               # MovieLens datasets (100k, 1M)
  output/                # Results and plots
  runner/                # Experiment orchestration and management
  utils/                 # Clustering, evaluation, normalization, plotting
  LICENSE                # License file
  main.py                # Main entry point
  README.md              # This file
  report.pdf             # A comprehensive report (in Italian)
```

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

All experiment settings are controlled via `config/config.json`. Example:

```json
{
    "dataset_name": "ml-100k",
    "normalizations": ["simple_centering"],
    "min_user_ratings": 150,
    "min_item_ratings": 150,
    "cluster_values": [4, 6],
    "m_values": [1.9, 2.7],
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
    "run_timestamp_format": "run_%Y_%m_%d_%H_%M_%S"
}
```

- **dataset_name**: `"ml-100k"` or `"ml-1m"`
- **normalizations**: List of normalization methods (see below)
- **cluster_values**: List of cluster counts to try
- **m_values**: List of fuzziness parameters
- **noise_std**: Standard deviation of Gaussian noise added to data
- **test_size**: Fraction of data for testing
- **output_dir**: Where results and images are saved


## Usage

Run the main experiment script:

```bash
python main.py
```

- Results and plots will be saved in a timestamped subdirectory under `output/`.
- You can adjust experiment parameters in `config/config.json` or create a new config file.


## Datasets

- **MovieLens 100k**: `dataset/ml-100k/`
- **MovieLens 1M**: `dataset/ml-1m/`
- Data is loaded and preprocessed automatically (dense user/item filtering, train/test split).


## Normalization Methods

Available normalization strategies (set in `config.json`):

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

- **Results**: Saved as JSON in `output/<timestamp>/results/`
- **Config**: Saved alongside results for reproducibility
- **Plots**: Saved in `output/<timestamp>/images/`
  - PCA scatter plots of user clusters
  - Membership histograms
  - Heatmaps of most uncertain users
  - Data distribution plots


## Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, new features, or improvements.


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
