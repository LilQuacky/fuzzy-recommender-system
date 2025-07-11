import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

"""
This script aggregates experiment results from the CSV file, sorts them by various metrics, and saves a summary and plots (barplot, heatmap, boxplot) for each metric in its own subdirectory under images/comparison/<metric>.
"""

def plot_precision_recall_curve(df, phase, metric_dir):
    """
    Plot the precision-recall curve for the given phase (train/test), if columns for thresholds are available.
    """
    prec_col = f"{phase}_precision_at_threshold"
    rec_col = f"{phase}_recall_at_threshold"
    if prec_col in df.columns and rec_col in df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(df[rec_col], df[prec_col], marker='o')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({phase})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(metric_dir, f'precision_recall_curve_{phase}.png'))
        plt.close()

def plot_metric_vs_threshold(df, phase, metric_base, metric_dir):
    """
    Plot the metric (precision, recall, f1_score) vs threshold for the given phase, if available.
    """
    col = f"{phase}_{metric_base}_at_threshold"
    threshold_col = f"{phase}_thresholds"
    if col in df.columns and threshold_col in df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(df[threshold_col], df[col], marker='o')
        plt.xlabel('Threshold')
        plt.ylabel(metric_base.capitalize())
        plt.title(f'{metric_base.capitalize()} vs Threshold ({phase})')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(metric_dir, f'{metric_base}_vs_threshold_{phase}.png'))
        plt.close()

def plot_train_test_comparison(df, metric_base, metric_dir):
    """
    Plot train vs test comparison for a given metric (precision, recall, f1_score, accuracy).
    """
    train_col = f"train_{metric_base}"
    test_col = f"test_{metric_base}"
    if train_col in df.columns and test_col in df.columns:
        plt.figure(figsize=(7, 5))
        plt.plot(df[train_col], label='Train', marker='o')
        plt.plot(df[test_col], label='Test', marker='o')
        plt.xlabel('Run')
        plt.ylabel(metric_base.capitalize())
        plt.title(f'Train vs Test {metric_base.capitalize()}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(metric_dir, f'train_vs_test_{metric_base}.png'))
        plt.close()

def main(csv_path='output/latest/results/results.csv', top_n=None):
    """
    Reads the results CSV, sorts by multiple metrics, saves a summary to file, and saves plots (heatmap, boxplot, lineplot, violinplot) for each metric in its own subdirectory under images/train/<metric>/ and images/test/<metric>/. Does not display any plots.
    """
    config_path = os.path.join(os.path.dirname(csv_path), 'config.json')
    if top_n is None:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            top_n = config.get('top_n', 5)
        else:
            top_n = 5
    df = pd.read_csv(csv_path)
    save_summary_to_file(df, csv_path, top_n)
    images_dir = os.path.join(os.path.dirname(csv_path), '..', 'images')
    # Nuova struttura: train/<metric>/ e test/<metric>/
    phases = ['train', 'test']
    metrics_by_phase = {
        'train': ["rmse", "mae", "precision", "recall", "accuracy", "f1_score"],
        'test':  ["rmse", "mae"]
    }
    for phase in phases:
        for metric_base in metrics_by_phase[phase]:
            metric = f"{phase}_{metric_base}"
            if metric not in df.columns or df[metric].dropna().empty:
                continue
            metric_dir = os.path.join(images_dir, phase, metric_base)
            os.makedirs(metric_dir, exist_ok=True)
            save_heatmap_metric(df, metric, metric_dir)
            save_boxplot_metric(df, metric, metric_dir)
            save_comparison_lineplot_nclusters(df, metric, metric_dir)
            save_comparison_violinplot(df, metric, metric_dir)

def save_summary_to_file(df, csv_path, top_n=5):
    """
    Saves the summary of the top N runs for each main metric to a text file in the same directory as the CSV.
    """
    summary_path = os.path.join(os.path.dirname(csv_path), 'summary.txt')
    metrics = [
        ("train_rmse", "Top {} runs by train_rmse:"),
        ("train_mae", "Top {} runs by train_mae:"),
        ("test_rmse", "Top {} runs by test_rmse:"),
        ("test_mae", "Top {} runs by test_mae:"),
        ("train_precision", "Top {} runs by train_precision:"),
        ("train_recall", "Top {} runs by train_recall:"),
        ("train_accuracy", "Top {} runs by train_accuracy:"),
        ("train_f1_score", "Top {} runs by train_f1_score:"),
        ("test_precision", "Top {} runs by test_precision:"),
        ("test_recall", "Top {} runs by test_recall:"),
        ("test_accuracy", "Top {} runs by test_accuracy:"),
        ("test_f1_score", "Top {} runs by test_f1_score:")
    ]
    with open(summary_path, 'w') as f:
        for metric, title in metrics:
            if metric not in df.columns or df[metric].dropna().empty:
                continue
            f.write(title.format(top_n) + "\n")
            # Best: min per metriche di errore, max per metriche di qualità
            if any(x in metric for x in ["rmse", "mae"]):
                ascending = True
            else:
                ascending = False
            df_sorted = df.sort_values(metric, ascending=ascending)
            top_df = df_sorted.head(top_n)
            top_df = top_df.reset_index(drop=True)
            top_df.index = range(1, len(top_df) + 1)
            f.write(top_df.to_string(index=True))
            f.write("\n\n")

def save_heatmap_metric(df, metric, metric_dir):
    """
    Saves a heatmap of the given metric by n_clusters and normalization as a PNG file in the specified metric subdirectory.
    """
    if 'n_clusters' not in df.columns or 'normalization' not in df.columns or metric not in df.columns or df[metric].dropna().empty:
        return
    pivot = df.pivot_table(index='n_clusters', columns='normalization', values=metric, aggfunc='min')
    if pivot.dropna(how='all').empty:
        return
    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis')
    plt.title(f'{metric} by n_clusters and normalization')
    plt.ylabel('n_clusters')
    plt.xlabel('Normalization')
    plt.tight_layout()
    plt.savefig(os.path.join(metric_dir, f'heatmap_{metric}.png'))
    plt.close()

def save_boxplot_metric(df, metric, metric_dir):
    """
    Saves a boxplot of the given metric grouped by clustering_method as a PNG file in the specified metric subdirectory.
    """
    if 'clustering_method' not in df.columns or metric not in df.columns or df[metric].dropna().empty:
        return
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='clustering_method', y=metric)
    plt.title(f'{metric} by Clustering Method')
    plt.xlabel('Clustering Method')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(metric_dir, f'boxplot_{metric}.png'))
    plt.close()

def save_comparison_lineplot_nclusters(df, metric, metric_dir):
    """
    Salva un lineplot che confronta il metric tra KMeans e FCM al variare di n_clusters.
    """
    if 'clustering_method' not in df.columns or 'n_clusters' not in df.columns or metric not in df.columns or df[metric].dropna().empty:
        return
    means = df.groupby(['clustering_method', 'n_clusters'])[metric].mean().reset_index()
    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(data=means, x='n_clusters', y=metric, hue='clustering_method', marker='o')
    plt.title(f'{metric} medio vs n_clusters (KMeans vs FCM)')
    plt.xlabel('Numero di cluster')
    plt.ylabel(f'Mean {metric}')
    plt.legend(title='Clustering Method')
    plt.tight_layout()
    plt.savefig(os.path.join(metric_dir, f'lineplot_nclusters_{metric}.png'))
    plt.close()

def save_comparison_violinplot(df, metric, metric_dir):
    """
    Salva un violinplot che mostra la distribuzione del metric per ciascun clustering_method.
    """
    if 'clustering_method' not in df.columns or metric not in df.columns or df[metric].dropna().empty:
        return
    plt.figure(figsize=(6, 6))
    sns.violinplot(data=df, x='clustering_method', y=metric, inner='quartile')
    plt.title(f'Distribuzione {metric} (Violinplot) tra KMeans e FCM')
    plt.xlabel('Clustering Method')
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(os.path.join(metric_dir, f'violinplot_{metric}.png'))
    plt.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main() 
