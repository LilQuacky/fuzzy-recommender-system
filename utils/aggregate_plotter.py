import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

"""
This script aggregates experiment results from the CSV file, sorts them by various metrics, and saves a summary and plots (barplot, heatmap, boxplot) for each metric in its own subdirectory under images/comparison/<metric>.
"""

def main(csv_path='output/latest/results/results.csv', top_n=None):
    """
    Reads the results CSV, sorts by multiple metrics, saves a summary to file, and saves plots (barplot, heatmap, boxplot) for each metric in its own subdirectory. Does not display any plots.
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
    comparison_dir = os.path.abspath(os.path.join(images_dir, 'comparison'))
    os.makedirs(comparison_dir, exist_ok=True)
    metrics = [
        ("train_rmse", True),
        ("train_mae", True),
        ("test_rmse", True),
        ("test_mae", True)
    ]
    for metric, ascending in metrics:
        if metric not in df.columns or df[metric].dropna().empty:
            continue
        metric_dir = os.path.join(comparison_dir, metric)
        os.makedirs(metric_dir, exist_ok=True)
        save_barplot_top_metric(df, metric, metric_dir, top_n, ascending)
        save_heatmap_metric(df, metric, metric_dir)
        save_boxplot_metric(df, metric, metric_dir)

def save_summary_to_file(df, csv_path, top_n=5):
    """
    Saves the summary of the top N runs for each main metric to a text file in the same directory as the CSV.
    """
    summary_path = os.path.join(os.path.dirname(csv_path), 'summary.txt')
    metrics = [
        ("train_rmse", "Top {} runs by train_rmse:"),
        ("train_mae", "Top {} runs by train_mae:"),
        ("test_rmse", "Top {} runs by test_rmse:"),
        ("test_mae", "Top {} runs by test_mae:")
    ]
    with open(summary_path, 'w') as f:
        for metric, title in metrics:
            if metric not in df.columns or df[metric].dropna().empty:
                continue
            f.write(title.format(top_n) + "\n")
            ascending = True
            df_sorted = df.sort_values(metric, ascending=ascending)
            top_df = df_sorted.head(top_n)
            top_df = top_df.reset_index(drop=True)
            top_df.index = range(1, len(top_df) + 1)
            f.write(top_df.to_string(index=True))
            f.write("\n\n")

def save_barplot_top_metric(df, metric, metric_dir, top_n, ascending):
    """
    Saves a barplot of the top N runs for a given metric as a PNG file in the specified metric subdirectory.
    """
    if metric not in df.columns or df[metric].dropna().empty:
        return
    df_sorted = df.sort_values(metric, ascending=ascending)
    top = df_sorted.head(top_n)
    top = top.reset_index(drop=True)
    top.index = range(1, len(top) + 1)
    plt.figure(figsize=(max(8, 1.2 * len(top)), 6))
    ax = sns.barplot(data=top, x=top.index, y=metric, hue='clustering_method')
    plt.title(f'Top {top_n} Runs by {metric}')
    plt.xlabel('Rank')
    plt.ylabel(metric)
    plt.legend(title='Clustering Method')
    plt.tight_layout(rect=(0, 0, 1, 1))
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_horizontalalignment('right')
    plot_path = os.path.join(metric_dir, f'barplot_top_{top_n}_{metric}.png')
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()

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

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main() 
