import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json

config_path = "config/sample.json"
with open(config_path, 'r') as f:
    config = json.load(f)

"""
Exploratory Data Analysis for MovieLens 100k dataset.
Generates plots for rating distributions, user/item statistics, sparsity, correlations, and genres.
Plots are saved in 'before_cut', 'after_cut', and 'comparison' subdirectories under 'output/eda'.
"""

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "dataset", "ml-100k")
OUT_DIR = os.path.join(BASE, "output", "eda")
BEFORE_DIR = os.path.join(OUT_DIR, "before_cut")
AFTER_DIR = os.path.join(OUT_DIR, "after_cut")
COMP_DIR = os.path.join(OUT_DIR, "comparison")
os.makedirs(BEFORE_DIR, exist_ok=True)
os.makedirs(AFTER_DIR, exist_ok=True)
os.makedirs(COMP_DIR, exist_ok=True)

MIN_USER_RATINGS = config.get("min_user_ratings")
MIN_ITEM_RATINGS = config.get("min_item_ratings")

data = pd.read_csv(os.path.join(DATA_DIR, "u.data"), sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

users = pd.read_csv(os.path.join(DATA_DIR, "u.user"), sep='|', names=['user_id', 'age', 'gender', 'occupation', 'zip_code'])

genre_cols = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary',
    'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
    'Thriller', 'War', 'Western'
]
item_cols = ['movie_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
items = pd.read_csv(os.path.join(DATA_DIR, "u.item"), sep='|', encoding='latin-1', names=item_cols, usecols=range(24))

def filter_data(data, min_user_ratings, min_item_ratings):
    user_counts = data['user_id'].value_counts()
    item_counts = data['item_id'].value_counts()
    users_to_keep = user_counts[user_counts >= min_user_ratings].index
    items_to_keep = item_counts[item_counts >= min_item_ratings].index
    filtered = data[data['user_id'].isin(users_to_keep) & data['item_id'].isin(items_to_keep)]
    return filtered

def eda_plots(data, items, outdir, prefix=""):
    n_users = data['user_id'].nunique()
    n_items = data['item_id'].nunique()
    n_ratings = data.shape[0]
    density = n_ratings / (n_users * n_items)
    with open(os.path.join(outdir, f'{prefix}basic_stats.txt'), 'w') as f:
        f.write(f"Users: {n_users}\nItems: {n_items}\nRatings: {n_ratings}\nDensity: {density:.4f}\n")

    # Rating distribution
    plt.figure(figsize=(8,5))
    sns.countplot(x='rating', data=data, color='C0')
    plt.title('Rating Distribution', fontsize=16)
    plt.xlabel('Rating', fontsize=13)
    plt.ylabel('Count', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.savefig(os.path.join(outdir, f'{prefix}rating_distribution.png'))
    plt.close()

    plt.figure(figsize=(8,5))
    sns.boxplot(x='rating', data=data, color='C0')
    plt.title('Rating Boxplot', fontsize=16)
    plt.xlabel('Rating', fontsize=13)
    plt.ylabel('Value', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.savefig(os.path.join(outdir, f'{prefix}rating_boxplot.png'))
    plt.close()

    # Ratings per user/item
    user_counts = data['user_id'].value_counts()
    item_counts = data['item_id'].value_counts()

    plt.figure(figsize=(10,5))
    sns.histplot(pd.DataFrame({'count': user_counts}), x='count', bins=30, color='skyblue')
    plt.title('Ratings per User', fontsize=16)
    plt.xlabel('Number of Ratings', fontsize=13)
    plt.ylabel('Number of Users', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.savefig(os.path.join(outdir, f'{prefix}ratings_per_user_hist.png'))
    plt.close()

    plt.figure(figsize=(10,5))
    sns.boxplot(x=user_counts, color='skyblue')
    plt.title('Ratings per User (Boxplot)', fontsize=16)
    plt.xlabel('Number of Ratings', fontsize=13)
    plt.ylabel('Value', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.savefig(os.path.join(outdir, f'{prefix}ratings_per_user_box.png'))
    plt.close()

    plt.figure(figsize=(10,5))
    sns.histplot(pd.DataFrame({'count': item_counts}), x='count', bins=30, color='salmon')
    plt.title('Ratings per Item', fontsize=16)
    plt.xlabel('Number of Ratings', fontsize=13)
    plt.ylabel('Number of Items', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.savefig(os.path.join(outdir, f'{prefix}ratings_per_item_hist.png'))
    plt.close()

    plt.figure(figsize=(10,5))
    sns.boxplot(x=item_counts, color='salmon')
    plt.title('Ratings per Item (Boxplot)', fontsize=16)
    plt.xlabel('Number of Ratings', fontsize=13)
    plt.ylabel('Value', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.savefig(os.path.join(outdir, f'{prefix}ratings_per_item_box.png'))
    plt.close()

    # Sparsity plot
    pivot = pd.pivot_table(data, index='user_id', columns='item_id', values='rating')
    plt.figure(figsize=(14,8))
    plt.spy(~pivot.isna(), markersize=0.5)
    plt.title('User-Item Matrix Sparsity', fontsize=16)
    plt.xlabel('Item ID', fontsize=13)
    plt.ylabel('User ID', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.savefig(os.path.join(outdir, f'{prefix}sparsity_matrix.png'))
    plt.close()

    # Correlation heatmaps (subset for readability)
    user_sample = user_counts.index[:50]
    item_sample = item_counts.index[:50]
    user_corr = pivot.loc[user_sample].T.corr(method='pearson')
    item_corr = pivot.loc[:, item_sample].corr(method='pearson')

    plt.figure(figsize=(14,10))
    sns.heatmap(user_corr, cmap='coolwarm', center=0)
    plt.title('User-User Correlation (Sample)', fontsize=16)
    plt.xlabel('User', fontsize=13)
    plt.ylabel('User', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(os.path.join(outdir, f'{prefix}user_correlation_heatmap.png'))
    plt.close()

    plt.figure(figsize=(14,10))
    sns.heatmap(item_corr, cmap='coolwarm', center=0)
    plt.title('Item-Item Correlation (Sample)', fontsize=16)
    plt.xlabel('Item', fontsize=13)
    plt.ylabel('Item', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(os.path.join(outdir, f'{prefix}item_correlation_heatmap.png'))
    plt.close()

    genre_sums = items[genre_cols].sum().sort_values(ascending=False)
    genre_df = pd.DataFrame({
        'genre': genre_sums.index,
        'count': genre_sums.values
    })
    plt.figure(figsize=(16,8))
    sns.barplot(data=genre_df, x='genre', y='count', hue='genre', dodge=False, palette='tab20')
    plt.title('Number of Movies per Genre', fontsize=16)
    plt.ylabel('Number of Movies', fontsize=13)
    plt.xlabel('Genre', fontsize=13)
    plt.xticks(rotation=45, fontsize=11)
    plt.yticks(fontsize=11)
    plt.legend([],[], frameon=False)
    plt.tight_layout()
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)
    plt.savefig(os.path.join(outdir, f'{prefix}movies_per_genre.png'))
    plt.close()

    user_sample = user_counts.index[:30]
    item_sample = item_counts.index[:30]
    pivot_sample = pivot.loc[user_sample, item_sample]
    plt.figure(figsize=(14, 10))
    sns.heatmap(pivot_sample, cmap='viridis', vmin=1, vmax=5, cbar_kws={'label': 'Rating'})
    plt.title('User-Item Ratings Heatmap (Sample)', fontsize=16)
    plt.xlabel('Item ID', fontsize=13)
    plt.ylabel('User ID', fontsize=13)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.savefig(os.path.join(outdir, f'{prefix}user_item_heatmap.png'))
    plt.close()

eda_plots(data, items, BEFORE_DIR, prefix="before_")

data_cut = filter_data(data, MIN_USER_RATINGS, MIN_ITEM_RATINGS)
eda_plots(data_cut, items, AFTER_DIR, prefix="after_")

def comparison_plot(img1, img2, out_path, title1, title2, suptitle):
    from PIL import Image
    fig, axes = plt.subplots(1, 2, figsize=(22, 10))
    axes[0].imshow(Image.open(img1))
    axes[0].set_title(title1, fontsize=30)
    axes[0].axis('off')
    axes[1].imshow(Image.open(img2))
    axes[1].set_title(title2, fontsize=30)
    axes[1].axis('off')
    plt.suptitle(suptitle, fontsize=36)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path)
    plt.close()

plot_names = [
    'rating_distribution.png',
    'rating_boxplot.png',
    'ratings_per_user_hist.png',
    'ratings_per_user_box.png',
    'ratings_per_item_hist.png',
    'ratings_per_item_box.png',
    'sparsity_matrix.png',
    'user_correlation_heatmap.png',
    'item_correlation_heatmap.png',
    'movies_per_genre.png',
]

for name in plot_names:
    before = os.path.join(BEFORE_DIR, f'before_{name}')
    after = os.path.join(AFTER_DIR, f'after_{name}')
    out = os.path.join(COMP_DIR, f'comparison_{name}')
    title1 = 'Before Cut'
    title2 = 'After Cut'
    suptitle = name.replace('_', ' ').replace('.png', '').title()
    if os.path.exists(before) and os.path.exists(after):
        comparison_plot(before, after, out, title1, title2, suptitle) 
