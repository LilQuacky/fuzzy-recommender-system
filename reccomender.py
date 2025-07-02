import numpy as np
import pandas as pd
import skfuzzy as fuzz
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import json
import time
import os

np.random.seed(42)  # Per riproducibilità globale


# --------------------------- FUNZIONI DI UTILITÀ ---------------------------

def load_data():
    """
    Carica i dati MovieLens 100k e restituisce:
    - dataframe ratings completo
    - matrice user-item pivotata (NaN dove mancano rating)
    """
    ratings = pd.read_csv("ml-100k/u.data", sep="\t", names=["user_id", "item_id", "rating", "timestamp"])
    R = ratings.pivot(index='user_id', columns='item_id', values='rating')
    return ratings, R


def normalize_user_ratings(R):
    """
    Normalizza i rating per utente sottraendo la media dell'utente.
    I NaN sono rimpiazzati con 0 per il clustering.
    """
    R_norm = R.subtract(R.mean(axis=1), axis=0)
    return R_norm.astype(float).fillna(0)


def split_train_test_per_user(R, test_size=0.2, random_state=42):
    """
    Suddivide la matrice R in train e test rimuovendo alcuni rating per ogni utente.
    Ritorna due DataFrame: R_train e R_test
    """
    R_train = R.copy()
    R_test = pd.DataFrame(index=R.index, columns=R.columns)

    for user in R.index:
        user_ratings = R.loc[user].dropna()
        if len(user_ratings) < 2:
            continue
        train_ratings, test_ratings = train_test_split(user_ratings.index, test_size=test_size,
                                                       random_state=random_state)
        R_test.loc[user, test_ratings] = R.loc[user, test_ratings]
        R_train.loc[user, test_ratings] = np.nan

    return R_train, R_test


# --------------------------- CLUSTERING & PREDIZIONI ---------------------------

def fcm_cluster(X, n_clusters=5, m=2.0, max_iter=1000, error=1e-5):
    """
    Applica Fuzzy C-Means al dataset X (scalato) e ritorna:
    - centroidi dei cluster
    - matrice di membership degli utenti
    """
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data=X.T, c=n_clusters, m=m, error=error, maxiter=max_iter, init=None, seed=42)
    return cntr, u


def predict_fcm_soft(cntr, membership):
    """
    Prevede i rating come combinazione pesata dei centroidi fuzzy.
    """
    n_clusters, n_users = membership.shape
    n_items = cntr.shape[1]
    pred = np.zeros((n_users, n_items))
    for c in range(n_clusters):
        weights = membership[c, :]
        pred += np.outer(weights, cntr[c, :])
    return pred


def denormalize(pred_norm, R):
    """
    Riporta i rating previsti dal dominio normalizzato al dominio originale.
    """
    means = R.mean(axis=1).values.reshape(-1, 1)
    return pred_norm + means


# --------------------------- VALUTAZIONE ---------------------------

def evaluate(y_true, y_pred):
    """
    Calcola RMSE e MAE solo sui valori non NaN del test set.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)
    mask = ~np.isnan(y_true)
    mse = mean_squared_error(y_true[mask], y_pred[mask])
    mae = mean_absolute_error(y_true[mask], y_pred[mask])
    return np.sqrt(mse), mae


# --------------------------- VISUALIZZAZIONE ---------------------------

def plot_clusters(R_scaled, membership):
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(R_scaled)
    cluster_labels = np.argmax(membership, axis=0)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced[:, 0], reduced[:, 1], c=cluster_labels, cmap='Set1', alpha=0.7)
    plt.title("User Clusters (Fuzzy C-Means)")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.colorbar(scatter, label='Cluster Label')
    plt.savefig("images/fuzzy_clusters_pca.png")

    # Istogramma distribuzione membership fuzzy
    plt.figure(figsize=(8, 4))
    max_membership = np.max(membership, axis=0)
    plt.hist(max_membership, bins=20, color='skyblue', edgecolor='black')
    plt.title("Distribuzione dei massimi valori di membership")
    plt.xlabel("Grado di appartenenza massimo per utente")
    plt.ylabel("Frequenza")
    plt.savefig("images/membership_histogram.png")

    # Heatmap delle membership più incerte
    entropy = -np.sum(membership * np.log(membership + 1e-10), axis=0)
    idx_uncertain = np.argsort(entropy)[-10:]  # top 10 più incerte
    plt.figure(figsize=(10, 6))
    sns.heatmap(membership[:, idx_uncertain], cmap='viridis', annot=True)
    plt.title("Heatmap Membership - 10 utenti più incerti")
    plt.xlabel("Utente")
    plt.ylabel("Cluster")
    plt.savefig("images/membership_heatmap.png")


# --------------------------- MAIN ---------------------------

def main():
    start_total = time.time()

    ratings, R = load_data()
    R_train, R_test = split_train_test_per_user(R, test_size=0.2)
    R_test_aligned = R_test.reindex(columns=R_train.columns, fill_value=np.nan)

    R_train_norm = normalize_user_ratings(R_train)
    R_test_norm = normalize_user_ratings(R_test_aligned)

    scaler = StandardScaler()
    R_train_scaled = scaler.fit_transform(R_train_norm)
    R_test_scaled = scaler.transform(R_test_norm)

    start_fcm = time.time()
    cntr, u = fcm_cluster(R_train_scaled, n_clusters=5, m=1.8, max_iter=2000, error=1e-6)
    u_test, _, _, _, _, _ = fuzz.cluster.cmeans_predict(
        R_test_scaled.T, cntr, 1.8, error=1e-6, maxiter=2000)

    pred_train_norm = predict_fcm_soft(cntr, u)
    pred_test_norm = predict_fcm_soft(cntr, u_test)
    pred_train = denormalize(pred_train_norm, R_train)
    pred_test = denormalize(pred_test_norm, R_test_aligned)
    end_fcm = time.time()

    rmse_train, mae_train = evaluate(R_train.values, pred_train)
    rmse_test, mae_test = evaluate(R_test_aligned.values, pred_test)

    plot_clusters(R_train_scaled, u)

    # Salva metriche su file
    metrics = {
        "train": {"rmse": rmse_train, "mae": mae_train},
        "test": {"rmse": rmse_test, "mae": mae_test},
        "execution_time": {"fcm_seconds": end_fcm - start_fcm, "total_seconds": time.time() - start_total}
    }
    os.makedirs("results", exist_ok=True)
    with open("results/fcm_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print("\nFCM Evaluation:")
    print(f" - Train RMSE: {rmse_train:.4f}, MAE: {mae_train:.4f}")
    print(f" - Test  RMSE: {rmse_test:.4f}, MAE: {mae_test:.4f}")
    print(f" - FCM Execution Time: {metrics['execution_time']['fcm_seconds']:.2f}s")


if __name__ == "__main__":
    main()
