#!/usr/bin/env python3
"""
Day 18 — Customer Segmentation (K-Means + PCA)
Usage:
    python3 run_kmeans.py              # run with default settings
    python3 run_kmeans.py --clusters 5
If you place a customers.csv file (Age, Income, SpendingScore) in the folder, the script uses it.
Otherwise it generates demo data.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sns.set(style="whitegrid")

def load_data(local_file="customers.csv"):
    """Try local file, otherwise generate demo dataset."""
    if os.path.exists(local_file):
        print(f"📦 Loading local dataset: {local_file}")
        df = pd.read_csv(local_file)
        expected = {'Age', 'Income', 'SpendingScore'}
        if not expected.issubset(set(df.columns)):
            raise ValueError(f"Local file must contain columns: {expected}")
    else:
        print("⚠️  Local file not found — creating demo dataset.")
        rng = np.random.default_rng(42)
        n = 300
        ages = rng.integers(18, 70, size=n)
        income = (rng.normal(50, 18, size=n) * 1000).clip(8_000, 150_000).round(0)
        spending = rng.integers(1, 100, size=n)
        df = pd.DataFrame({
            "Age": ages,
            "Income": income,
            "SpendingScore": spending
        })
    print(f"Dataset shape: {df.shape}")
    return df

def preprocess(df):
    features = df[['Age', 'Income', 'SpendingScore']].astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    print("🔧 Data standardized.")
    return X, scaler

def find_k_elbow(X, kmax=8):
    distortions = []
    K = range(1, kmax+1)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        distortions.append(km.inertia_)
    return K, distortions

def run_kmeans(X, n_clusters=4):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels = km.fit_predict(X)
    print(f"🎯 K-Means complete — found {n_clusters} clusters.")
    return km, labels

def visualize_elbow(K, distortions, out="elbow.png"):
    plt.figure(figsize=(6,4))
    plt.plot(K, distortions, 'o-', linewidth=2)
    plt.xlabel("k (clusters)")
    plt.ylabel("Inertia (distortion)")
    plt.title("Elbow method for k")
    plt.xticks(K)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📁 Saved elbow plot -> {out}")

def visualize_pca(X, labels, out="clusters_pca.png"):
    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X)
    plt.figure(figsize=(8,6))
    palette = sns.color_palette("tab10", n_colors=len(np.unique(labels)))
    sns.scatterplot(x=pcs[:,0], y=pcs[:,1], hue=labels, palette=palette, legend="full", s=50)
    plt.title("Customer Segments (PCA projection)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="Cluster", loc="best")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"📁 Saved PCA plot -> {out}")

def save_results(df, labels, out_csv="clustered_customers.csv", insights="cluster_insights.csv"):
    df_out = df.copy()
    df_out["Cluster"] = labels
    df_out.to_csv(out_csv, index=False)
    print(f"💾 Saved clustered customers -> {out_csv}")

    insights_df = df_out.groupby("Cluster").agg({
        "Age": "mean",
        "Income": "mean",
        "SpendingScore": "mean",
        "Cluster": "count"
    }).rename(columns={"Cluster": "Count"})
    insights_df = insights_df.round(2)
    insights_df.to_csv(insights)
    print(f"📝 Saved cluster summary -> {insights}")

def main(args):
    df = load_data()
    X, scaler = preprocess(df)

    # elbow plot (optional guidance)
    K, distortions = find_k_elbow(X, kmax=8)
    visualize_elbow(K, distortions, out="elbow.png")

    km, labels = run_kmeans(X, n_clusters=args.clusters)
    visualize_pca(X, labels, out="clusters_pca.png")
    save_results(df, labels)

    print("✅ All done. Check the generated files (clusters_pca.png, elbow.png, clustered_customers.csv, cluster_insights.csv).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Customer segmentation (K-Means + PCA)")
    parser.add_argument("--clusters", "-k", type=int, default=4, help="number of clusters (default 4)")
    args = parser.parse_args()
    main(args)