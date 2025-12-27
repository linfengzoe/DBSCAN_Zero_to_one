
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import KMeans

# 手动DBSCAN算法
from manual_dbscan import my_dbscan, NOISE


OUTDIR = "paper_repro_outputs"


def ensure_dir():
    os.makedirs(OUTDIR, exist_ok=True)


def savefig(fname: str):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, fname), dpi=300)
    plt.close()


def plot_scatter(X, labels, title, fname):
    plt.figure(figsize=(6.2, 4.2))
    plt.scatter(X[:, 0], X[:, 1], c=labels, s=12, cmap="plasma")
    plt.title(title)
    plt.xlabel("x1")
    plt.ylabel("x2")
    savefig(fname)


def plot_k_distance(X: np.ndarray, k: int, fname: str):
    """
    Fig.5：k-距离图（k=5）
    朴素O(n^2)距离，仅用于论文复现规模。
    """
    X = np.asarray(X, float)
    n = len(X)
    dist2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)
    kth = np.sqrt(np.partition(dist2, kth=k, axis=1)[:, k])  # 第k近邻距离
    kth_sorted = np.sort(kth)

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(np.arange(n), kth_sorted)
    plt.xlabel("Points (sorted)")
    plt.ylabel(f"{k}-distance")
    plt.title(f"Fig.5  k-distance graph on Two Moons (k={k})")
    savefig(fname)


def main():
    ensure_dir()

  
    # Two Moons + DBSCAN基准/扰动 + K-Means对照

    X, _ = make_moons(n_samples=500, noise=0.1, random_state=42)

    # baseline (eps=0.2, minPts=5)
    lab1 = my_dbscan(X, eps=0.2, min_pts=5)
    plot_scatter(X, lab1, "Fig.1  Manual DBSCAN (Eps=0.2, MinPts=5)", "fig1_dbscan_eps0.2_minpts5.png")

    #  eps too small
    lab2 = my_dbscan(X, eps=0.05, min_pts=5)
    plot_scatter(X, lab2, "Fig.2  Manual DBSCAN fragmentation (Eps=0.05)", "fig2_dbscan_eps0.05.png")

    #  eps too large
    lab3 = my_dbscan(X, eps=0.50, min_pts=5)
    plot_scatter(X, lab3, "Fig.3  Manual DBSCAN merging (Eps=0.50)", "fig3_dbscan_eps0.50.png")

    # K-Means (K=2)
    km = KMeans(n_clusters=2, n_init=10, random_state=42).fit(X)
    plot_scatter(X, km.labels_, "Fig.4  K-Means (K=2)", "fig4_kmeans_k2.png")


    # k-距离图 Two Moons (n=1000, noise=0.06), k=5

    Xk, _ = make_moons(n_samples=500, noise=0.1, random_state=42)
    plot_k_distance(Xk, k=5, fname="fig5_k_distance_k5.png")

  
    # Circles & Varied Density：DBSCAN / K-Means 对照

    # Circles
    Xc, _ = make_circles(n_samples=500, factor=0.5, noise=0.05, random_state=42)
    lab6 = my_dbscan(Xc, eps=0.2, min_pts=5)
    plot_scatter(Xc, lab6, "Fig.6  Manual DBSCAN on Circles", "fig6_dbscan_circles.png")

    kmc = KMeans(n_clusters=2, n_init=10, random_state=42).fit(Xc)
    plot_scatter(Xc, kmc.labels_, "Fig.7  K-Means on Circles", "fig7_kmeans_circles.png")

    # Varied Density
    centers = [(-2, -2), (2, 2), (2, -2)]
    stds = [0.20, 0.55, 0.30]
    Xv, _ = make_blobs(n_samples=[250, 180, 200], centers=centers, cluster_std=stds, random_state=42)

    lab8 = my_dbscan(Xv, eps=0.25, min_pts=5)
    plot_scatter(Xv, lab8, "Fig.8  Manual DBSCAN on Varied Density", "fig8_dbscan_varied_density.png")

    kmv = KMeans(n_clusters=3, n_init=10, random_state=42).fit(Xv)
    plot_scatter(Xv, kmv.labels_, "Fig.9  K-Means on Varied Density", "fig9_kmeans_varied_density.png")



    print(" Images-only reproduction done.")
    print("Output folder:", OUTDIR)


if __name__ == "__main__":
    main()