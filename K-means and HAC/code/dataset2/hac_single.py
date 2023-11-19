import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

def make_filename(base_dir, plot_type, k=None, file_ext="png"):
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if plot_type == "cluster":
        return f"{base_dir}hac_clusters(k={k}).{file_ext}"
    elif plot_type == "dendrogram":
        return f"{base_dir}dendrogram.{file_ext}"
    else:
        raise ValueError("Invalid plot type specified")

def plot_clusters(data, labels, k, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for cluster_idx in range(k):
        cluster_points = data[labels == cluster_idx]
        ax.scatter(cluster_points.iloc[:, 0], cluster_points.iloc[:, 1], cluster_points.iloc[:, 2])

    # Save the figure
    plt.savefig(filename)
    plt.close()

def plot_dendrogram(Z, filename, max_d=None):
    plt.figure(figsize=(10, 7))
    plt.title("Hierarchical Clustering Dendrogram")
    dendrogram(Z, truncate_mode = "lastp")
    
    # Draw a line to show the cut-off (optional)
    if max_d:
        plt.axhline(y=max_d, c='k')

    plt.savefig(filename)
    plt.close()

def hac_clustering(data, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='single')
    return model.fit_predict(data)

def read_dataset(file):
    dataset = file

    return pd.read_csv(dataset)

def main():
    data = read_dataset("data/dataset2.csv")

    # Create and plot the dendrogram
    Z = linkage(data, method='single')
    dendrogram_filename = make_filename("visuals/dataset2/hac_single/dendrogram/", "dendrogram")
    plot_dendrogram(Z, dendrogram_filename)

    # Perform HAC and plot clusters for different values of k
    initial_k = 2
    final_k = 10
    for k in range(initial_k, final_k+1):
        labels = hac_clustering(data, k)
        cluster_filename = make_filename("visuals/dataset2/hac_single/cluster_graphs/", "cluster", k)
        plot_clusters(data, labels, k, cluster_filename)

if __name__ == '__main__':
    main()
