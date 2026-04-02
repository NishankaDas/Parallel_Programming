from mpi4py import MPI
import pandas as pd
import numpy as np
import sys
from sklearn.decomposition import PCA

# ----------------------------
# MPI Setup
# ----------------------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Import matplotlib only on master
if rank == 0:
    import matplotlib.pyplot as plt

# ----------------------------
# Parameters
# ----------------------------
if len(sys.argv) > 1:
    K = int(sys.argv[1])
else:
    K = 2

max_iters = 100

# ----------------------------
# Load Data (only master)
# ----------------------------
if rank == 0:
    df = pd.read_csv("Cancer_Data.csv")
    df = df.drop(columns=["id", "Unnamed: 32"], errors="ignore")
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    X = df.drop("diagnosis", axis=1).values

    # Normalize
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    n_samples = X.shape[0]

    # PCA for 2D plotting
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Random centroid initialization
    indices = np.random.choice(n_samples, K, replace=False)
    centroids = X[indices]

else:
    X = None
    X_2d = None
    centroids = None
    pca = None

# Broadcast data
X = comm.bcast(X, root=0)
centroids = comm.bcast(centroids, root=0)

n_samples = X.shape[0]

# ----------------------------
# Divide data among processes
# ----------------------------
chunk = n_samples // size
start = rank * chunk
end = (rank + 1) * chunk if rank != size - 1 else n_samples
X_local = X[start:end]

# ----------------------------
# K-Means Loop
# ----------------------------
for iteration in range(max_iters):
    local_labels = []

    for x in X_local:
        distances = np.sqrt(np.sum((centroids - x) ** 2, axis=1))
        cluster = np.argmin(distances)
        local_labels.append(cluster)

    local_labels = np.array(local_labels)

    local_sum = np.zeros_like(centroids)
    local_count = np.zeros(K)

    for i, label in enumerate(local_labels):
        local_sum[label] += X_local[i]
        local_count[label] += 1

    global_sum = np.zeros_like(centroids)
    global_count = np.zeros(K)

    comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
    comm.Allreduce(local_count, global_count, op=MPI.SUM)

    new_centroids = np.copy(centroids)
    for k in range(K):
        if global_count[k] > 0:
            new_centroids[k] = global_sum[k] / global_count[k]

    if np.allclose(centroids, new_centroids, atol=1e-4):
        centroids = new_centroids
        break

    centroids = new_centroids

# ----------------------------
# Final labels
# ----------------------------
local_labels = []
for x in X_local:
    distances = np.sqrt(np.sum((centroids - x) ** 2, axis=1))
    cluster = np.argmin(distances)
    local_labels.append(cluster)

local_labels = np.array(local_labels)

all_labels = comm.gather(local_labels, root=0)

# ----------------------------
# Output + Plots
# ----------------------------
if rank == 0:
    final_labels = np.concatenate(all_labels)

    print("\nFinal Centroids:\n", centroids)

    # Plot preparation only on rank 0
    centroid_2d = pca.transform(centroids)

    # Before clustering
    plt.figure(figsize=(8, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], alpha=0.7)
    plt.title("Before Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(True)
    plt.show()

    # After clustering
    plt.figure(figsize=(8, 6))
    for k in range(K):
        plt.scatter(X_2d[final_labels == k, 0], X_2d[final_labels == k, 1], alpha=0.7, label=f"Cluster {k}")

    plt.scatter(centroid_2d[:, 0], centroid_2d[:, 1], marker='X', s=200, c='red', label='Centroids')
    plt.title("After Clustering")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.grid(True)
    plt.show()
