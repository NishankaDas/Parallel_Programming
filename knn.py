from mpi4py import MPI
import pandas as pd
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ----------------------------
# Load and preprocess dataset
# ----------------------------
if rank == 0:
    df = pd.read_csv("Cancer_Data.csv")

    # Drop unnecessary columns
    df = df.drop(columns=["id", "Unnamed: 32"], errors='ignore')

    # Convert diagnosis to numeric (M=1, B=0)
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # Shuffle dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split into features and labels
    X = df.drop("diagnosis", axis=1).values
    y = df["diagnosis"].values

    # Normalize features
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Train-test split (80-20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

else:
    X_train = None
    X_test = None
    y_train = None
    y_test = None

# Broadcast data to all processes
X_train = comm.bcast(X_train, root=0)
y_train = comm.bcast(y_train, root=0)
X_test = comm.bcast(X_test, root=0)
y_test = comm.bcast(y_test, root=0)

# ----------------------------
# KNN Implementation
# ----------------------------
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict(X_train, y_train, x, k=5):
    distances = np.array([euclidean_distance(x, x_train) for x_train in X_train])
    k_indices = np.argsort(distances)[:k]
    k_labels = y_train[k_indices]

    # Majority vote
    return np.bincount(k_labels).argmax()

# ----------------------------
# Parallel Prediction
# ----------------------------
n_test = len(X_test)

# Split test data across processes
chunk_size = n_test // size
start = rank * chunk_size
end = (rank + 1) * chunk_size if rank != size - 1 else n_test

X_local = X_test[start:end]
y_local = y_test[start:end]

predictions_local = []

for x in X_local:
    pred = knn_predict(X_train, y_train, x, k=5)
    predictions_local.append(pred)

predictions_local = np.array(predictions_local)

# Gather results
predictions = comm.gather(predictions_local, root=0)

# ----------------------------
# Evaluate
# ----------------------------
if rank == 0:
    predictions = np.concatenate(predictions)

    accuracy = np.mean(predictions == y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")