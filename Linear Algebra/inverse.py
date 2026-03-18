from mpi4py import MPI
import random
import numpy as np
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ---------- Matrix Inverse Function (Gauss-Jordan) ----------
def matrix_inverse(A):
    n = A.shape[0]

    # Create augmented matrix [A | I]
    I = np.identity(n)
    aug = np.hstack((A.astype(float), I))

    for i in range(n):

        # Pivot element
        pivot = aug[i, i]

        if abs(pivot) < 1e-10:
            raise ValueError("Matrix is singular and cannot be inverted")

        # Normalize pivot row
        aug[i] = aug[i] / pivot

        # Eliminate other rows
        for j in range(n):
            if j != i:
                factor = aug[j, i]
                aug[j] = aug[j] - factor * aug[i]

    # Extract inverse matrix
    inverse = aug[:, n:]
    return inverse


# ---------- Create Random Matrix ----------
def create_matrix(a, b):
    return np.array([[random.randint(1, 9) for _ in range(b)] for _ in range(a)])


# ---------- Main Program ----------
if rank == 0:
    x = int(input("Enter number of rows: "))
    y = int(input("Enter number of cols: "))

    if x != y:
        print("Inverse requires a square matrix.")
        comm.Abort()

    A = create_matrix(x, y)

    if x < 10:
        print("\nGenerated Matrix:")
        print(A)

else:
    A = None


start_time = time.perf_counter()

# Broadcast matrix
A = comm.bcast(A, root=0)

# Each process computes inverse
local_inverse = matrix_inverse(A)

# Gather results
all_inverse = comm.gather(local_inverse, root=0)

end_time = time.perf_counter()


if rank == 0:
    print("\nInverse matrices computed by all processes:")
    for i, inv in enumerate(all_inverse):
        print(f"\nProcess {i}:")
        print(inv)

    print("\nFinal Inverse Matrix:")
    print(all_inverse[0])


elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
