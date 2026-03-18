from mpi4py import MPI
import random
import numpy as np
import time


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------- Power Iteration Eigenvector Function ----------
def eigenvector_power(A, iterations=100):
    n = A.shape[0]
    # random initial vector
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(iterations):
        Av = np.dot(A, v)

        # normalize vector
        v = Av / np.linalg.norm(Av)

    return v

# ---------- Create Random Matrix ----------
def create_matrix(a, b):
    return np.array([[random.randint(0, 9) for _ in range(b)] for _ in range(a)])

# ---------- Main Program ----------
if rank == 0:
    x = int(input("Enter number of rows: "))
    y = int(input("Enter number of cols: "))

    if x != y:
        print("Eigenvectors require a square matrix.")
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

# Each process computes eigenvector
local_vec = eigenvector_power(A)

# Gather results
all_vecs = comm.gather(local_vec, root=0)
end_time = time.perf_counter()
if rank == 0:
    print("\nEigenvectors computed by all processes:")
    for i, v in enumerate(all_vecs):
        print(f"Process {i}:", v)
    print("\nDominant Eigenvector ≈", all_vecs[0])
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
