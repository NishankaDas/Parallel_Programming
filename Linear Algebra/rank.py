from mpi4py import MPI
import random
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# ---------- Matrix Rank Function (Gaussian Elimination) ----------
def matrix_rank(A):
    A = A.astype(float)
    m, n = A.shape
    rank_val = 0
    row = 0

    for col in range(n):
        pivot = None
        for r in range(row, m):
            if abs(A[r, col]) > 1e-10:
                pivot = r
                break
        if pivot is not None:
            # Swap rows
            A[[row, pivot]] = A[[pivot, row]]
            # Eliminate below
            for r in range(row + 1, m):
                factor = A[r, col] / A[row, col]
                A[r, col:] -= factor * A[row, col:]
            row += 1
            rank_val += 1
        if row == m:
            break
    return rank_val
# ---------- Create Random Matrix ----------
def create_matrix(a, b):
    return np.array([[random.randint(0, 9) for _ in range(b)] for _ in range(a)])
# ---------- Main Program ----------
if rank == 0:
    x = int(input("Enter number of rows: "))
    y = int(input("Enter number of cols: "))
    A = create_matrix(x, y)

    print("\nGenerated Matrix:")
    print(A)
else:
    A = None
start_time = time.perf_counter()
# Broadcast matrix to all processes
A = comm.bcast(A, root=0)
# Each process computes rank
local_rank = matrix_rank(A)
# Gather results at root
all_ranks = comm.gather(local_rank, root=0)
end_time = time.perf_counter()
if rank == 0:
    print("\nComputed ranks from all processes:", all_ranks)
    print("Final Rank:", all_ranks[0])
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
