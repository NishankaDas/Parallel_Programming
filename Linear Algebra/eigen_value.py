from mpi4py import MPI
import random
import numpy as np
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# ---------- Power Iteration Eigenvalue Function ----------
def eigenvalue_power(A, iterations=100):
    n = A.shape[0]
    # Random initial vector
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)
    eigenvalue = 0
    for _ in range(iterations):
        Av = np.dot(A, v)
        # Rayleigh quotient approximation
        eigenvalue = np.dot(v, Av)
        # Normalize vector
        v = Av / np.linalg.norm(Av)
    return eigenvalue
    
# ---------- Create Random Matrix ----------
def create_matrix(a, b):
    return np.array([[random.randint(0, 9) for _ in range(b)] for _ in range(a)])
    
# ---------- Main Program ----------
if rank == 0:
    x = int(input("Enter number of rows: "))
    y = int(input("Enter number of cols: "))
    if x != y:
        print("Eigenvalues require a square matrix.")
        comm.Abort()
    A = create_matrix(x, y)
    if x<10:
        print(A)
else:
    A = None
    
start_time = time.perf_counter()

# Broadcast matrix to all processes
A = comm.bcast(A, root=0)

# Each process computes eigenvalue (parallel demonstration)
local_eigen = eigenvalue_power(A)

# Gather results at root
all_eigen = comm.gather(local_eigen, root=0)
end_time = time.perf_counter()
if rank == 0:
    print("\nEigenvalues computed by all processes:", all_eigen)
    print("Largest Eigenvalue ≈", all_eigen[0])
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
