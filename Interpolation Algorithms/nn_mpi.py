from mpi4py import MPI
import numpy as np
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ---------- Nearest Neighbour Interpolation ----------
def nearest_neighbor(x, y, xp):
    nearest_index = 0
    min_dist = abs(x[0] - xp)

    for i in range(1, len(x)):
        dist = abs(x[i] - xp)
        if dist < min_dist:
            min_dist = dist
            nearest_index = i

    return y[nearest_index]


# ---------- Generate Random Data ----------
def generate_data(n):
    x = sorted([random.uniform(0, 9) for _ in range(n)])
    y = [random.uniform(0, 9) for _ in range(n)]
    return np.array(x), np.array(y)


# ---------- Main Program ----------
if rank == 0:
    n = int(input("Enter number of data points (N): "))

    x, y = generate_data(n)

    print("\nGenerated x values:")
    print(x)

    print("\nGenerated y values:")
    print(y)

    xp = float(input("\nEnter interpolation point xp: "))

else:
    x = None
    y = None
    xp = None


start_time = time.perf_counter()

# Broadcast data
x = comm.bcast(x, root=0)
y = comm.bcast(y, root=0)
xp = comm.bcast(xp, root=0)

# Each process computes interpolation
local_result = nearest_neighbor(x, y, xp)

# Gather results
results = comm.gather(local_result, root=0)

end_time = time.perf_counter()


if rank == 0:
    print("\nResults from all processes:", results)
    print("Final Nearest Neighbour Value:", results[0])

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
