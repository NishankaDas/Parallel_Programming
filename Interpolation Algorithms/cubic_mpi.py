from mpi4py import MPI
import numpy as np
import random
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


# ---------- Cubic Interpolation (Lagrange form) ----------
def cubic_interpolation(x, y, xp):
    n = len(x)

    # find segment where xp lies
    for i in range(1, n-2):
        if x[i] <= xp <= x[i+1]:
            # take 4 points
            x0, x1, x2, x3 = x[i-1], x[i], x[i+1], x[i+2]
            y0, y1, y2, y3 = y[i-1], y[i], y[i+1], y[i+2]

            # Lagrange cubic interpolation
            L0 = ((xp-x1)*(xp-x2)*(xp-x3))/((x0-x1)*(x0-x2)*(x0-x3))
            L1 = ((xp-x0)*(xp-x2)*(xp-x3))/((x1-x0)*(x1-x2)*(x1-x3))
            L2 = ((xp-x0)*(xp-x1)*(xp-x3))/((x2-x0)*(x2-x1)*(x2-x3))
            L3 = ((xp-x0)*(xp-x1)*(xp-x2))/((x3-x0)*(x3-x1)*(x3-x2))

            yp = y0*L0 + y1*L1 + y2*L2 + y3*L3
            return yp

    return None


# ---------- Generate Random Data ----------
def generate_data(n):
    x = sorted([random.uniform(0, 9) for _ in range(n)])
    y = [random.uniform(0, 9) for _ in range(n)]
    return np.array(x), np.array(y)


# ---------- Main Program ----------
if rank == 0:
    n = int(input("Enter number of data points (N >= 4): "))

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
local_result = cubic_interpolation(x, y, xp)

# Gather results
results = comm.gather(local_result, root=0)

end_time = time.perf_counter()


if rank == 0:
    print("\nResults from all processes:", results)
    print("Final Cubic Interpolated Value:", results[0])

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")
