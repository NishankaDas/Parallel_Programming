#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define N 160
#define MAX_ITER 500

int main(int argc, char *argv[]) {
    int rank, size;
    int i, j, iter;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (N % size != 0) {
        if (rank == 0)
            printf("Error: N must be divisible by number of processes\n");
        MPI_Finalize();
        return 1;
    }

    int rows = N / size;

    // Allocate contiguous memory
    double *u = (double *)malloc((rows + 2) * N * sizeof(double));
    double *u_new = (double *)malloc((rows + 2) * N * sizeof(double));

    // Access macro
    #define U(i,j) u[(i)*N + (j)]
    #define U_NEW(i,j) u_new[(i)*N + (j)]

    // Initialize
    for (i = 0; i < rows + 2; i++)
        for (j = 0; j < N; j++)
            U(i,j) = 0.0;

    // Top boundary (global)
    if (rank == 0) {
        for (j = 0; j < N; j++)
            U(1,j) = 100.0;
    }

    MPI_Status status;

    for (iter = 0; iter < MAX_ITER; iter++) {

        // Exchange ghost rows
        if (rank > 0) {
            MPI_Sendrecv(&U(1,0), N, MPI_DOUBLE, rank-1, 0,
                         &U(0,0), N, MPI_DOUBLE, rank-1, 0,
                         MPI_COMM_WORLD, &status);
        }

        if (rank < size-1) {
            MPI_Sendrecv(&U(rows,0), N, MPI_DOUBLE, rank+1, 0,
                         &U(rows+1,0), N, MPI_DOUBLE, rank+1, 0,
                         MPI_COMM_WORLD, &status);
        }

        // Compute
        #pragma omp parallel for private(j)
        for (i = 1; i <= rows; i++) {
            for (j = 1; j < N-1; j++) {
                U_NEW(i,j) = 0.25 * (U(i+1,j) + U(i-1,j) +
                                    U(i,j+1) + U(i,j-1));
            }
        }

        // Preserve boundaries
        if (rank == 0) {
            for (j = 0; j < N; j++)
                U_NEW(1,j) = 100.0;
        }

        for (i = 1; i <= rows; i++) {
            U_NEW(i,0) = 0.0;
            U_NEW(i,N-1) = 0.0;
        }

        // Copy back
        #pragma omp parallel for private(j)
        for (i = 1; i <= rows; i++) {
            for (j = 0; j < N; j++) {
                U(i,j) = U_NEW(i,j);
            }
        }
    }

    // Gather results
    double *global = NULL;
    if (rank == 0)
        global = (double *)malloc(N * N * sizeof(double));

    MPI_Gather(&U(1,0), rows * N, MPI_DOUBLE,
               global, rows * N, MPI_DOUBLE,
               0, MPI_COMM_WORLD);

    // Output
    if (rank == 0) {
        FILE *fp = fopen("heat.dat", "w");

        for (i = 0; i < N; i++) {
            for (j = 0; j < N; j++) {
                fprintf(fp, "%d %d %f\n", i, j, global[i*N + j]);
            }
            fprintf(fp, "\n");
        }

        fclose(fp);
        printf("Data written to heat.dat\n");
    }

    free(u);
    free(u_new);
    if (rank == 0) free(global);

    MPI_Finalize();
    return 0;
}
