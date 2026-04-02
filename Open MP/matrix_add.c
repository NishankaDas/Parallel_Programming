#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main() {
    int n;

    printf("Enter size N of NxN matrix: ");
    scanf("%d", &n);

    // Dynamic allocation
    int **A = (int **)malloc(n * sizeof(int *));
    int **B = (int **)malloc(n * sizeof(int *));
    int **C_seq = (int **)malloc(n * sizeof(int *));
    int **C_par = (int **)malloc(n * sizeof(int *));

    for (int i = 0; i < n; i++) {
        A[i] = (int *)malloc(n * sizeof(int));
        B[i] = (int *)malloc(n * sizeof(int));
        C_seq[i] = (int *)malloc(n * sizeof(int));
        C_par[i] = (int *)malloc(n * sizeof(int));
    }

    // Seed for random numbers
    srand(time(NULL));

    // Generate random matrices (values 0–99)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }

    double start, end;

    // 🔹 Sequential addition
    start = omp_get_wtime();

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C_seq[i][j] = A[i][j] + B[i][j];

    end = omp_get_wtime();
    double seq_time = end - start;

    // 🔹 Parallel addition (OpenMP)
    start = omp_get_wtime();

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C_par[i][j] = A[i][j] + B[i][j];

    end = omp_get_wtime();
    double par_time = end - start;

    // (Optional) Print small matrices only
    if (n <= 5) {
        printf("\nMatrix A:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("%d ", A[i][j]);
            printf("\n");
        }

        printf("\nMatrix B:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("%d ", B[i][j]);
            printf("\n");
        }

        printf("\nResult Matrix:\n");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++)
                printf("%d ", C_seq[i][j]);
            printf("\n");
        }
    }

    // Time comparison
    printf("\nSequential Time: %f seconds\n", seq_time);
    printf("Parallel Time (OpenMP): %f seconds\n", par_time);

    // Speedup
    printf("Speedup: %f\n", seq_time / par_time);

    // Free memory
    for (int i = 0; i < n; i++) {
        free(A[i]); free(B[i]);
        free(C_seq[i]); free(C_par[i]);
    }
    free(A); free(B); free(C_seq); free(C_par);

    return 0;
}