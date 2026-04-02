#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// Function to swap two elements
void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Sequential Bubble Sort
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}

// Parallel Bubble Sort using OpenMP (Odd-Even Sort)
void parallelBubbleSort(int arr[], int n) {
    for (int i = 0; i < n; i++) {

        // Even phase
        #pragma omp parallel for
        for (int j = 0; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }

        // Odd phase
        #pragma omp parallel for
        for (int j = 1; j < n - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(&arr[j], &arr[j + 1]);
            }
        }
    }
}

// Function to copy array
void copyArray(int source[], int dest[], int n) {
    for (int i = 0; i < n; i++) {
        dest[i] = source[i];
    }
}

int main() {
    int n = 10000;  // size of array
    int *arr1 = (int *)malloc(n * sizeof(int));
    int *arr2 = (int *)malloc(n * sizeof(int));

    // Generate random data
    for (int i = 0; i < n; i++) {
        arr1[i] = rand() % 10000;
    }

    copyArray(arr1, arr2, n);

    double start, end;

    // Sequential timing
    start = omp_get_wtime();
    bubbleSort(arr1, n);
    end = omp_get_wtime();
    printf("Sequential Bubble Sort Time: %f seconds\n", end - start);

    // Parallel timing
    start = omp_get_wtime();
    parallelBubbleSort(arr2, n);
    end = omp_get_wtime();
    printf("Parallel Bubble Sort (OpenMP) Time: %f seconds\n", end - start);

    free(arr1);
    free(arr2);

    return 0;
}