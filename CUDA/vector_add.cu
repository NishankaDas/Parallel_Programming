#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(float *A, float *B, float *C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] * B[i];
    }
}

int main() {
    int n = 999999999;
    size_t size = n * sizeof(float);

    // Host vectors
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize input vectors
    for (int i = 0; i < n; i++) {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 23.0f;
    }

    // Device vectors
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy data to GPU
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // Copy result back
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print some results
    std::cout << "Sample output:\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_A[i] << " + " << h_B[i]
                  << " = " << h_C[i] << std::endl;
    }

    // Free memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
