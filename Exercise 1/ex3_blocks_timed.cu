#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 512

__global__ void add(int *a, int *b, int *c) {
    // Each block runs only one addition
    c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];
}

void random_ints(int *x, int size) {
    for (int i = 0; i < size; i++)
        x[i] = rand() % 100;
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main(void) {
    double dStart, dElaps;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    int size = N * sizeof(int);

    // Allocate GPU memory
    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    // Allocate host memory and create random values
    a = (int *)malloc(size);
    random_ints(a, N);

    b = (int *)malloc(size);
    random_ints(b, N);

    c = (int *)malloc(size);

    // Copy input arrays from CPU to GPU
    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    // 512 blocks, 1 thread per block
    dStart = cpuSecond();
    add<<<N, 1>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    dElaps = cpuSecond() - dStart;
    printf("Time taken - %.7f\n", dElaps);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    // Copy result back to CPU
    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    // Print all 512 results
    for (int r = 0; r < N; r++)
        printf("%d + %d = %d\n", a[r], b[r], c[r]);

    // Verify GPU answers
    int errors = 0;

    for (int r = 0; r < N; r++) {
        if (c[r] != a[r] + b[r])
            errors++;
    }

    printf("Verification: %d mismatches out of %d\n", errors, N);

    // Cleanup
    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
