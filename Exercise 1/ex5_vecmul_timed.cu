#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 10000000
#define THREADS_PER_BLOCK 512

__global__ void vecMul(int *a, int *b, int *c, int n) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] * b[i];
    }
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

    size_t size = (size_t)N * sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (int *)malloc(size);
    random_ints(a, N);

    b = (int *)malloc(size);
    random_ints(b, N);

    c = (int *)malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    int blocks =
        (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    printf("Launching %d blocks x %d threads\n",
           blocks, THREADS_PER_BLOCK);

    dStart = cpuSecond();
    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    dElaps = cpuSecond() - dStart;
    printf("Time taken - %.7f\n", dElaps);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n",
               cudaGetErrorString(err));

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = N - 1000; i < N; i++) {
        printf("%d) %d x %d = %d\n",
               i, a[i], b[i], c[i]);
    }

    int errors = 0;

    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] * b[i])
            errors++;
    }

    printf("Verification: %d mismatches out of %d\n",
           errors, N);

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
