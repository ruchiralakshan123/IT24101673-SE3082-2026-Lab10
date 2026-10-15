#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 512

__global__ void addT(int *a, int *b, int *c) {
    c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];
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

    // 1 block, 512 threads
    dStart = cpuSecond();
    addT<<<1, N>>>(d_a, d_b, d_c);
    cudaDeviceSynchronize();
    dElaps = cpuSecond() - dStart;
    printf("Time taken - %.7f\n", dElaps);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (int r = 0; r < N; r++)
        printf("%d) %d + %d = %d\n", r, a[r], b[r], c[r]);

    int errors = 0;

    for (int r = 0; r < N; r++) {
        if (c[r] != a[r] + b[r])
            errors++;
    }

    printf("Verification: %d mismatches out of %d\n", errors, N);

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
