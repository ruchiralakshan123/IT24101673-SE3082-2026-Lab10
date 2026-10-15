#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define N 100000000
#define THREADS_PER_BLOCK 512

#define CHECK(call)                                                     \
    do {                                                                   \
        const cudaError_t error = (call);                                  \
        if (error != cudaSuccess) {                                        \
            printf("Error: %s:%d, ", __FILE__, __LINE__);                  \
            printf("code:%d, reason: %s\n", error,                         \
                   cudaGetErrorString(error));                             \
            exit(1);                                                       \
        }                                                                  \
    } while (0)

__global__ void vecMul(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

void random_ints(int *x, size_t size) {
    for (size_t i = 0; i < size; i++)
        x[i] = rand() % 100;
}

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return (double)tp.tv_sec + (double)tp.tv_usec * 1.e-6;
}

int main(void) {
    int *a, *b, *c, *d_a, *d_b, *d_c;
    size_t size = (size_t)N * sizeof(int);
    double kStart, kElaps, wStart, wElaps;

    CHECK(cudaMalloc((void **)&d_a, size));
    CHECK(cudaMalloc((void **)&d_b, size));
    CHECK(cudaMalloc((void **)&d_c, size));

    a = (int *)malloc(size); random_ints(a, N);
    b = (int *)malloc(size); random_ints(b, N);
    c = (int *)malloc(size);

    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    printf("Launching %d blocks x %d threads\n", blocks, THREADS_PER_BLOCK);

    // ---- whole sequence: starts before the first HostToDevice copy ----
    wStart = cpuSecond();
    CHECK(cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice));

    // ---- kernel only ----
    kStart = cpuSecond();
    vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());
    kElaps = cpuSecond() - kStart;

    // ---- whole sequence: stops after the last DeviceToHost copy ----
    CHECK(cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost));
    wElaps = cpuSecond() - wStart;

    for (int i = N - 5; i < N; i++)
        printf("%d) %d x %d = %d\n", i, a[i], b[i], c[i]);

    size_t errors = 0;
    for (size_t i = 0; i < N; i++)
        if (c[i] != a[i] * b[i]) errors++;
    printf("Verification: %zu mismatches out of %d\n", errors, N);

    printf("Kernel only time    - %.7f\n", kElaps);
    printf("Whole sequence time - %.7f\n", wElaps);
    printf("Whole / kernel ratio - %.1f\n", wElaps / kElaps);

    free(a); free(b); free(c);
    CHECK(cudaFree(d_a)); CHECK(cudaFree(d_b)); CHECK(cudaFree(d_c));
    return 0;
}
