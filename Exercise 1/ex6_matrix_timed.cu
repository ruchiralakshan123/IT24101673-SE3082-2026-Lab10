#include <cstdio>
#include <cstdlib>
#include <sys/time.h>

#define ROWS 10000
#define COLS 10000

// Element-wise product: C[r][c] = A[r][c] * B[r][c]  (NOT matrix multiplication)
__global__ void mulElem(int *a, int *b, int *c, int rows, int cols) {

    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;
        c[idx] = a[idx] * b[idx];
    }
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
    double dStart, dElaps;
    int *a, *b, *c;
    int *d_a, *d_b, *d_c;

    size_t count = (size_t)ROWS * COLS;
    size_t size = count * sizeof(int);

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    a = (int *)malloc(size);
    random_ints(a, count);

    b = (int *)malloc(size);
    random_ints(b, count);

    c = (int *)malloc(size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 16);

    dim3 numBlocks(
        (COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,
        (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y
    );

    printf("Grid: %d x %d blocks\n", numBlocks.x, numBlocks.y);

    dStart = cpuSecond();
    mulElem<<<numBlocks, threadsPerBlock>>>(
        d_a, d_b, d_c, ROWS, COLS
    );
    cudaDeviceSynchronize();
    dElaps = cpuSecond() - dStart;
    printf("Time taken - %.7f\n", dElaps);

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess)
        printf("Kernel launch failed: %s\n",
               cudaGetErrorString(err));

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    for (size_t r = count - 1000; r < count; r++) {
        printf("C[%zu][%zu] = %d x %d = %d\n",
               r / COLS,
               r % COLS,
               a[r],
               b[r],
               c[r]);
    }

    size_t errors = 0;

    for (size_t r = 0; r < count; r++) {
        if (c[r] != a[r] * b[r])
            errors++;
    }

    printf("Verification: %zu mismatches out of %zu\n",
           errors, count);

    free(a);
    free(b);
    free(c);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
