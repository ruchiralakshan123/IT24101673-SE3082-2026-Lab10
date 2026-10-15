# SE3082 Lab 10 - CUDA (Google Colab Edition)

**Student name:** Senarath D.M.G.R.L

**Student ID:** IT24101673

CUDA practical exercises for **SE3082 - Parallel Computing Lab Sheet 10**, continuing Lab 9.
Target GPU: NVIDIA Tesla T4 on Google Colab (`-arch=sm_75`). The notebook
[`SE3082_Lab10_IT24101673.ipynb`](./SE3082_Lab10_IT24101673.ipynb) contains every program, the
run cells, and the written answers. Use **Runtime -> Run all** on a T4 runtime.

## Project structure

| Folder | Contents |
| --- | --- |
| `Exercise 1/` | `ex1_timed.cu` (timed 512-block vector add), `ex1_nosync.cu` (no `cudaDeviceSynchronize()` experiment), `ex3_blocks_timed.cu`, `ex4_threads_timed.cu`, `ex5_vecmul_timed.cu`, `ex6_matrix_timed.cu` (timed Lab 9 programs, Part b), `ex1c_copy_vs_kernel.cu` (kernel-only vs whole-sequence time for 100 M elements, Part c) |
| `Exercise 2/` | `ex2_check.cu` (all 5 TODOs wrapped in `CHECK()`), `ex2_expA_threads.cu` (2048 threads -> code 9), `ex2_expB_alloc.cu` (1 PB allocation -> code 2), `ex2_expC_nocheck.cu` (plain program, no CHECK) |
| `Exercise 3/` | `ex3_picture.cu` - PictureKernel on an 800 x 600 picture, 16 x 16 blocks, 50 x 38 grid, saturating at 255 |
| `Exercise 4/` | `ex4_volume.cu` - element-wise product of two 500 x 500 x 500 volumes, 8 x 8 x 8 blocks, 63 x 63 x 63 grid |

## Run

```bash
nvcc -arch=sm_75 ex2_check.cu -o ex2_check && ./ex2_check
```

(For another GPU drop the dot from the compute capability, e.g. 8.9 -> `-arch=sm_89`.)
The last notebook cell zips all `.cu` files and the captured `*_output.txt` files.

## Expected self-check values

| Exercise | Expected |
| --- | --- |
| 1 / 2 | `Verification: 0 mismatches out of 512`, time ~ 0.00000xx s |
| 3 | `Grid: 50 x 38 blocks of 16 x 16 threads`, 486,400 threads, 0 mismatches out of 480000, about 240,000 pixels capped |
| 4 | `Grid: 63 x 63 x 63 blocks`, `C[499][498][0] = 76 x 15 = 1140`, `C[499][499][499] = 47 x 68 = 3196`, 0 mismatches out of 125000000 |

Timings depend on the machine, so run each timed program three times and compare.
