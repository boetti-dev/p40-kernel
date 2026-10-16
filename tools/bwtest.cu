// bwtest.cu -- pure streaming bandwidth ceiling of the card.
//
// WHY THIS MATTERS
// ----------------
// The stock PQ2_0 decode path on the P40 reaches 157.5 GB/s, i.e. 45.5% of the
// 346 GB/s datasheet peak.  That number alone does not say whether the kernel is
// bad or the card is simply slower than its datasheet.  This tool answers that:
// it streams a buffer with a trivial read kernel and a read+write copy, so the
// ceiling is measured, not assumed.
//
//   * ceiling ~160 GB/s -> the stock kernel is already at the limit, and the
//     only lever left is moving fewer bytes (PTQ1_0, 17% smaller).
//   * ceiling ~280 GB/s -> there is real kernel headroom, and it is worth
//     profiling the stock mmvq instead of replacing it.
//
// Build (needs nvcc):
//   nvcc -O3 -std=c++17 -arch=sm_61 tools/bwtest.cu -o build/bwtest
//   ./build/bwtest            # 1 GiB buffer, both tests
//   ./build/bwtest 4096       # 4 GiB buffer (keep below free VRAM)
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(x)                                                          \
    do {                                                                       \
        cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "%s -> %s\n", #x, cudaGetErrorString(e_));    \
            return 1;                                                          \
        }                                                                      \
    } while (0)

// Grid-stride read of 16 bytes per thread, accumulated so nothing is optimised
// away.  Same access shape a weight stream has: read-only, no reuse.
__global__ void __launch_bounds__(256) read_kernel(const uint4 * __restrict__ p,
                                                   size_t n4, unsigned * __restrict__ sink) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0, 0, 0, 0);
    for (; i < n4; i += stride) {
        const uint4 v = p[i];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if ((acc.x | acc.y | acc.z | acc.w) == 0xDEADBEEFu) sink[0] = 1;   // never true
}

__global__ void __launch_bounds__(256) copy_kernel(const uint4 * __restrict__ src,
                                                   uint4 * __restrict__ dst, size_t n4) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n4; i += stride) dst[i] = src[i];
}

static double run_read(const uint4 * buf, size_t bytes, int blocks) {
    const size_t n4 = bytes / sizeof(uint4);
    unsigned * sink = nullptr;
    cudaMalloc(&sink, sizeof(unsigned));
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);

    read_kernel<<<blocks, 256>>>(buf, n4, sink);          // warm up
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 5; ++r) read_kernel<<<blocks, 256>>>(buf, n4, sink);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaFree(sink); cudaEventDestroy(a); cudaEventDestroy(b);
    return (double)bytes * 5 / (ms / 1e3) / 1e9;
}

static double run_copy(const uint4 * src, uint4 * dst, size_t bytes, int blocks) {
    const size_t n4 = bytes / sizeof(uint4);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    copy_kernel<<<blocks, 256>>>(src, dst, n4);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int r = 0; r < 5; ++r) copy_kernel<<<blocks, 256>>>(src, dst, n4);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a); cudaEventDestroy(b);
    // a copy moves bytes twice (read + write)
    return (double)bytes * 2 * 5 / (ms / 1e3) / 1e9;
}

int main(int argc, char ** argv) {
    const size_t mib = (argc > 1) ? (size_t)atoi(argv[1]) : 1024;
    const size_t bytes = mib * 1024 * 1024;

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

    const int blocks = prop.multiProcessorCount * 8;

    std::printf("device: %s, compute %d.%d, %d SMs\n", prop.name, prop.major,
                prop.minor, prop.multiProcessorCount);
    std::printf("buffer: %zu MiB, grid: %d blocks x 256 threads\n\n", mib, blocks);

    uint4 * a = nullptr, * b = nullptr;
    CUDA_CHECK(cudaMalloc(&a, bytes));
    CUDA_CHECK(cudaMalloc(&b, bytes));
    CUDA_CHECK(cudaMemset(a, 1, bytes));

    const double rd = run_read(a, bytes, blocks);
    const double cp = run_copy(a, b, bytes, blocks);

    std::printf("  read-only  : %7.1f GB/s\n", rd);
    std::printf("  read+write : %7.1f GB/s\n\n", cp);

    // Reference points measured on a real P40 for PQ2_0 decode (docs/MEASURED.md)
    const double p40_stock = 157.5, p40_rows1 = 124.7, p40_rows4 = 81.9;
    std::printf("  for comparison, PQ2_0 decode on a P40 (measured):\n");
    std::printf("    stock mmvq      %7.1f GB/s\n", p40_stock);
    std::printf("    this kernel r1  %7.1f GB/s\n", p40_rows1);
    std::printf("    this kernel r4  %7.1f GB/s\n\n", p40_rows4);

    if (rd > 0) {
        std::printf("  read-only is %.2fx the stock decode rate\n", rd / p40_stock);
        if (rd / p40_stock < 1.25) {
            std::printf("  => the CARD is the limit.  The stock kernel is done; the only\n"
                        "     lever left is moving fewer bytes: PTQ1_0 (-17%%).\n"
                        "     See docs/NEXT_STEPS.md\n");
        } else {
            std::printf("  => the card has headroom the stock kernel does not reach.\n"
                        "     Profiling the stock mmvq is worth it: the fork has no\n"
                        "     Pascal MMVQ parameter table (MMVQ_PARAMETERS_GENERIC).\n"
                        "     See docs/NEXT_STEPS.md\n");
        }
    }

    cudaFree(a); cudaFree(b);
    return 0;
}
