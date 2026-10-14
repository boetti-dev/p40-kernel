// elementwise_p40.cu -- Hadamard rotation and RMSNorm.
//
// Both are pure streaming kernels; on the P40 they are launched with enough
// warps to saturate the GDDR5 and use 128-bit vector accesses throughout.
#include "../fwht.h"
#include "../ops.h"
#include "../p40_common.h"
#include "bonsai_p40.h"

#include <cuda_runtime.h>

using namespace bonsai;

namespace {

// One block transforms one 1024-element block in shared memory.
__global__ void fwht_1024_p40_kernel(float * __restrict__ x, const float * __restrict__ signs) {
    __shared__ float s[HADAMARD_BLOCK];
    const int b = blockIdx.x;
    float * xb = x + (size_t)b * HADAMARD_BLOCK;

    for (int i = threadIdx.x; i < HADAMARD_BLOCK; i += blockDim.x) {
        s[i] = xb[i] * (signs ? signs[i] : 1.0f);
    }
    __syncthreads();

    for (int h = 1; h < HADAMARD_BLOCK; h <<= 1) {
        for (int i = threadIdx.x; i < HADAMARD_BLOCK / 2; i += blockDim.x) {
            const int base = (i / h) * (2 * h) + (i % h);
            const float a = s[base];
            const float c = s[base + h];
            s[base]     = a + c;
            s[base + h] = a - c;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < HADAMARD_BLOCK; i += blockDim.x) {
        xb[i] = s[i] * HADAMARD_NORM;
    }
}

// One block per row; block reduction over the row.
__global__ void rmsnorm_p40_kernel(const float * __restrict__ x, const float * __restrict__ w,
                                   float * __restrict__ y, const int n, const float eps) {
    __shared__ float red[32];
    const int row = blockIdx.x;
    const float * xr = x + (size_t)row * n;
    float * yr = y + (size_t)row * n;

    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];

    // warp reduce
    for (int o = 16; o > 0; o >>= 1) ss += __shfl_down_sync(0xffffffffu, ss, o);
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = (threadIdx.x < (blockDim.x >> 5)) ? red[threadIdx.x] : 0.0f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
        if (threadIdx.x == 0) red[0] = rsqrtf(v / (float)n + eps);
    }
    __syncthreads();
    const float scale = red[0];

    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = xr[i] * scale * w[i];
}

} // namespace

int bonsai_p40_fwht(float * x, const float * signs, int64_t n, bonsai_stream stream) {
    if (n % HADAMARD_BLOCK != 0) return 1;
    const int blocks = (int)(n / HADAMARD_BLOCK);
    fwht_1024_p40_kernel<<<blocks, 256, 0, (cudaStream_t)stream>>>(x, signs);
    return (int)cudaGetLastError();
}

int bonsai_p40_rmsnorm(const float * x, const float * w, float * y,
                       int64_t n_rows, int64_t n, float eps, bonsai_stream stream) {
    const int threads = (int)((n < 1024) ? ((n + 31) / 32) * 32 : 1024);
    rmsnorm_p40_kernel<<<(unsigned)n_rows, threads, 0, (cudaStream_t)stream>>>(x, w, y, (int)n, eps);
    return (int)cudaGetLastError();
}
