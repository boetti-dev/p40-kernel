// test_p40.cpp -- host-side verification of every numeric core used by the P40
// kernels.  No GPU is required: the kernel math lives in BONSAI_HD functions
// that compile identically on the host, so these tests exercise the exact
// arithmetic the P40 will run.
//
// Build:  g++ -O2 -std=c++17 -I src -I include tests/test_p40.cpp -o test_p40
#include "../src/pq2_0.h"
#include "../src/p40_ptx.h"
#include "../src/fwht.h"
#include "../src/ops.h"
#include "../include/bonsai_model.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using namespace bonsai;

static int g_fail = 0;
static int g_pass = 0;

#define CHECK(cond, ...) do { \
    if (!(cond)) { std::printf("  FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); ++g_fail; } \
    else { ++g_pass; } \
} while (0)

static void check_close(float a, float b, float tol, const char * what) {
    const float d = std::fabs(a - b);
    CHECK(d <= tol, "%s: got %.6g expected %.6g (|d|=%.3g > %.3g)", what, a, b, d, tol);
}

// ---------------------------------------------------------------------------
// 1. PQ2_0 format: quantise -> dequantise round trip, and the DP4A +
//    permuted-Q8_1 dot must reproduce the exact integer reference dot.
// ---------------------------------------------------------------------------
static void test_pq2_0_format() {
    std::printf("[pq2_0] format + dp4a dot\n");
    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    const int n = 5120;                       // one real qwen35 row
    std::vector<float> w(n);
    for (int i = 0; i < n; ++i) w[i] = std::round(ud(rng) * 1.2f);  // {-1,0,+1}

    std::vector<block_pq2_0> wq(n / QK_PQ2_0);
    quantize_row_pq2_0(w.data(), wq.data(), n);

    std::vector<float> wd(n);
    dequantize_row_pq2_0(wq.data(), wd.data(), n);
    double err = 0.0;
    for (int i = 0; i < n; ++i) err += std::fabs(wd[i] - w[i]);
    CHECK(err / n < 0.02, "pq2_0 ternary round-trip mean abs err %.4f", err / n);

    std::vector<float> x(n);
    for (int i = 0; i < n; ++i) x[i] = ud(rng) * 3.0f;

    std::vector<block_q8_1> a_perm(n / QK8_1), a_plain(n / QK8_1);
    quantize_row_q8_1(x.data(), a_perm.data(), n, true);
    quantize_row_q8_1(x.data(), a_plain.data(), n, false);

    // the kernel path
    const float kernel_dot = pq2_0_row_dot_perm(wq.data(), a_perm.data(), n);

    // exact integer reference over the plain layout
    double ref_exact = 0.0;
    for (int b = 0; b < n / QK_PQ2_0; ++b) {
        const double dw = f16_to_f32(wq[b].d);
        for (int iqs = 0; iqs < QI_PQ2_0; ++iqs) {
            const block_q8_1 & ab = a_plain[b * QI_PQ2_0 + iqs];
            const double da = f16_to_f32(ab.d);
            int64_t s = 0;
            for (int j = 0; j < 32; ++j) {
                const int jj = iqs * 32 + j;
                const int code = (wq[b].qs[jj >> 2] >> ((jj & 3) * 2)) & 0x03;
                s += (int64_t)(code - 1) * (int64_t)ab.qs[j];
            }
            ref_exact += dw * da * (double)s;
        }
    }
    check_close(kernel_dot, (float)ref_exact, 1e-2f, "kernel dp4a vs exact integer ref");

    // and it must be close to the true float dot (quantisation error only)
    float true_dot = 0.0f;
    for (int b = 0; b < n / QK_PQ2_0; ++b)
        true_dot += pq2_0_dot_f32(wq.data() + b, x.data() + b * QK_PQ2_0);
    CHECK(std::fabs(true_dot - kernel_dot) / (std::fabs(true_dot) + 1e-3f) < 0.05f,
          "quantised dot within 5%% of float dot (%.5f vs %.5f)", kernel_dot, true_dot);

    // permuted activations must dequantise to the same vector as the plain ones
    double l1 = 0.0;
    for (int c = 0; c < n / QK8_1; ++c) {
        for (int j = 0; j < QK8_1; ++j) {
            const int src = 16 * (j >> 4) + 4 * (j & 3) + ((j >> 2) & 3);
            l1 += std::fabs((float)a_perm[c].qs[j] - (float)a_plain[c].qs[src]);
        }
    }
    CHECK(l1 == 0.0, "permuted activation layout is a pure permutation");
}

// ---------------------------------------------------------------------------
// 2. Hadamard: orthonormal/involutive, and equal to the naive Sylvester matrix.
// ---------------------------------------------------------------------------
static void test_fwht() {
    std::printf("[fwht] Hadamard block 1024\n");
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    std::vector<float> x(HADAMARD_BLOCK);
    for (auto & v : x) v = ud(rng);

    std::vector<float> y = x;
    fwht_1024(y.data(), nullptr);
    fwht_1024(y.data(), nullptr);           // involution
    double e = 0.0;
    for (int i = 0; i < HADAMARD_BLOCK; ++i) e += std::fabs(y[i] - x[i]);
    CHECK(e / HADAMARD_BLOCK < 1e-4, "fwht is an involution (mean err %.2e)", e / HADAMARD_BLOCK);

    // naive Sylvester: H[i][j] = (-1)^popcount(i&j) / 32
    std::vector<float> ref(HADAMARD_BLOCK, 0.0f);
    for (int i = 0; i < HADAMARD_BLOCK; ++i) {
        double acc = 0.0;
        for (int j = 0; j < HADAMARD_BLOCK; ++j) {
            const int s = __builtin_popcount((unsigned)(i & j)) & 1;
            acc += (s ? -1.0 : 1.0) * (double)x[j];
        }
        ref[i] = (float)(acc / 32.0);
    }
    std::vector<float> z = x;
    fwht_1024(z.data(), nullptr);
    double m = 0.0;
    for (int i = 0; i < HADAMARD_BLOCK; ++i) m = std::fmax(m, std::fabs(z[i] - ref[i]));
    CHECK(m < 1e-3, "fwht matches naive Sylvester Hadamard (max err %.2e)", m);

    // signed variant: transform(signs*x) == fwht_signed(x)
    std::vector<float> sg(HADAMARD_BLOCK);
    for (auto & v : sg) v = (ud(rng) > 0) ? 1.0f : -1.0f;
    std::vector<float> a = x, b = x;
    fwht_1024(a.data(), sg.data());
    for (int i = 0; i < HADAMARD_BLOCK; ++i) b[i] *= sg[i];
    fwht_1024(b.data(), nullptr);
    double d2 = 0.0;
    for (int i = 0; i < HADAMARD_BLOCK; ++i) d2 += std::fabs(a[i] - b[i]);
    CHECK(d2 < 1e-3, "signed fwht equals sign-then-transform");
}

// ---------------------------------------------------------------------------
// 3. RMSNorm, RoPE, SwiGLU against direct formulas.
// ---------------------------------------------------------------------------
static void test_ops() {
    std::printf("[ops] rmsnorm / rope / swiglu\n");
    std::mt19937 rng(99);
    std::uniform_real_distribution<float> ud(-2.0f, 2.0f);

    const int n = 5120;
    std::vector<float> x(n), w(n), y(n);
    for (auto & v : x) v = ud(rng);
    for (auto & v : w) v = ud(rng) * 0.5f + 1.0f;
    const float eps = 1e-6f;
    rmsnorm(x.data(), w.data(), y.data(), n, eps);

    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double)x[i] * x[i];
    const double inv = 1.0 / std::sqrt(ss / n + eps);
    double m = 0.0;
    for (int i = 0; i < n; ++i) m = std::fmax(m, std::fabs(y[i] - x[i] * inv * w[i]));
    CHECK(m < 1e-4, "rmsnorm matches direct formula (max err %.2e)", m);

    // rope: apply to a 256-dim head over 64 rotary dims
    std::vector<float> q(256), qr(256);
    for (auto & v : q) v = ud(rng);
    qr = q;
    const int pos = 37;
    rope_apply(qr.data(), 256, 64, pos, 1.0e7f);
    double mr = 0.0;
    for (int i = 0; i < 32; ++i) {
        const double inv_f = std::pow(1.0e7, -2.0 * i / 64.0);
        const double ang = pos * inv_f;
        const double x0 = q[i], x1 = q[i + 32];
        mr = std::fmax(mr, std::fabs(qr[i] - (x0 * std::cos(ang) - x1 * std::sin(ang))));
        mr = std::fmax(mr, std::fabs(qr[i + 32] - (x0 * std::sin(ang) + x1 * std::cos(ang))));
    }
    CHECK(mr < 1e-4, "rope matches direct rotation (max err %.2e)", mr);
    // dims beyond rope_dim are untouched
    CHECK(qr[64] == q[64] && qr[255] == q[255], "rope leaves the tail dimensions untouched");

    // swiglu
    std::vector<float> g(n), u(n), o(n);
    for (auto & v : g) v = ud(rng);
    for (auto & v : u) v = ud(rng);
    swiglu(g.data(), u.data(), o.data(), n);
    double ms = 0.0;
    for (int i = 0; i < n; ++i) {
        const double s = g[i] / (1.0 + std::exp(-(double)g[i]));
        ms = std::fmax(ms, std::fabs(o[i] - s * u[i]));
    }
    CHECK(ms < 1e-5, "swiglu matches direct formula (max err %.2e)", ms);
}

// ---------------------------------------------------------------------------
// 4. Gated DeltaNet step against a double-precision analytic reference.
// ---------------------------------------------------------------------------
static void test_gdn() {
    std::printf("[gdn] gated delta net step\n");
    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    const int K = 128, V = 128;
    std::vector<float> S((size_t)K * V), q(K), k(K), v(V), out(V);
    for (auto & x : S) x = ud(rng) * 0.1f;
    for (auto & x : q) x = ud(rng);
    for (auto & x : k) x = ud(rng);
    for (auto & x : v) x = ud(rng);
    const float g = -0.37f, beta = 0.8f;

    std::vector<float> S0 = S;
    gdn_step(S.data(), q.data(), k.data(), v.data(), g, beta, out.data(), K, V);

    std::vector<double> Sr((size_t)K * V);
    for (size_t i = 0; i < S.size(); ++i) Sr[i] = (double)S0[i] * std::exp((double)g);
    std::vector<double> kv(V, 0.0);
    for (int j = 0; j < V; ++j)
        for (int i = 0; i < K; ++i) kv[j] += (double)k[i] * Sr[(size_t)i * V + j];
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < V; ++j)
            Sr[(size_t)i * V + j] += (double)beta * k[i] * ((double)v[j] - kv[j]);
    double mstate = 0.0, mout = 0.0;
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < V; ++j)
            mstate = std::fmax(mstate, std::fabs(S[(size_t)i * V + j] - Sr[(size_t)i * V + j]));
    for (int j = 0; j < V; ++j) {
        double o = 0.0;
        for (int i = 0; i < K; ++i) o += (double)q[i] * Sr[(size_t)i * V + j];
        mout = std::fmax(mout, std::fabs(out[j] - o));
    }
    CHECK(mstate < 1e-4, "gdn state matches definition (max err %.2e)", mstate);
    CHECK(mout < 1e-4, "gdn output matches definition (max err %.2e)", mout);
}

// ---------------------------------------------------------------------------
// 5. Online-softmax attention (the flash recurrence used by the kernel)
//    against a naive full-softmax reference.
// ---------------------------------------------------------------------------
static void online_attn(const float * q, const float * k, const float * v, float * o,
                        int n_tok, int head, int kvhead, int n_head, int n_head_kv,
                        int D, float scale) {
    const float * qp = q + ((size_t)(n_tok - 1) * n_head + head) * D;
    float m = -INFINITY, l = 0.0f;
    std::vector<float> acc(D, 0.0f);
    for (int j = 0; j < n_tok; ++j) {
        const float * kp = k + ((size_t)j * n_head_kv + kvhead) * D;
        float s = 0.0f;
        for (int d = 0; d < D; ++d) s += qp[d] * kp[d];
        s *= scale;
        const float m_new = std::fmax(m, s);
        const float alpha = std::exp(m - m_new);
        const float p = std::exp(s - m_new);
        l = l * alpha + p;
        const float * vp = v + ((size_t)j * n_head_kv + kvhead) * D;
        for (int d = 0; d < D; ++d) acc[d] = acc[d] * alpha + p * vp[d];
        m = m_new;
    }
    for (int d = 0; d < D; ++d) o[d] = acc[d] / l;
}

static void test_attn() {
    std::printf("[attn] online softmax vs naive\n");
    std::mt19937 rng(55);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    const int n_tok = 64, n_head = 24, n_head_kv = 4, D = 256;
    std::vector<float> q((size_t)n_tok * n_head * D), k((size_t)n_tok * n_head_kv * D),
                       v((size_t)n_tok * n_head_kv * D);
    for (auto & x : q) x = ud(rng);
    for (auto & x : k) x = ud(rng);
    for (auto & x : v) x = ud(rng);
    const float scale = 1.0f / std::sqrt((float)D);

    const int head = 5, kvhead = head / (n_head / n_head_kv);
    std::vector<float> o(D);
    online_attn(q.data(), k.data(), v.data(), o.data(), n_tok, head, kvhead, n_head, n_head_kv, D, scale);

    std::vector<double> sc(n_tok);
    double mx = -1e30;
    const float * qp = q.data() + ((size_t)(n_tok - 1) * n_head + head) * D;
    for (int j = 0; j < n_tok; ++j) {
        const float * kp = k.data() + ((size_t)j * n_head_kv + kvhead) * D;
        double s = 0.0;
        for (int d = 0; d < D; ++d) s += (double)qp[d] * kp[d];
        sc[j] = s * scale;
        mx = std::fmax(mx, sc[j]);
    }
    double Z = 0.0;
    for (int j = 0; j < n_tok; ++j) { sc[j] = std::exp(sc[j] - mx); Z += sc[j]; }
    double mmax = 0.0;
    for (int d = 0; d < D; ++d) {
        double acc = 0.0;
        for (int j = 0; j < n_tok; ++j) {
            const float * vp = v.data() + ((size_t)j * n_head_kv + kvhead) * D;
            acc += sc[j] * vp[d];
        }
        mmax = std::fmax(mmax, std::fabs(o[d] - acc / Z));
    }
    CHECK(mmax < 1e-4, "online softmax matches naive (max err %.2e)", mmax);
}
// ---------------------------------------------------------------------------
// 6. Model geometry must match the shipped GGUF header.
// ---------------------------------------------------------------------------
static void test_geometry() {
    std::printf("[model] qwen35 geometry\n");
    using M = model_27b;
    CHECK(M::n_layer == 64 && M::n_embd == 5120 && M::n_ff == 17408, "core dims");
    CHECK(M::q_dim == 24 * 256 && M::q_dim == 6144, "q_dim = n_head*head_dim = 6144");
    CHECK(M::kv_dim == 1024, "kv_dim = 4*256 = 1024");
    CHECK(M::q_proj_dim == 12288, "q_proj_dim = 2*6144 = 12288 (query+gate)");
    CHECK(M::ssm_inner == M::ssm_v_heads * M::ssm_state && M::ssm_inner == 6144, "ssm inner 6144");
    CHECK(M::ssm_qkv_dim == 2 * 2048 + M::ssm_inner && M::ssm_qkv_dim == 10240, "ssm qkv 10240");
    CHECK(M::vocab == 248320, "vocab 248320");

    int full = 0, lin = 0;
    for (int i = 0; i < M::n_layer; ++i) (M::is_full_layer(i) ? full : lin)++;
    CHECK(full == 16 && lin == 48, "layer split 16 full / 48 linear (got %d/%d)", full, lin);

    const int rows[] = {M::n_embd, M::n_ff, M::q_proj_dim, M::ssm_qkv_dim, M::ssm_inner, M::kv_dim};
    for (int r : rows) {
        CHECK(r % QK_PQ2_0 == 0, "row length %d is a PQ2_0 multiple", r);
        CHECK(r % QK8_1 == 0, "row length %d is a Q8_1 multiple", r);
    }
}

// ---------------------------------------------------------------------------
// 7. Simulate the GEMV kernel's warp loop exactly: 4 rows per warp, lane
//    striding over the PQ2_0 blocks, row clamping at the tail.  The index
//    arithmetic mirrors gemv_pq2_0_p40_kernel and must reproduce the serial
//    row dot for every row.
// ---------------------------------------------------------------------------
static void test_gemv_loop() {
    std::printf("[gemv] 4-rows-per-warp strided loop\n");
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    const int n_in = 5120;
    const int n_out = 5;                       // deliberately not a multiple of 4
    const int WARPS = 8, ROWS_PER_WARP = 4, ROWS_PER_BLOCK = WARPS * ROWS_PER_WARP;
    const int nblocks = n_in / QK_PQ2_0;
    const int last = n_out - 1;

    std::vector<float> w((size_t)n_out * n_in), x(n_in);
    for (auto & v : w) v = std::round(ud(rng) * 1.2f);
    for (auto & v : x) v = ud(rng) * 3.0f;

    std::vector<block_pq2_0> wq((size_t)n_out * nblocks);
    for (int r = 0; r < n_out; ++r)
        quantize_row_pq2_0(w.data() + (size_t)r * n_in, wq.data() + (size_t)r * nblocks, n_in);
    std::vector<block_q8_1> xq(n_in / QK8_1);
    quantize_row_q8_1(x.data(), xq.data(), n_in, true);

    const int nblk = (n_out + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    std::vector<float> out(n_out, -12345.0f);
    int rows_touched = 0;

    for (int blk = 0; blk < nblk; ++blk) {
        for (int warp = 0; warp < WARPS; ++warp) {
            const int row0 = (blk * WARPS + warp) * ROWS_PER_WARP;
            if (row0 >= n_out) continue;

            // row pointers, clamped exactly like the kernel
            const block_pq2_0 * wr[ROWS_PER_WARP];
            for (int r = 0; r < ROWS_PER_WARP; ++r)
                wr[r] = wq.data() + (size_t)std::min(row0 + r, last) * nblocks;

            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                // each lane sums its own strided subset of blocks
                float lane_acc[32];
                for (int lane = 0; lane < 32; ++lane) {
                    float acc = 0.0f;
                    for (int b = lane; b < nblocks; b += 32) {
                        const block_q8_1 * ab = xq.data() + (size_t)b * QI_PQ2_0;
                        for (int iqs = 0; iqs < QI_PQ2_0; ++iqs)
                            acc += pq2_0_dot_q8_1_perm(wr[r] + b, ab, iqs);
                    }
                    lane_acc[lane] = acc;
                }
                // the kernel reduces the 32 lane partials with shfl_down
                float total = 0.0f;
                for (int o = 16; o > 0; o >>= 1)
                    for (int l = 0; l < 32; ++l) {
                        const int src = l + o;
                        if (src < 32) lane_acc[l] += lane_acc[src];
                    }
                total = lane_acc[0];
                if (row0 + r < n_out) { out[row0 + r] = total; ++rows_touched; }
            }
        }
    }

    for (int r = 0; r < n_out; ++r) {
        float refd = 0.0f;
        for (int b = 0; b < nblocks; ++b) {
            const block_q8_1 * ab = xq.data() + (size_t)b * QI_PQ2_0;
            for (int iqs = 0; iqs < QI_PQ2_0; ++iqs)
                refd += pq2_0_dot_q8_1_perm(wq.data() + (size_t)r * nblocks + b, ab, iqs);
        }
        check_close(out[r], refd, 1e-2f, "gemv row value");
        CHECK(out[r] != -12345.0f, "gemv stored row %d", r);
    }
    CHECK(rows_touched == n_out, "every row written exactly once (%d vs %d)", rows_touched, n_out);

    int visits = 0;
    for (int lane = 0; lane < 32; ++lane)
        for (int b = lane; b < nblocks; b += 32) ++visits;
    CHECK(visits == nblocks, "lane striding covers each block exactly once (%d vs %d)", visits, nblocks);
}

// ---------------------------------------------------------------------------
// 8. Simulate the GEMM block tiling (row/col mapping + k-block loop) and
//    require an exact match against the reference matmul.
// ---------------------------------------------------------------------------
static void test_gemm_tiling() {
    std::printf("[gemm] tiled matmul\n");
    std::mt19937 rng(31337);
    std::uniform_real_distribution<float> ud(-1.0f, 1.0f);

    const int n_out = 64, n_in = 256, n_tok = 5;
    const int nblocks = n_in / QK_PQ2_0, nchunk = n_in / QK8_1;
    std::vector<float> w((size_t)n_out * n_in), x((size_t)n_in * n_tok);
    for (auto & v : w) v = std::round(ud(rng) * 1.2f);
    for (auto & v : x) v = ud(rng) * 3.0f;

    std::vector<block_pq2_0> wq((size_t)n_out * nblocks);
    for (int r = 0; r < n_out; ++r)
        quantize_row_pq2_0(w.data() + (size_t)r * n_in, wq.data() + (size_t)r * nblocks, n_in);

    // activations, chunk-major [n_in/32][n_tok], each column quantised alone
    std::vector<block_q8_1> X((size_t)nchunk * n_tok);
    for (int c = 0; c < nchunk; ++c)
        for (int t = 0; t < n_tok; ++t) {
            std::vector<float> one(QK8_1);
            for (int d = 0; d < QK8_1; ++d) one[d] = x[(size_t)(c * QK8_1 + d) * n_tok + t];
            std::vector<block_q8_1> tmp(1);
            quantize_row_q8_1(one.data(), tmp.data(), QK8_1, true);
            X[(size_t)c * n_tok + t] = tmp[0];
        }

    const int BM = 32, BN = 16, WARPS = 8, ROWS = WARPS * BM;
    std::vector<float> Y((size_t)n_out * n_tok, 0.0f);
    for (int r0 = 0; r0 < n_out; r0 += ROWS)
        for (int c0 = 0; c0 < n_tok; c0 += BN) {
            const int ncol = std::min(BN, n_tok - c0);
            for (int warp = 0; warp < WARPS; ++warp)
                for (int lane = 0; lane < 32; ++lane) {
                    const int row = r0 + warp * BM + lane;
                    if (row >= n_out) continue;
                    const block_pq2_0 * wrow = wq.data() + (size_t)row * nblocks;
                    float acc[BN] = {};
                    for (int b = 0; b < nblocks; ++b)
                        for (int iqs = 0; iqs < QI_PQ2_0; ++iqs) {
                            const block_pq2_0 * wb = wrow + b;
                            for (int j = 0; j < ncol; ++j) {
                                const block_q8_1 * ab = X.data() + (size_t)(b * QI_PQ2_0 + iqs) * n_tok + (c0 + j);
                                acc[j] += pq2_0_dot_q8_1_block(wb, ab, iqs);
                            }
                        }
                    for (int j = 0; j < ncol; ++j) Y[(size_t)row * n_tok + (c0 + j)] = acc[j];
                }
        }

    double m = 0.0;
    for (int r = 0; r < n_out; ++r) {
        std::vector<float> wd(n_in);
        dequantize_row_pq2_0(wq.data() + (size_t)r * nblocks, wd.data(), n_in);
        for (int t = 0; t < n_tok; ++t) {
            double ref = 0.0;
            for (int b = 0; b < nblocks; ++b)
                for (int iqs = 0; iqs < QI_PQ2_0; ++iqs) {
                    const block_q8_1 & a = X[(size_t)(b * QI_PQ2_0 + iqs) * n_tok + t];
                    const double da = f16_to_f32(a.d);
                    for (int j = 0; j < 32; ++j) {
                        const int jj = iqs * 32 + j;
                        const int code = (wq[(size_t)r * nblocks + b].qs[jj >> 2] >> ((jj & 3) * 2)) & 3;
                        const int src = 16 * (j >> 4) + 4 * (j & 3) + ((j >> 2) & 3);
                        ref += (double)((code - 1) * f16_to_f32(wq[(size_t)r * nblocks + b].d))
                             * (double)a.qs[src] * da;
                    }
                }
            m = std::fmax(m, std::fabs(Y[(size_t)r * n_tok + t] - ref));
        }
    }
    CHECK(m < 1e-2, "gemm tiled result matches reference (max err %.3e)", m);
}

// ---------------------------------------------------------------------------
// 9. The PTX helper layer must be behaviourally identical through its host
//    fallback path (the device paths are the same instruction, guarded by
//    __CUDA_ARCH__).  Also re-checks the mixed-signedness dp4a convention:
//    the weight operand is *unsigned* (codes 0..3), the activation *signed*.
// ---------------------------------------------------------------------------
static void test_ptx_layer() {
    std::printf("[ptx] host fallback path\n");

    // p40_dp4a_us must equal the signed dp4a for our 0..3 code range
    std::mt19937 rng(808);
    std::uniform_int_distribution<int> code(0, 3), act(-128, 127);
    int mismatches = 0;
    for (int t = 0; t < 20000; ++t) {
        uint32_t w = 0;
        int a = 0;
        for (int k = 0; k < 4; ++k) {
            w |= (uint32_t)code(rng) << (8 * k);
            a |= (int)((uint8_t)(int8_t)act(rng)) << (8 * k);
        }
        const int c = act(rng);
        if (p40_dp4a_us(w, a, c) != dp4a((int)w, a, c)) ++mismatches;
    }
    CHECK(mismatches == 0, "p40_dp4a_us matches signed dp4a for unsigned codes (%d mismatches)", mismatches);

    // p40_ldg_u16 / p40_ldg_u16x2 must be plain loads on the host
    const uint16_t buf[4] = {0x1234, 0xABCD, 0x00FF, 0xFF00};
    CHECK(p40_ldg_u16(buf) == 0x1234, "p40_ldg_u16 value");
    CHECK(p40_ldg_u16x2(buf) == (0x1234u | (0xABCDu << 16)), "p40_ldg_u16x2 packs little endian");
    p40_prefetch_l2(buf);              // must be a no-op on the host
    CHECK(true, "p40_prefetch_l2 host no-op");
}

int main() {
    test_pq2_0_format();
    test_fwht();
    test_ops();
    test_gdn();
    test_attn();
    test_geometry();
    test_gemv_loop();
    test_gemm_tiling();
    test_ptx_layer();
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
