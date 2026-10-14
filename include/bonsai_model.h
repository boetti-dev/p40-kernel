// bonsai_model.h -- the exact geometry of Ternary-Bonsai-2-27B (qwen35),
// read straight out of the shipped GGUF header.  The P40 kernel hardcodes
// these numbers on purpose: "exclusively this model" means no dynamic shape
// dispatch, no generic fallbacks, every tile/stride chosen for these values.
//
//   general.architecture            qwen35
//   block_count                     64
//   embedding_length                5120
//   feed_forward_length             17408
//   attention.head_count            24
//   attention.head_count_kv         4
//   attention.key_length            256
//   attention.value_length          256
//   full_attention_interval         4     -> 48 linear (GDN) + 16 full layers
//   ssm.conv_kernel                 4
//   ssm.state_size                  128
//   ssm.group_count                 16
//   ssm.time_step_rank              48
//   ssm.inner_size                  6144
//   rope.dimension_count            64
//   rope.freq_base                  1e7
//   prism.hadamard.block_size       1024
#pragma once

#include <cstdint>

namespace bonsai {

struct model_27b {
    static constexpr int n_layer        = 64;
    static constexpr int n_embd         = 5120;
    static constexpr int n_ff           = 17408;
    static constexpr int n_head         = 24;
    static constexpr int n_head_kv      = 4;
    static constexpr int head_dim       = 256;   // key_length == value_length
    static constexpr int q_dim          = n_head * head_dim;        // 6144
    static constexpr int kv_dim         = n_head_kv * head_dim;     // 1024
    static constexpr int q_proj_dim     = 2 * q_dim;                // 12288 (q + gate)
    static constexpr int rope_dim       = 64;
    static constexpr float rope_base    = 1.0e7f;
    static constexpr float rms_eps      = 1.0e-6f;
    static constexpr int vocab          = 248320;

    static constexpr int full_interval  = 4;
    static constexpr int n_linear       = 48;
    static constexpr int n_full         = 16;

    // linear-attention (gated delta net) geometry
    static constexpr int ssm_k_heads    = 16;    // group_count
    static constexpr int ssm_v_heads    = 48;    // inner_size / state_size
    static constexpr int ssm_state      = 128;   // state_size
    static constexpr int ssm_conv_k     = 4;
    static constexpr int ssm_inner      = 6144;  // ssm_v_heads * ssm_state
    static constexpr int ssm_qkv_dim    = 10240; // q(2048)+k(2048)+v(6144)
    static constexpr int hadamard_block = 1024;

    static constexpr bool is_full_layer(int i) { return ((i + 1) % full_interval) == 0; }
};

// Per-tensor element counts, used to size the decode GEMV grid.
struct tensor_shape {
    const char * name;
    int64_t n_out;   // rows  (ne1)
    int64_t n_in;    // cols  (ne0)
};

} // namespace bonsai
