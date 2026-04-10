#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"

#include <iostream>
#include <cuda_bf16.h>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

#define CEIL_DIV(value, divisor) (((value) + (divisor) - 1) / (divisor))

template <int B_r, int B_c, int d_model, int NUM_CONSUMER_WGS>
struct attn_layout {
    using qo_tile = st_bf<B_r, d_model>;
    using kv_tile = st_bf<B_c, d_model>;
    using qo_gl = gl<bf16, -1, -1, -1, d_model, qo_tile>;
    using kv_gl = gl<bf16, -1, -1, -1, d_model, kv_tile>;

    // Base building block in TK is 16x16
    static_assert(B_r >= 16 && B_c >= 16, "Grouping params (B_r, B_c) do not satisfy TK's 16x16 base layout");
    static_assert(B_r % 16 == 0 && B_c % 16 == 0 , "Grouping params (B_r, B_c) are not divisible by TK's base tile layout");
    static_assert(d_model % 16 == 0, "d_model is not divisible by TK's base tile layout");

    struct globals {qo_gl Q, O; kv_gl K, V;};
    // Describes one stage of the smem input buffer. Only Kj & Vj will be streamed
    struct input_block {kv_tile K, V;};
    // Qi will be fixed against the streamed KV tiles so we will put it in a scratch_block which wont get overridden in ring buffer.
    struct scratch_block {qo_tile Q[NUM_CONSUMER_WGS];};
    struct common_state {int batch, head, base_q_tile;};
    struct consumer_state {
        col_vec<rt_fl<16, kv_tile::rows>> max_vec, norm_vec; // per-warp row stats for a local 16 x B_c score fragment
        rt_fl<16, qo_tile::cols> o_reg; // per-warp output accumulator fragment
        // current-iteration working buffers
        rt_fl<16, kv_tile::rows> attn_score;
        rt_bf<16, kv_tile::rows> bf16_attn_score;
    };
};

template <int B_r, int B_c, int d_model>
struct attn_template {
    static constexpr int NUM_CONSUMER_WARPS = 12; // 3 warp groups
    static constexpr int INPUT_PIPE_STAGES = 2;
    using layout = attn_layout<B_r, B_c, d_model, NUM_CONSUMER_WARPS/4>;
    __device__ static inline void common_setup (common_setup_args<layout> args) {
        // How many tiles? Eg N=3072, B_r=64 -> 48 query tiles
        int T_r = CEIL_DIV(args.globals.Q.rows(), B_r);
        int T_c = CEIL_DIV(args.globals.K.rows(), B_c);
        int q_rows_per_cta = (NUM_CONSUMER_WARPS/4) * B_r;
        int total_cta_per_head = CEIL_DIV(args.globals.Q.rows(), q_rows_per_cta);
        // TBC...
    }
    struct producer {};
    struct consumer {};
};