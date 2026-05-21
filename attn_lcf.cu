#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"

#include <iostream>
#include <cuda_bf16.h>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

#define CEIL_DIV(value, divisor) (((value) + (divisor) - 1) / (divisor))

template <int D>
__device__ __forceinline__ float attn_temperature_scale() {
    return rsqrtf(static_cast<float>(D)) * 1.44269504089f;
}

template <>
__device__ __forceinline__ float attn_temperature_scale<64>() {
    return 0.125f * 1.44269504089f;
}

template <>
__device__ __forceinline__ float attn_temperature_scale<128>() {
    return 0.08838834764f * 1.44269504089f;
}

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
        col_vec<rt_fl<16, kv_tile::rows>> max_vec_last_scaled, max_vec_scaled;
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
        int q_rows_per_task = (NUM_CONSUMER_WARPS/4) * B_r;
        int tasks_per_head = CEIL_DIV(args.globals.Q.rows(), q_rows_per_task);
        int total_tasks = args.globals.Q.batch() * args.globals.Q.depth() * tasks_per_head;
        int task_id = gridDim.x * args.task_iter + blockIdx.x;

        if (task_id < total_tasks) {
            args.common.batch = task_id / (tasks_per_head * args.globals.Q.depth());
            args.common.head = (task_id % (tasks_per_head * args.globals.Q.depth())) / tasks_per_head;
            int query_band = (task_id % (tasks_per_head * args.globals.Q.depth())) % tasks_per_head; // CTA query-chunk index within (batch, head)
            args.common.base_q_tile = query_band * (NUM_CONSUMER_WARPS/4); // first query tile index handled by this CTA
        } else {
            args.num_iters = -1;
            return;
        }
        args.num_iters = CEIL_DIV(args.globals.K.rows(), B_c);
    }
    struct producer {
        __device__ static inline void setup(producer_setup_args<layout> args) {
            warpgroup::producer_registers(); // deallocate registers
        }
        __device__ static inline void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == 0) { // technically only one thread issues the load
                warp::tma::expect(args.inputs_arrived, args.input);
                warp::tma::load_async(
                    args.input.K, 
                    args.globals.K, 
                    {args.common.batch, args.common.head, args.iter, 0}, 
                    args.inputs_arrived);
                warp::tma::load_async(
                    args.input.V, 
                    args.globals.V, 
                    {args.common.batch, args.common.head, args.iter, 0}, 
                    args.inputs_arrived);
            } else if(laneid() == 0) arrive(args.inputs_arrived);
        }
    };
    struct consumer {
        __device__ static inline void setup(consumer_setup_args<layout> args) {
            warpgroup::consumer_registers<NUM_CONSUMER_WARPS/4>(); // allocate more registers
            // Query tile idx WG handles?
            int q_tile_idx = args.common.base_q_tile + warpgroup::groupid();
            if (q_tile_idx * layout::qo_tile::rows < args.globals.Q.rows()) {
                warpgroup::load(
                    args.scratch.Q[warpgroup::groupid()],
                    args.globals.Q,
                    {args.common.batch, args.common.head, q_tile_idx, 0}
                );
            }
            // Initialise consumer WG running states
            args.state.max_vec = base_types::constants<float>::neg_infty();
            args.state.norm_vec = 0.0f;
            args.state.o_reg = 0.0f;
            warpgroup::sync(warpgroup::groupid());
        }
        __device__ static inline void compute(consumer_compute_args<layout> args) {
            const float temperature_scale = attn_temperature_scale<d_model>();
            // S = Q * K.T
            warpgroup::mm<transpose::N, transpose::T>(
                args.state.attn_score,
                args.scratch.Q[warpgroup::groupid()],
                args.input.K
            );
            args.state.max_vec_last_scaled = args.state.max_vec * temperature_scale;
            warpgroup::mma_async_wait();

            // Mask the tail of the final KV tile.
            warp::right_fill(
                args.state.attn_score,
                args.state.attn_score,
                args.globals.K.rows() - args.iter * B_c,
                base_types::constants<float>::neg_infty()
            );
            args.state.max_vec = warp::max<axis::COL>(args.state.attn_score, args.state.max_vec);
            args.state.max_vec_scaled = args.state.max_vec * temperature_scale;
            args.state.attn_score = warp::exp2((args.state.attn_score * temperature_scale) - args.state.max_vec_scaled);
            args.state.max_vec_last_scaled = warp::exp2(args.state.max_vec_last_scaled - args.state.max_vec_scaled);

            args.state.norm_vec *= args.state.max_vec_last_scaled;
            args.state.norm_vec = warp::sum<axis::COL>(args.state.attn_score, args.state.norm_vec);
            args.state.o_reg *= args.state.max_vec_last_scaled;
            args.state.bf16_attn_score = args.state.attn_score;

            warpgroup::mma<transpose::N, transpose::N>(
                args.state.o_reg,
                args.state.bf16_attn_score,
                args.input.V
            );
            warpgroup::mma_async_wait();
            if (laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static inline void finish(consumer_finish_args<layout> args) {
            int q_tile_idx = args.common.base_q_tile + warpgroup::groupid();
            if (q_tile_idx * layout::qo_tile::rows < args.globals.Q.rows()) {
                args.state.o_reg /= args.state.norm_vec;
                auto &o_smem = reinterpret_cast<typename layout::qo_tile&>(args.scratch.Q[warpgroup::groupid()]);
                warpgroup::store(o_smem, args.state.o_reg);
                warpgroup::sync(warpgroup::groupid());
                if (warpgroup::warpid() == 0) {
                    warp::tma::store_async(
                        args.globals.O,
                        o_smem,
                        {args.common.batch, args.common.head, q_tile_idx, 0}
                    );
                }
                warp::tma::store_async_read_wait();
            }
            __syncwarp();
            if (laneid() == 0) arrive(args.finish_finished);
        }
    };
};
