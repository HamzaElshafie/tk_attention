#include "kittens.cuh"
#include "prototype.cuh"
#include "common.cuh"

#include <iostream>
#include <cuda_bf16.h>

using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

template <int B_r, int B_c, int d_model>
struct attn_layout {
    using qo_tile = st_bf<B_r, d_model>;
    using kv_tile = st_bf<B_c, d_model>;
};

struct attn_template {
    using layout = attn_layout<>;
    struct producer {};
    struct consumer {};
};