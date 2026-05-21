#include "attn_lcf.cu"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#ifndef ATTN_D
#define ATTN_D 128
#endif

#ifndef ATTN_B_R
#define ATTN_B_R 64
#endif

#ifndef ATTN_B_C
#define ATTN_B_C ((ATTN_D == 64) ? 192 : 128)
#endif

#define CUDA_CHECK(expr) tk_bench_cuda_check((expr), #expr, __FILE__, __LINE__)

namespace {

struct Options {
    int batch = 16;
    int heads = 16;
    int seq = 3072;
    int warmup = 20;
    int iters = 100;
    int seed = 42;
    bool json = true;
};

void tk_bench_cuda_check(cudaError_t err, const char *expr, const char *file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line
                  << " while running " << expr << ": "
                  << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

void usage(const char *argv0) {
    std::cerr
        << "Usage: " << argv0 << " [options]\n"
        << "  --batch <int>   Batch size, default 16\n"
        << "  --heads <int>   Number of heads, default 16\n"
        << "  --seq <int>     Sequence length, default 3072\n"
        << "  --warmup <int>  Warmup iterations, default 20\n"
        << "  --iters <int>   Timed iterations, default 100\n"
        << "  --seed <int>    RNG seed, default 42\n"
        << "  --text          Print human-readable output instead of JSON\n";
}

int parse_int_arg(int argc, char **argv, int &i) {
    if (i + 1 >= argc) {
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }
    return std::atoi(argv[++i]);
}

Options parse_options(int argc, char **argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--batch") == 0) {
            options.batch = parse_int_arg(argc, argv, i);
        } else if (std::strcmp(argv[i], "--heads") == 0) {
            options.heads = parse_int_arg(argc, argv, i);
        } else if (std::strcmp(argv[i], "--seq") == 0) {
            options.seq = parse_int_arg(argc, argv, i);
        } else if (std::strcmp(argv[i], "--warmup") == 0) {
            options.warmup = parse_int_arg(argc, argv, i);
        } else if (std::strcmp(argv[i], "--iters") == 0) {
            options.iters = parse_int_arg(argc, argv, i);
        } else if (std::strcmp(argv[i], "--seed") == 0) {
            options.seed = parse_int_arg(argc, argv, i);
        } else if (std::strcmp(argv[i], "--text") == 0) {
            options.json = false;
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }
    return options;
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : ptr_(nullptr) {
        CUDA_CHECK(cudaMalloc(&ptr_, count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (ptr_ != nullptr) {
            cudaFree(ptr_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T *get() const {
        return ptr_;
    }

private:
    T *ptr_;
};

void fill_randn_bf16(std::vector<bf16> &dst, int seed, int stream_offset) {
    std::mt19937 rng(static_cast<uint32_t>(seed + 1009 * stream_offset));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (bf16 &value : dst) {
        value = __float2bfloat16(dist(rng));
    }
}

double attention_flops(int batch, int heads, int seq, int dim) {
    const double b = static_cast<double>(batch);
    const double h = static_cast<double>(heads);
    const double n = static_cast<double>(seq);
    const double d = static_cast<double>(dim);
    return 2.0 * b * h * n * n * d + 4.0 * b * h * n * n + 2.0 * b * h * n * n * d;
}

int ceil_div_host(int value, int divisor) {
    return (value + divisor - 1) / divisor;
}

double mean(std::vector<float> values) {
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

double median(std::vector<float> values) {
    std::sort(values.begin(), values.end());
    const size_t mid = values.size() / 2;
    if (values.size() % 2 == 0) {
        return 0.5 * (values[mid - 1] + values[mid]);
    }
    return values[mid];
}

} // namespace

int main(int argc, char **argv) {
    constexpr int kDim = ATTN_D;
    constexpr int kBr = ATTN_B_R;
    constexpr int kBc = ATTN_B_C;
    using kernel_template = attn_template<kBr, kBc, kDim>;
    using layout = typename kernel_template::layout;

    const Options options = parse_options(argc, argv);
    if (options.batch <= 0 || options.heads <= 0 || options.seq <= 0 || options.iters <= 0 || options.warmup < 0) {
        std::cerr << "Invalid benchmark dimensions or iteration counts." << std::endl;
        return EXIT_FAILURE;
    }

    const size_t elements = static_cast<size_t>(options.batch) * options.heads * options.seq * kDim;
    std::vector<bf16> q(elements), k(elements), v(elements);
    fill_randn_bf16(q, options.seed, 0);
    fill_randn_bf16(k, options.seed, 1);
    fill_randn_bf16(v, options.seed, 2);

    DeviceBuffer<bf16> d_q(elements), d_k(elements), d_v(elements), d_o(elements);
    CUDA_CHECK(cudaMemcpy(d_q.get(), q.data(), elements * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k.get(), k.data(), elements * sizeof(bf16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v.get(), v.data(), elements * sizeof(bf16), cudaMemcpyHostToDevice));

    typename layout::qo_gl Qg(d_q.get(), static_cast<size_t>(options.batch), static_cast<size_t>(options.heads), static_cast<size_t>(options.seq), nullptr);
    typename layout::qo_gl Og(d_o.get(), static_cast<size_t>(options.batch), static_cast<size_t>(options.heads), static_cast<size_t>(options.seq), nullptr);
    typename layout::kv_gl Kg(d_k.get(), static_cast<size_t>(options.batch), static_cast<size_t>(options.heads), static_cast<size_t>(options.seq), nullptr);
    typename layout::kv_gl Vg(d_v.get(), static_cast<size_t>(options.batch), static_cast<size_t>(options.heads), static_cast<size_t>(options.seq), nullptr);
    typename layout::globals globals = {Qg, Og, Kg, Vg};

    const unsigned long smem_size = kittens::MAX_SHARED_MEMORY - 2000;
    CUDA_CHECK(cudaFuncSetAttribute(
        prototype::lcf::kernel<kernel_template>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size
    ));

    cudaDeviceProp props{};
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    constexpr int kConsumerGroups = kernel_template::NUM_CONSUMER_WARPS / 4;
    const int total_tasks = options.batch * options.heads * ceil_div_host(options.seq, kConsumerGroups * kBr);
    const int grid_blocks = std::min(total_tasks, props.multiProcessorCount);
    constexpr int block_size = prototype::detail::NUM_THREADS_v<kernel_template>;

    for (int i = 0; i < options.warmup; ++i) {
        prototype::lcf::kernel<kernel_template><<<grid_blocks, block_size, smem_size>>>(globals);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    std::vector<float> times_ms;
    times_ms.reserve(options.iters);
    for (int i = 0; i < options.iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        prototype::lcf::kernel<kernel_template><<<grid_blocks, block_size, smem_size>>>(globals);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float elapsed_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
        times_ms.push_back(elapsed_ms);
    }

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    const double flops = attention_flops(options.batch, options.heads, options.seq, kDim);
    const double median_ms = median(times_ms);
    const double mean_ms = mean(times_ms);
    const double min_ms = *std::min_element(times_ms.begin(), times_ms.end());
    const double median_tflops = flops / (median_ms * 1e-3) / 1e12;
    const double mean_tflops = flops / (mean_ms * 1e-3) / 1e12;
    const double min_time_tflops = flops / (min_ms * 1e-3) / 1e12;

    if (options.json) {
        std::cout << std::fixed << std::setprecision(6)
                  << "{"
                  << "\"backend\":\"TK (Ours)\","
                  << "\"batch\":" << options.batch << ","
                  << "\"heads\":" << options.heads << ","
                  << "\"seq\":" << options.seq << ","
                  << "\"dim\":" << kDim << ","
                  << "\"dtype\":\"bf16\","
                  << "\"causal\":false,"
                  << "\"seed\":" << options.seed << ","
                  << "\"warmup\":" << options.warmup << ","
                  << "\"iters\":" << options.iters << ","
                  << "\"median_ms\":" << median_ms << ","
                  << "\"mean_ms\":" << mean_ms << ","
                  << "\"min_ms\":" << min_ms << ","
                  << "\"median_tflops\":" << median_tflops << ","
                  << "\"mean_tflops\":" << mean_tflops << ","
                  << "\"min_time_tflops\":" << min_time_tflops << ","
                  << "\"grid_blocks\":" << grid_blocks << ","
                  << "\"block_size\":" << block_size << ","
                  << "\"smem_bytes\":" << smem_size
                  << "}" << std::endl;
    } else {
        std::cout << std::fixed << std::setprecision(4)
                  << "TK (Ours) B=" << options.batch
                  << " H=" << options.heads
                  << " N=" << options.seq
                  << " D=" << kDim
                  << " median=" << median_ms << " ms"
                  << " throughput=" << median_tflops << " TFLOP/s" << std::endl;
    }

    return EXIT_SUCCESS;
}
