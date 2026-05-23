# TK LCF Attention

This repo benchmarks a simple ThunderKittens pipeline templated `lcf` (load-compute-finish) forward attention kernel against FlashAttention-2 and FlashAttention-3.

Importantly, I use seeded random BF16 Q/K/V tensors so the reported throughput is not inflated by predictable inputs like all zeros or all ones following the power throttling insights in [Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data!](https://www.thonking.ai/p/strangely-matrix-multiplications)

Blog: [Dissecting ThunderKittens: Anatomy of a Compact DSL for High-Performance AI Kernels](https://hamzaelshafie.bearblog.dev/dissecting-thunderkittens-anatomy-of-a-compact-dsl-for-high-performance-ai-kernels/)

<br>

<p align="center">
  <img 
    width="800" 
    height="450" 
    alt="attention_bench" 
    src="https://github.com/user-attachments/assets/6952d2c5-59fe-4053-b4a3-3f1fdab4e6a5" 
  />
</p>

<br>

<p align="center">
  <img 
    width="647" 
    height="349" 
    alt="nvidia-smi" 
    src="https://github.com/user-attachments/assets/25536a1d-cdab-4589-8a25-e78dc8339971" 
  />
</p>

<br>

## 1) Environment

Clone this repo and ThunderKittens:

```bash
git clone https://github.com/HamzaElshafie/tk_attention.git
git clone https://github.com/HazyResearch/ThunderKittens.git ~/ThunderKittens
```

Create a conda environment and install pinned Python dependencies:

```bash
conda create -n tk-attn python=3.12 -y
conda activate tk-attn
pip install -r requirements.txt
```

Set the CUDA and ThunderKittens paths:

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=${CUDA_HOME}/bin:${PATH}
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
export THUNDERKITTENS_ROOT=~/ThunderKittens
```

Verify the machine:

```bash
nvidia-smi
nvcc --version
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
```

FlashAttention-3 is optional. Install it from the FlashAttention repo:

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
export PYTHONPATH=$PWD:${PYTHONPATH}
```

Use an H100/H800 Hopper image with an R570+ NVIDIA driver and CUDA 12.8 toolkit. Current ThunderKittens and FlashAttention-3 are much smoother on this stack than on CUDA 12.6/R560 images.

## 2) Build TK Benchmark

Build the TK attention benchmark executable:

```bash
mkdir -p build
nvcc -std=c++20 -O3 -arch=sm_90a --expt-extended-lambda --expt-relaxed-constexpr \
  -DKITTENS_SM90 \
  -I${THUNDERKITTENS_ROOT} \
  -I${THUNDERKITTENS_ROOT}/include \
  -I${THUNDERKITTENS_ROOT}/prototype \
  bench_tk_attn.cu -lcuda -o build/bench_tk_attn
```

The default benchmark build is for `D=128`, `B_r=64`, `B_c=128`. Override these at compile time if needed:

```bash
nvcc -std=c++20 -O3 -arch=sm_90a --expt-extended-lambda --expt-relaxed-constexpr \
  -DKITTENS_SM90 \
  -DATTN_D=128 -DATTN_B_R=64 -DATTN_B_C=128 \
  -I${THUNDERKITTENS_ROOT} \
  -I${THUNDERKITTENS_ROOT}/include \
  -I${THUNDERKITTENS_ROOT}/prototype \
  bench_tk_attn.cu -lcuda -o build/bench_tk_attn
```

## 3) Correctness Data

Use `gentests.py` to generate PyTorch SDPA reference data for the standalone correctness harness in `attn_lcf.cu`:

```bash
python3 gentests.py randn --batch 4 --heads 16 --seq 3072 --dim 128
```

Available correctness cases:

```text
ones
randn
qk_test
v_orientation
```

These cases are for debugging correctness and layout issues. (Taken from TK's repo)

## 4) Benchmark

Run a small smoke test first:

```bash
python3 bench_attention.py \
  --batch 1 --heads 1 --seqs 128 --dim 128 \
  --warmup 2 --iters 5 \
  --tk-executable build/bench_tk_attn
```

Run the default non-causal BF16 attention benchmark:

```bash
python3 bench_attention.py \
  --batch 16 --heads 16 --dim 128 \
  --seqs 768 1536 3072 6144 12288 \
  --warmup 20 --iters 100 \
  --tk-executable build/bench_tk_attn \
  --output-dir results
```

The benchmark compares:

- `TK (Ours)`: `bench_tk_attn.cu`
- `FlashAttention-2`: PyTorch SDPA forced to `SDPBackend.FLASH_ATTENTION`
- `FlashAttention-3`: `flash_attn_interface.flash_attn_func` when installed

Outputs:

- `results/attention_bench.csv`: median/mean/min latency and TFLOP/s.
- `results/attention_bench_meta.json`: GPU, CUDA/PyTorch, seed, shape, clock/power metadata when available.

## 5) Plot

Generate the grouped bar chart:

```bash
python3 plot_attention_bench.py \
  --csv results/attention_bench.csv \
  --output results/attention_bench.png \
  --pdf
```

The plot uses median TFLOP/s:
