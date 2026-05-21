#!/usr/bin/env python3
import argparse
import csv
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import mean, median
from typing import Any

import torch
import torch.nn.functional as F


BACKENDS = ["TK (Ours)", "FlashAttention-2", "FlashAttention-3"]
CSV_FIELDS = [
    "backend",
    "batch",
    "heads",
    "seq",
    "dim",
    "dtype",
    "causal",
    "seed",
    "warmup",
    "iters",
    "median_ms",
    "mean_ms",
    "min_ms",
    "median_tflops",
    "mean_tflops",
    "min_time_tflops",
    "status",
    "message",
]


def attention_flops(batch: int, heads: int, seq: int, dim: int) -> float:
    return (
        2.0 * batch * heads * seq * seq * dim
        + 4.0 * batch * heads * seq * seq
        + 2.0 * batch * heads * seq * seq * dim
    )


def tflops(flops: float, ms: float) -> float:
    return flops / (ms * 1e-3) / 1e12


def make_tensors(batch: int, heads: int, seq: int, dim: int, seed: int):
    torch.manual_seed(seed)
    q = torch.randn((batch, heads, seq, dim), dtype=torch.bfloat16, device="cuda")
    k = torch.randn((batch, heads, seq, dim), dtype=torch.bfloat16, device="cuda")
    v = torch.randn((batch, heads, seq, dim), dtype=torch.bfloat16, device="cuda")
    return q, k, v


def measure_cuda_events(fn, warmup: int, iters: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        times_ms.append(start.elapsed_time(end))
    torch.cuda.synchronize()
    return times_ms


def result_from_times(
    backend: str,
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    seed: int,
    warmup: int,
    iters: int,
    times_ms: list[float],
) -> dict[str, Any]:
    flops = attention_flops(batch, heads, seq, dim)
    median_ms = median(times_ms)
    mean_ms = mean(times_ms)
    min_ms = min(times_ms)
    return {
        "backend": backend,
        "batch": batch,
        "heads": heads,
        "seq": seq,
        "dim": dim,
        "dtype": "bf16",
        "causal": False,
        "seed": seed,
        "warmup": warmup,
        "iters": iters,
        "median_ms": median_ms,
        "mean_ms": mean_ms,
        "min_ms": min_ms,
        "median_tflops": tflops(flops, median_ms),
        "mean_tflops": tflops(flops, mean_ms),
        "min_time_tflops": tflops(flops, min_ms),
        "status": "ok",
        "message": "",
    }


def unavailable_result(
    backend: str,
    batch: int,
    heads: int,
    seq: int,
    dim: int,
    seed: int,
    warmup: int,
    iters: int,
    message: str,
) -> dict[str, Any]:
    return {
        "backend": backend,
        "batch": batch,
        "heads": heads,
        "seq": seq,
        "dim": dim,
        "dtype": "bf16",
        "causal": False,
        "seed": seed,
        "warmup": warmup,
        "iters": iters,
        "median_ms": "",
        "mean_ms": "",
        "min_ms": "",
        "median_tflops": "",
        "mean_tflops": "",
        "min_time_tflops": "",
        "status": "unavailable",
        "message": message,
    }


def benchmark_fa2(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, warmup: int, iters: int) -> list[float]:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def run():
        F.scaled_dot_product_attention(q, k, v, is_causal=False)

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return measure_cuda_events(run, warmup, iters)


def load_fa3():
    try:
        import flash_attn_interface  # type: ignore

        return flash_attn_interface.flash_attn_func, ""
    except Exception as exc:
        return None, f"flash_attn_interface unavailable: {exc}"


def benchmark_fa3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, warmup: int, iters: int) -> list[float]:
    flash_attn_func, reason = load_fa3()
    if flash_attn_func is None:
        raise RuntimeError(reason)

    q_bnhd = q.transpose(1, 2).contiguous()
    k_bnhd = k.transpose(1, 2).contiguous()
    v_bnhd = v.transpose(1, 2).contiguous()

    def run():
        flash_attn_func(q_bnhd, k_bnhd, v_bnhd, causal=False)

    return measure_cuda_events(run, warmup, iters)


def benchmark_tk(
    executable: Path,
    batch: int,
    heads: int,
    seq: int,
    seed: int,
    warmup: int,
    iters: int,
) -> dict[str, Any]:
    if not executable.exists():
        raise RuntimeError(f"TK benchmark executable not found: {executable}")

    cmd = [
        str(executable),
        "--batch",
        str(batch),
        "--heads",
        str(heads),
        "--seq",
        str(seq),
        "--seed",
        str(seed),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    completed = subprocess.run(cmd, check=True, text=True, capture_output=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("TK benchmark produced no output")
    return json.loads(lines[-1])


def collect_metadata(args: argparse.Namespace) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "shapes": {
            "batch": args.batch,
            "heads": args.heads,
            "seqs": args.seqs,
            "dim": args.dim,
        },
        "seed": args.seed,
        "warmup": args.warmup,
        "iters": args.iters,
        "dtype": "bf16",
        "causal": False,
        "tk_executable": str(args.tk_executable),
    }

    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        meta["gpu_name"] = torch.cuda.get_device_name(device)
        meta["gpu_capability"] = torch.cuda.get_device_capability(device)

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            query = subprocess.run(
                [
                    nvidia_smi,
                    "--query-gpu=name,driver_version,power.limit,clocks.current.sm,clocks.max.sm,temperature.gpu",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                text=True,
                capture_output=True,
            )
            meta["nvidia_smi"] = query.stdout.strip()
        except Exception as exc:
            meta["nvidia_smi_error"] = str(exc)

    flash_attn_func, reason = load_fa3()
    meta["flashattention3_available"] = flash_attn_func is not None
    if reason:
        meta["flashattention3_message"] = reason
    return meta


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark TK attention against FA2 and FA3")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--seqs", type=int, nargs="+", default=[768, 1536, 3072, 6144, 12288])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--tk-executable", type=Path, default=Path("build/bench_tk_attn"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--skip-tk", action="store_true")
    parser.add_argument("--skip-fa2", action="store_true")
    parser.add_argument("--skip-fa3", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the attention benchmark")
    if args.dim != 128:
        print("Warning: FA paths support runtime dim, but bench_tk_attn must be compiled with matching ATTN_D.", file=sys.stderr)

    rows: list[dict[str, Any]] = []
    for seq in args.seqs:
        print(f"Benchmarking B={args.batch} H={args.heads} N={seq} D={args.dim}")
        q = k = v = None
        if not (args.skip_fa2 and args.skip_fa3):
            q, k, v = make_tensors(args.batch, args.heads, seq, args.dim, args.seed)

        if not args.skip_tk:
            try:
                row = benchmark_tk(args.tk_executable, args.batch, args.heads, seq, args.seed, args.warmup, args.iters)
                row["status"] = "ok"
                row["message"] = ""
                rows.append(row)
                print(f"  TK:  {row['median_tflops']:.2f} TFLOP/s")
            except Exception as exc:
                rows.append(unavailable_result("TK (Ours)", args.batch, args.heads, seq, args.dim, args.seed, args.warmup, args.iters, str(exc)))
                print(f"  TK unavailable: {exc}")

        if not args.skip_fa2:
            try:
                assert q is not None and k is not None and v is not None
                times = benchmark_fa2(q, k, v, args.warmup, args.iters)
                row = result_from_times("FlashAttention-2", args.batch, args.heads, seq, args.dim, args.seed, args.warmup, args.iters, times)
                rows.append(row)
                print(f"  FA2: {row['median_tflops']:.2f} TFLOP/s")
            except Exception as exc:
                rows.append(unavailable_result("FlashAttention-2", args.batch, args.heads, seq, args.dim, args.seed, args.warmup, args.iters, str(exc)))
                print(f"  FA2 unavailable: {exc}")

        if not args.skip_fa3:
            try:
                assert q is not None and k is not None and v is not None
                times = benchmark_fa3(q, k, v, args.warmup, args.iters)
                row = result_from_times("FlashAttention-3", args.batch, args.heads, seq, args.dim, args.seed, args.warmup, args.iters, times)
                rows.append(row)
                print(f"  FA3: {row['median_tflops']:.2f} TFLOP/s")
            except Exception as exc:
                rows.append(unavailable_result("FlashAttention-3", args.batch, args.heads, seq, args.dim, args.seed, args.warmup, args.iters, str(exc)))
                print(f"  FA3 unavailable: {exc}")

        del q, k, v
        torch.cuda.empty_cache()

    csv_path = args.output_dir / "attention_bench.csv"
    meta_path = args.output_dir / "attention_bench_meta.json"
    write_csv(csv_path, rows)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(collect_metadata(args), indent=2) + "\n")
    print(f"Wrote {csv_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
