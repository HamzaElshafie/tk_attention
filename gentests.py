#!/usr/bin/env python3
import argparse
import math
from pathlib import Path

import numpy as np
import torch


def make_inputs(test_name: str, batch: int, heads: int, seq: int, dim: int):
    shape = (batch, heads, seq, dim)

    if test_name == "ones":
        q = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
        k = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
        v = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
    elif test_name == "randn":
        torch.random.manual_seed(42)
        q = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        k = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
        v = torch.randn(shape, dtype=torch.bfloat16, device="cuda")
    elif test_name == "qk_test":
        if seq % dim != 0:
            raise ValueError("qk_test requires seq to be divisible by dim")
        eye = torch.eye(dim, dtype=torch.bfloat16, device="cuda").reshape(1, 1, dim, dim)
        q = eye.repeat(batch, heads, seq // dim, 1) * 10
        k = eye.repeat(batch, heads, seq // dim, 1) * 10
        v = eye.repeat(batch, heads, seq // dim, 1) * 10
    elif test_name == "v_orientation":
        q = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
        k = torch.ones(shape, dtype=torch.bfloat16, device="cuda")
        v = (torch.arange(dim, dtype=torch.bfloat16, device="cuda") / dim).reshape(1, 1, 1, dim)
        v = v.repeat(batch, heads, seq, 1)
    else:
        raise ValueError(f"unknown test: {test_name}")

    return q, k, v


def flatten_for_cpp(tensor: torch.Tensor) -> np.ndarray:
    return tensor.to(torch.float32).flatten().detach().cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reference data for attn_lcf.cu")
    parser.add_argument("test", choices=["ones", "randn", "qk_test", "v_orientation"])
    parser.add_argument("--batch", "-b", type=int, default=4)
    parser.add_argument("--heads", "-H", type=int, default=16)
    parser.add_argument("--seq", "-n", type=int, default=3072)
    parser.add_argument("--dim", "-d", type=int, default=128)
    parser.add_argument("--output-dir", "-o", type=Path, default=Path("testdata"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to generate bf16 reference data")

    print(f"Generating {args.test}: B={args.batch} H={args.heads} N={args.seq} D={args.dim}")
    q, k, v = make_inputs(args.test, args.batch, args.heads, args.seq, args.dim)
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"{args.test}_B{args.batch}_H{args.heads}_N{args.seq}_D{args.dim}.txt"

    tensors = [
        ("Q", flatten_for_cpp(q)),
        ("K", flatten_for_cpp(k)),
        ("V", flatten_for_cpp(v)),
        ("O_REF", flatten_for_cpp(o)),
    ]

    with output.open("wb") as f:
        for name, values in tensors:
            print(f"Writing {name}")
            np.savetxt(f, values.reshape(1, -1), fmt="%.8g", delimiter=" ", newline=" ")

    softmax_scale = 1.0 / math.sqrt(args.dim)
    print(f"Wrote {output}")
    print(f"Reference: torch scaled_dot_product_attention, is_causal=False, scale={softmax_scale:.8g}")


if __name__ == "__main__":
    main()
