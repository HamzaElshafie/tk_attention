#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required for plotting: pip install matplotlib") from exc


BACKEND_ORDER = ["TK (Ours)", "FlashAttention-2", "FlashAttention-3"]
COLORS = {
    "TK (Ours)": "#76B7B2",
    "FlashAttention-2": "#59A14F",
    "FlashAttention-3": "#8E6BBE",
}


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            if not row.get("median_tflops"):
                continue
            rows.append(row)
    return rows


def plot(csv_path: Path, output: Path, save_pdf: bool) -> None:
    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError(f"No successful benchmark rows found in {csv_path}")

    by_seq = defaultdict(dict)
    for row in rows:
        by_seq[int(row["seq"])][row["backend"]] = float(row["median_tflops"])

    seqs = sorted(by_seq)
    x_positions = list(range(len(seqs)))
    width = 0.24
    offsets = {
        "TK (Ours)": -width,
        "FlashAttention-2": 0.0,
        "FlashAttention-3": width,
    }

    fig, ax = plt.subplots(figsize=(12, 7))
    for backend in BACKEND_ORDER:
        values = [by_seq[seq].get(backend) for seq in seqs]
        xs = [x + offsets[backend] for x in x_positions]
        visible_xs = [x for x, value in zip(xs, values) if value is not None]
        visible_values = [value for value in values if value is not None]
        bars = ax.bar(visible_xs, visible_values, width=width, label=backend, color=COLORS[backend])
        for bar, value in zip(bars, visible_values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{value:.0f}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    first = rows[0]
    title = f"Attn Fwd (Non Causal, B={first['batch']}, H={first['heads']}, D={first['dim']})"
    ax.set_title(title, fontsize=20, pad=14)
    ax.set_xlabel("Sequence Length", fontsize=14)
    ax.set_ylabel("Throughput (TFLOP/s)", fontsize=14)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(seq) for seq in seqs], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.legend(fontsize=13)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    ymax = max(float(row["median_tflops"]) for row in rows)
    ax.set_ylim(0, ymax * 1.18)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    if save_pdf:
        fig.savefig(output.with_suffix(".pdf"))
    print(f"Wrote {output}")
    if save_pdf:
        print(f"Wrote {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot attention benchmark results")
    parser.add_argument("--csv", type=Path, default=Path("results/attention_bench.csv"))
    parser.add_argument("--output", type=Path, default=Path("results/attention_bench.png"))
    parser.add_argument("--pdf", action="store_true", help="Also save a PDF next to the PNG")
    args = parser.parse_args()
    plot(args.csv, args.output, args.pdf)


if __name__ == "__main__":
    main()
