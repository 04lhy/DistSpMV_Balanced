#!/usr/bin/env python
"""
Visualization script for DistSpMV_Balanced experimental results.

Reads JSON result files produced by ``run_exp.sh`` and generates:
  1. GFlops vs. number of processes (line chart)
  2. Communication volume heatmap (processes x matrix)
  3. Speedup / parallel efficiency curves
  4. Preprocessing time breakdown (stacked bar)

Usage::

  python scripts/plot_results.py --results results/ --output figures/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

# Optional imports — provide helpful error messages if missing
try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import seaborn as sns
    sns.set_style("whitegrid")
    HAS_SNS = True
except ImportError:
    HAS_SNS = False


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────

def load_results(results_dir: str) -> List[dict]:
    """Load all JSON result files from *results_dir*."""
    records = []
    for fname in sorted(os.listdir(results_dir)):
        if fname.endswith(".json"):
            path = os.path.join(results_dir, fname)
            with open(path) as f:
                records.append(json.load(f))
    return records


def pivot(records: List[dict]) -> Tuple[Dict, List[str], List[int]]:
    """Organize records by matrix name and process count.

    Returns (data[matrix][nprocs] = gflops, matrix_names, proc_counts).
    """
    data: Dict[str, Dict[int, float]] = defaultdict(dict)
    matrices = set()
    procs = set()
    for r in records:
        m = r["matrix"]
        p = r["nprocs"]
        matrices.add(m)
        procs.add(p)
        data[m][p] = r.get("gflops", 0.0)
    return data, sorted(matrices), sorted(procs)


# ──────────────────────────────────────────────────────────────────────
# Plotting functions
# ──────────────────────────────────────────────────────────────────────

def plot_gflops(records: List[dict], outdir: str) -> None:
    """GFlops vs process count, one line per matrix."""
    if not HAS_MPL:
        print("[SKIP] matplotlib not installed")
        return

    data, matrices, procs = pivot(records)

    fig, ax = plt.subplots(figsize=(10, 6))
    for m in matrices:
        xs, ys = [], []
        for p in procs:
            if p in data[m]:
                xs.append(p)
                ys.append(data[m][p])
        if len(xs) > 1:
            ax.plot(xs, ys, "o-", linewidth=1.5, markersize=5, label=m)

    ax.set_xlabel("Number of MPI Processes", fontsize=12)
    ax.set_ylabel("GFlops", fontsize=12)
    ax.set_title("DistSpMV_Balanced: GFlops vs Process Count", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "gflops_vs_procs.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] gflops_vs_procs.png")


def plot_speedup(records: List[dict], outdir: str) -> None:
    """Speedup S(p) = T(1) / T(p).  Ideal = dashed diagonal."""
    if not HAS_MPL:
        print("[SKIP] matplotlib not installed")
        return

    data, matrices, procs = pivot(records)
    # Compute speedup: use gflops ratio as proxy: S(p) = GFlops(p) / GFlops(1)
    fig, ax = plt.subplots(figsize=(10, 6))

    for m in matrices:
        base = data[m].get(1)
        if base is None or base == 0:
            continue
        xs, ys = [], []
        for p in procs:
            if p in data[m]:
                xs.append(p)
                ys.append(data[m][p] / base)
        if len(xs) > 1:
            ax.plot(xs, ys, "o-", linewidth=1.5, markersize=5, label=m)

    # Ideal speedup line
    ax.plot(procs, procs, "k--", linewidth=1, alpha=0.5, label="Ideal")

    ax.set_xlabel("Number of MPI Processes", fontsize=12)
    ax.set_ylabel("Speedup S(p)", fontsize=12)
    ax.set_title("DistSpMV_Balanced: Speedup", fontsize=14)
    ax.legend(fontsize=8, ncol=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "speedup.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] speedup.png")


def plot_comm_heatmap(records: List[dict], outdir: str) -> None:
    """Communication volume heatmap: processes x matrix."""
    if not HAS_MPL:
        print("[SKIP] matplotlib not installed")
        return

    matrices = sorted(set(r["matrix"] for r in records))
    procs = sorted(set(r["nprocs"] for r in records))

    # Build heatmap data: average comm_volume_recv per matrix-process pair
    hm = np.zeros((len(matrices), len(procs)))
    for i, m in enumerate(matrices):
        for j, p in enumerate(procs):
            vals = [r.get("comm_volume_recv", 0) for r in records
                    if r["matrix"] == m and r["nprocs"] == p]
            hm[i, j] = np.mean(vals) if vals else 0

    fig, ax = plt.subplots(figsize=(12, max(6, len(matrices) * 0.4)))
    im = ax.imshow(hm, aspect="auto", cmap="YlOrRd")

    ax.set_xticks(range(len(procs)))
    ax.set_xticklabels(procs)
    ax.set_yticks(range(len(matrices)))
    ax.set_yticklabels(matrices, fontsize=9)
    ax.set_xlabel("MPI Processes", fontsize=12)
    ax.set_title("Communication Volume (recv elements) Heatmap", fontsize=14)

    # Annotate cells
    for i in range(len(matrices)):
        for j in range(len(procs)):
            if hm[i, j] > 0:
                ax.text(j, i, f"{hm[i,j]:.0f}", ha="center", va="center",
                        fontsize=7, color="white" if hm[i, j] > hm.max() / 2 else "black")

    fig.colorbar(im, ax=ax, label="Recv Volume")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "comm_heatmap.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] comm_heatmap.png")


def plot_preprocess_breakdown(records: List[dict], outdir: str) -> None:
    """Stacked bar chart of preprocessing time components."""
    if not HAS_MPL:
        print("[SKIP] matplotlib not installed")
        return

    # Aggregate by matrix (first occurrence only, or average)
    matrices = sorted(set(r["matrix"] for r in records))

    reorder_times = []
    algo1_times = []
    algo2_times = []

    for m in matrices:
        vals = [r for r in records if r["matrix"] == m]
        # Take the first entry for each matrix
        r = vals[0]
        reorder_times.append(r.get("reorder_time_s", 0))
        algo1_times.append(r.get("algo1_time_s", 0))
        algo2_times.append(r.get("algo2_time_s", 0))

    x = np.arange(len(matrices))
    width = 0.6

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x, reorder_times, width, label="Reordering")
    ax.bar(x, algo1_times, width, bottom=reorder_times, label="Algorithm 1")
    bottoms = np.array(reorder_times) + np.array(algo1_times)
    ax.bar(x, algo2_times, width, bottom=bottoms, label="Algorithm 2")

    ax.set_xticks(x)
    ax.set_xticklabels(matrices, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Time (s)", fontsize=12)
    ax.set_title("Preprocessing Time Breakdown", fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "preprocess_breakdown.png"), dpi=150)
    plt.close(fig)
    print(f"[OK] preprocess_breakdown.png")


# ──────────────────────────────────────────────────────────────────────
# Text report
# ──────────────────────────────────────────────────────────────────────

def print_summary_table(records: List[dict]) -> None:
    """Print a Markdown-formatted summary table to stdout."""
    matrices = sorted(set(r["matrix"] for r in records))
    procs = sorted(set(r["nprocs"] for r in records))

    print("\n## Performance Summary\n")
    header = "| Matrix | " + " | ".join(f"P={p}" for p in procs) + " |"
    sep = "|--------|" + "|".join("--------|" for _ in procs)
    print(header)
    print(sep)

    for m in matrices:
        cells = [m]
        for p in procs:
            vals = [r["gflops"] for r in records
                    if r["matrix"] == m and r["nprocs"] == p]
            if vals:
                cells.append(f"{np.mean(vals):.3f}")
            else:
                cells.append("—")
        print("| " + " | ".join(cells) + " |")

    print("\n*GFlops = (2 × nnz) / (avg_time × 10⁹)*\n")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate plots and summary from DistSpMV_Balanced results",
    )
    p.add_argument("--results", default="results",
                   help="Directory containing JSON result files")
    p.add_argument("--output", default="figures",
                   help="Output directory for plots")
    args = p.parse_args()

    if not os.path.isdir(args.results):
        print(f"ERROR: results directory '{args.results}' not found")
        sys.exit(1)

    records = load_results(args.results)
    if not records:
        print(f"ERROR: no JSON files found in '{args.results}'")
        sys.exit(1)

    print(f"Loaded {len(records)} result records")

    os.makedirs(args.output, exist_ok=True)

    if not HAS_MPL:
        print("\n[WARN] matplotlib not installed — skipping plots.")
        print("  Install with: pip install matplotlib seaborn\n")

    plot_gflops(records, args.output)
    plot_speedup(records, args.output)
    plot_comm_heatmap(records, args.output)
    plot_preprocess_breakdown(records, args.output)
    print_summary_table(records)

    print(f"\nAll plots saved to {args.output}/")


if __name__ == "__main__":
    main()
