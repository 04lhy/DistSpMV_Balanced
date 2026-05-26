"""
DistSpMV_Balanced — Main entry point
======================================

Orchestrates the full pipeline:

  1. Read Matrix Market file (rank 0), broadcast CSR to all ranks.
  2. (Optional) Reorder matrix via METIS / RCM / identity.
  3. Distribute rows across MPI processes (contiguous 1-D).
  4. Algorithm 1 — diagonal block boundary expansion.
  5. Algorithm 2 — communication schedule construction.
  6. Algorithms 3+4 — repeated SpMV benchmarking.

Usage::

  mpiexec -n 4 python -m src.main \
      --matrix data/cant.mtx \
      --threads 4 \
      --reorder rcm \
      --benchmark 50 \
      --output results/cant_p4.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
from scipy.sparse import csr_matrix, issparse

# Ensure the src package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    read_mtx, extract_csr, verify_spmv,
    row_partition_range, gflops, set_seed, setup_logging,
)
from src.reordering import reorder_matrix
from src.partition import diagonal_block_expand
from src.comm_setup import build_schedule
from src.spmv_mpi_omp import DistSpMVSolver

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DistSpMV_Balanced — Distributed Sparse Matrix-Vector Multiplication",
    )
    p.add_argument("--matrix", required=True, help="Path to .mtx file")
    p.add_argument("--threads", type=int, default=1,
                   help="Number of threads per rank for local SpMV (default: 1)")
    p.add_argument("--reorder", choices=["metis", "rcm", "none"], default="rcm",
                   help="Matrix reordering method (default: rcm)")
    p.add_argument("--benchmark", type=int, default=50,
                   help="Number of SpMV repetitions for timing (default: 50)")
    p.add_argument("--warmup", type=int, default=5,
                   help="Number of warmup SpMV calls (default: 5)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")
    p.add_argument("--output", default="",
                   help="JSON output path for metrics (optional)")
    p.add_argument("--no-verify", action="store_true",
                   help="Skip correctness verification against scipy")
    p.add_argument("--verbose", action="store_true",
                   help="Enable DEBUG-level logging")
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def run_pipeline(comm, args: argparse.Namespace) -> dict:
    from mpi4py import MPI

    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    set_seed(args.seed)

    t_total_start = time.perf_counter()

    # ── Step 1: Load matrix (rank 0 reads, broadcasts CSR) ──
    if rank == 0:
        A_orig = read_mtx(args.matrix)
        nrows, ncols = A_orig.shape
        nnz_global = A_orig.nnz
        logger.info("Global matrix: %d x %d, %d nnz", nrows, ncols, nnz_global)
    else:
        A_orig = None
        nrows = ncols = nnz_global = 0

    nrows = comm.bcast(nrows, root=0)
    ncols = comm.bcast(ncols, root=0)
    nnz_global = comm.bcast(nnz_global, root=0)

    # ── Step 2: Reordering ──
    t_reorder_start = time.perf_counter()

    if rank == 0:
        A_reordered, perm = reorder_matrix(
            A_orig, method=args.reorder, nparts=nprocs, seed=args.seed,
        )
        A_reordered.sort_indices()
        A_reordered.eliminate_zeros()
        nrows_g, ncols_g = A_reordered.shape
        nnz_g = A_reordered.nnz
        rowptr_g, colidx_g, val_g = extract_csr(A_reordered)
    else:
        A_reordered = None
        perm = None
        nrows_g = nrows
        ncols_g = ncols
        nnz_g = 0

    nrows_g = comm.bcast(nrows_g, root=0)
    ncols_g = comm.bcast(ncols_g, root=0)
    nnz_g = comm.bcast(nnz_g, root=0)

    # Broadcast CSR arrays
    if rank != 0:
        rowptr_g = np.empty(nrows_g + 1, dtype=np.int32)
    comm.Bcast(rowptr_g, root=0)

    if rank != 0:
        colidx_g = np.empty(nnz_g, dtype=np.int32)
        val_g = np.empty(nnz_g, dtype=np.float64)
    comm.Bcast(colidx_g, root=0)
    comm.Bcast(val_g, root=0)

    t_reorder = time.perf_counter() - t_reorder_start

    # ── Step 3: Row-wise distribution ──
    r_start, r_end = row_partition_range(nrows_g, nprocs, rank)
    nlocal = r_end - r_start

    # Extract local CSR
    local_rowptr = rowptr_g[r_start:r_end + 1].copy()
    local_rowptr -= local_rowptr[0]  # 0-based for local indexing

    local_nnz = int(local_rowptr[-1])
    start_nnz = int(rowptr_g[r_start])
    end_nnz = start_nnz + local_nnz
    local_colidx = colidx_g[start_nnz:end_nnz].copy()
    local_val = val_g[start_nnz:end_nnz].copy()

    logger.info(
        "Local partition: rows [%d, %d) nlocal=%d nnz=%d",
        r_start, r_end, nlocal, local_nnz,
    )

    # ── Step 4: Algorithm 1 — diagonal block expansion ──
    t_algo1_start = time.perf_counter()
    left, right = diagonal_block_expand(
        local_rowptr, local_colidx, r_start, r_end, ncols_g, comm,
    )
    t_algo1 = time.perf_counter() - t_algo1_start

    # ── Step 5: Algorithm 2 — communication schedule ──
    t_algo2_start = time.perf_counter()
    sendid, recvid, recv_idx_map = build_schedule(
        local_rowptr, local_colidx, r_start, r_end, left, right, comm,
    )
    t_algo2 = time.perf_counter() - t_algo2_start

    # ── Step 6: Algorithms 3+4 — Build solver ──
    solver = DistSpMVSolver(
        local_rowptr, local_colidx, local_val,
        r_start, r_end, left, right,
        sendid, recvid, recv_idx_map,
        comm, n_threads=args.threads,
    )

    # ── Step 7: Prepare shared x vector ──
    np.random.seed(args.seed)
    x_global = np.random.rand(ncols_g).astype(np.float64)
    x_global = comm.bcast(x_global, root=0)

    # ── Step 8: Benchmark ──
    t_preprocess = time.perf_counter() - t_total_start

    avg_time, std_time = solver.multiply_benchmark(
        x_global, n_warmup=args.warmup, n_repeat=args.benchmark,
    )

    gflops_val = gflops(nnz_global, avg_time)

    # ── Step 9: Correctness verification ──
    is_correct, error = True, 0.0
    if not args.no_verify:
        y_local = solver.multiply(x_global)

        # Gather all y_local pieces to rank 0
        recv_counts = np.array(comm.gather(nlocal, root=0) or [], dtype=np.int32)
        if rank == 0:
            y_gathered = np.empty(nrows_g, dtype=np.float64)
            displs = np.zeros(nprocs, dtype=np.int32)
            displs[1:] = np.cumsum(recv_counts[:-1])
            comm.Gatherv(
                y_local,
                [y_gathered, (recv_counts, displs), MPI.DOUBLE],
                root=0,
            )
            # Verify against scipy reference
            if A_reordered is not None:
                y_ref = A_reordered @ x_global
                is_correct, error = verify_spmv(y_gathered, A_reordered, x_global)
                logger.info(
                    "Correctness: %s (relative error = %.2e)",
                    "PASS" if is_correct else "FAIL", error,
                )
        else:
            comm.Gatherv(y_local, None, root=0)

        is_correct = comm.bcast(is_correct, root=0)
        error = comm.bcast(error, root=0)

    # ── Step 10: Report ──
    if rank == 0:
        report = {
            "matrix": os.path.basename(args.matrix),
            "nrows": nrows_g,
            "ncols": ncols_g,
            "nnz_global": nnz_global,
            "nprocs": nprocs,
            "n_threads": args.threads,
            "reorder": args.reorder,
            "correct": is_correct,
            "error": error,
            "preprocess_time_s": t_preprocess,
            "reorder_time_s": t_reorder,
            "algo1_time_s": t_algo1,
            "algo2_time_s": t_algo2,
            "avg_spmv_time_s": avg_time,
            "std_spmv_time_s": std_time,
            "gflops": gflops_val,
            "diag_boundary_left": int(left),
            "diag_boundary_right": int(right),
            "nnz_local": int(local_nnz),
            "nnz_diag": int(solver.nnz_diag),
            "nnz_offdiag": int(solver.nnz_offdiag),
            "comm_volume_recv": int(sum(len(v) for v in recvid.values())),
            "comm_volume_send": int(sum(len(v) for v in sendid.values())),
        }
        logger.info(
            "Results: avg_time=%.6f s  GFlops=%.4f  diag_nnz=%d  offdiag_nnz=%d",
            avg_time, gflops_val, solver.nnz_diag, solver.nnz_offdiag,
        )

        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(report, f, indent=2)
            logger.info("Metrics written to %s", args.output)

        return report

    return {}


# ──────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(rank, level)

    run_pipeline(comm, args)


if __name__ == "__main__":
    main()
