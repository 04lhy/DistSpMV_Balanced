"""
CSR utilities, MTX I/O, metrics computation, and sparse matrix helpers.

All matrix data is stored in CSR (Compressed Sparse Row) format:
    rowptr: int32 array of length (nrows + 1)
    colidx: int32 array of length nnz
    val:    float64 array of length nnz

Convention: 0-based indexing throughout.  MTX files use 1-based indexing
and are converted on read.
"""

from __future__ import annotations

import os
import time
import logging
from typing import Tuple, Optional

import numpy as np
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────
# Type aliases
# ──────────────────────────────────────────────────────────────────────
CSRData = Tuple[np.ndarray, np.ndarray, np.ndarray]  # (rowptr, colidx, val)


# ──────────────────────────────────────────────────────────────────────
# MTX / Matrix Market I/O
# ──────────────────────────────────────────────────────────────────────

def read_mtx(filepath: str, dtype: np.dtype = np.float64) -> csr_matrix:
    """Read a Matrix Market (.mtx) file and return a SciPy CSR matrix.

    Handles both 'coordinate' and 'array' formats; converts symmetric /
    skew-symmetric / Hermitian matrices to general form by duplicating
    off-diagonal entries.

    Parameters
    ----------
    filepath : str
        Path to the .mtx file (may be gzip-compressed; scipy handles this).
    dtype : np.dtype
        Value type (default float64).

    Returns
    -------
    csr_matrix
    """
    from scipy.io import mmread
    t0 = time.perf_counter()
    A = mmread(filepath)
    if not isinstance(A, csr_matrix):
        A = A.tocsr()
    if A.dtype != dtype:
        A = A.astype(dtype)
    A.sort_indices()
    A.sum_duplicates()
    A.eliminate_zeros()
    elapsed = time.perf_counter() - t0
    logger.info("Read %s: %d x %d, %d nnz (%.2f s)",
                os.path.basename(filepath), A.shape[0], A.shape[1],
                A.nnz, elapsed)
    return A


def extract_csr(A: csr_matrix) -> CSRData:
    """Return (rowptr, colidx, val) as contiguous C-order int32 / float64 arrays.

    The returned arrays are **copies** so the caller can mutate them safely.
    """
    return (
        A.indptr.astype(np.int32, copy=True),
        A.indices.astype(np.int32, copy=True),
        A.data.astype(np.float64, copy=True),
    )


def csr_to_matrix(rowptr: np.ndarray, colidx: np.ndarray, val: np.ndarray,
                  shape: Tuple[int, int]) -> csr_matrix:
    """Reconstruct a SciPy CSR matrix for verification."""
    return csr_matrix((val, colidx, rowptr), shape=shape)


# ──────────────────────────────────────────────────────────────────────
# Verification
# ──────────────────────────────────────────────────────────────────────

def compute_error(y_test: np.ndarray, y_ref: np.ndarray) -> float:
    """Relative L2 error: ||y_test - y_ref||_2 / ||y_ref||_2."""
    denom = np.linalg.norm(y_ref)
    if denom == 0.0:
        return np.linalg.norm(y_test)
    return float(np.linalg.norm(y_test - y_ref) / denom)


def verify_spmv(y_computed: np.ndarray, A: csr_matrix, x: np.ndarray,
                tol: float = 1e-6) -> Tuple[bool, float]:
    """Check that y_computed ≈ A @ x within tolerance."""
    y_ref = A @ x
    err = compute_error(y_computed, y_ref)
    return err < tol, err


# ──────────────────────────────────────────────────────────────────────
# GFlops
# ──────────────────────────────────────────────────────────────────────

def gflops(nnz: int, time_sec: float) -> float:
    """Compute GFlops = (2 * nnz) / (time_sec * 1e9).

    Each SpMV multiply-add pair counts as 2 floating-point operations.
    """
    if time_sec <= 0:
        return 0.0
    return (2.0 * nnz) / (time_sec * 1e9)


# ──────────────────────────────────────────────────────────────────────
# Row-wise distribution helper
# ──────────────────────────────────────────────────────────────────────

def row_partition_range(nrows: int, nprocs: int, rank: int) -> Tuple[int, int]:
    """Return (start, end) for *contiguous* row-wise 1-D distribution.

    Rows are divided as evenly as possible across `nprocs` ranks.
    """
    base = nrows // nprocs
    rem = nrows % nprocs
    if rank < rem:
        start = rank * (base + 1)
        end = start + base + 1
    else:
        start = rank * base + rem
        end = start + base
    return start, end


# ──────────────────────────────────────────────────────────────────────
# Logging helpers
# ──────────────────────────────────────────────────────────────────────

def setup_logging(rank: int, level: int = logging.INFO) -> None:
    """Configure per-rank logging with a '[rank N]' prefix."""
    fmt = f"[rank {rank:2d}] %(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(level=level, format=fmt, datefmt="%H:%M:%S")


# ──────────────────────────────────────────────────────────────────────
# Random seed
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Fix random seeds for deterministic runs."""
    np.random.seed(seed)
