"""
Matrix reordering via graph partitioning for improved data locality.

Paper reference (Section III.A): METIS is used to partition the adjacency
graph so that rows assigned to the same process have nonzeros clustered
near the diagonal, reducing off-diagonal (communication-bound) entries.

Two strategies are provided:
  1. ``metis``  — uses pymetis (requires system METIS library)
  2. ``rcm``    — Reverse Cuthill-McKee (pure scipy, no external deps)
  3. ``none``   — identity permutation (no reordering)

The module applies a **symmetric** permutation P·A·P^T, so both rows
and columns are reordered identically.  This preserves the square
structure and semantic meaning for graph Laplacians / adjacency matrices.
"""

from __future__ import annotations

import logging
import time
from typing import Tuple, Optional, List

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def reorder_matrix(A: csr_matrix, method: str = "rcm",
                   nparts: int = 1, seed: int = 42) -> Tuple[csr_matrix, np.ndarray]:
    """Apply symmetric reordering P·A·P^T and return (reordered_A, perm).

    Parameters
    ----------
    A : csr_matrix
        Input square matrix (n x n).
    method : str
        One of {"metis", "rcm", "none"}.
    nparts : int
        Number of partitions (only used for "metis").
    seed : int
        Random seed for METIS.

    Returns
    -------
    (A_reordered, perm) where perm[i] is the new global index of original row i.
    """
    n = A.shape[0]
    if method == "none" or nparts <= 1:
        logger.info("Reordering: none (identity permutation)")
        return A, np.arange(n, dtype=np.int32)

    t0 = time.perf_counter()
    if method == "metis":
        perm = _reorder_metis(A, nparts, seed)
    elif method == "rcm":
        perm = _reorder_rcm(A)
    else:
        raise ValueError(f"Unknown reordering method: {method}")

    # Build permutation matrix P: P[i, perm[i]] = 1
    # Then A_reordered = P @ A @ P.T
    P = csr_matrix((np.ones(n, dtype=np.float64),
                    (np.arange(n, dtype=np.int32), perm)),
                   shape=(n, n))
    A_reordered = P @ A @ P.T
    A_reordered.sort_indices()
    A_reordered.sum_duplicates()
    A_reordered.eliminate_zeros()

    elapsed = time.perf_counter() - t0
    logger.info("Reordering (%s): %.2f s, bandwidth reduction applied", method, elapsed)
    return A_reordered, perm.astype(np.int32)


# ──────────────────────────────────────────────────────────────────────
# METIS-based reordering
# ──────────────────────────────────────────────────────────────────────

def _reorder_metis(A: csr_matrix, nparts: int, seed: int) -> np.ndarray:
    """Partition the graph of A into nparts, then reorder by partition id.

    Requires ``pymetis`` (``pip install pymetis``) and a system METIS
    installation.
    """
    try:
        import pymetis
    except ImportError:
        raise ImportError(
            "pymetis is required for METIS reordering. "
            "Install with: pip install pymetis\n"
            "Or use --reorder rcm for a pure-scipy alternative."
        )

    n = A.shape[0]

    # Build adjacency in METIS format: for each vertex, list its neighbours
    # METIS expects 0-based indexing.  We build the adjacency from the
    # symmetric structure A + A^T to represent an undirected graph.
    A_sym = A + A.T
    A_sym.sort_indices()
    A_sym.sum_duplicates()
    A_sym.eliminate_zeros()

    adjacency: List[List[int]] = [[] for _ in range(n)]
    for i in range(n):
        row_start = A_sym.indptr[i]
        row_end = A_sym.indptr[i + 1]
        neighbours = A_sym.indices[row_start:row_end].tolist()
        # Remove self-loops for partitioning purposes
        adjacency[i] = [v for v in neighbours if v != i]

    logger.info("Calling pymetis.part_graph(K=%d, seed=%d) ...", nparts, seed)
    _, parts = pymetis.part_graph(nparts, adjacency, seed=seed)

    # Reorder: sort vertices by (partition_id, original_index)
    # This clusters rows from the same partition together
    order = sorted(range(n), key=lambda i: (parts[i], i))
    perm = np.array(order, dtype=np.int32)
    return perm


# ──────────────────────────────────────────────────────────────────────
# RCM-based reordering (no external dependencies)
# ──────────────────────────────────────────────────────────────────────

def _reorder_rcm(A: csr_matrix) -> np.ndarray:
    """Reverse Cuthill-McKee ordering using SciPy.

    RCM reduces bandwidth, which naturally clusters nonzeros near the
    diagonal — a cheaper (but less effective) alternative to METIS for
    communication reduction.
    """
    n = A.shape[0]
    # RCM works on the symmetrized structure
    A_sym = A + A.T
    A_sym.sort_indices()
    A_sym.sum_duplicates()
    A_sym.eliminate_zeros()

    perm = reverse_cuthill_mckee(A_sym, symmetric_mode=True)
    return perm.astype(np.int32)
