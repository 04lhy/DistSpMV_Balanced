"""
Algorithm 1: Diagonal Block Column Boundary Expansion
======================================================

Paper reference: Section III.B, Algorithm 1, Fig. 3.

Given a row-wise distributed CSR matrix, this module determines for each
MPI process the *diagonal block* column range [left, right) that balances:

  - **Communication**: columns outside [left, right) require remote vector
    elements (MPI messages).
  - **Computation**: each row should have at least ``lower_bound`` nonzeros
    inside the diagonal block so that local work is balanced.

The algorithm proceeds in four steps:

  S1. Count diag / off-diag nonzeros per local row using initial boundaries
      [r_start, r_end).
  S2. Compute ``lower_bound = Allreduce_max(max_i diag_nnz[i])``, capped at N.
  S3. Greedily expand the left / right boundary one column at a time,
      always choosing the side that benefits more *deficient* rows.
  S4. Return (left, right) for this rank.

After Algorithm 1, every process knows which columns it "owns".  Algorithm 2
then builds the communication schedule.
"""

from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def diagonal_block_expand(
    rowptr: np.ndarray,       # int32, shape (nlocal+1,)
    colidx: np.ndarray,       # int32, shape (nnz_local,)
    r_start: int,             # first global row owned by this process
    r_end: int,               # one past last global row
    ncols: int,               # total columns in the global matrix (N)
    comm,                     # MPI intra-node communicator
) -> Tuple[int, int]:
    """Run Algorithm 1 and return (left_boundary, right_boundary).

    Parameters
    ----------
    rowptr : np.ndarray
        CSR row pointers for locally-owned rows, length = nlocal + 1.
    colidx : np.ndarray
        CSR column indices for locally-owned rows.
    r_start : int
        First global row index owned by this rank (inclusive).
    r_end : int
        One past the last global row index (exclusive).
    ncols : int
        Number of columns in the global matrix.
    comm :
        MPI communicator (e.g. MPI.COMM_WORLD).

    Returns
    -------
    (left, right) : Tuple[int, int]
        Column range [left, right) that forms the diagonal block for this
        process.  Columns in this range are *locally owned*; all other
        columns require communication.
    """
    nlocal = r_end - r_start
    if nlocal == 0:
        return 0, 0

    # ── S1: Count diag / off-diag per row (initial boundaries) ──
    left = r_start
    right = min(r_end, ncols)

    total_nnz_per_row = np.diff(rowptr).astype(np.int32)  # length nlocal

    diag, off_left, off_right = _count_blocks(
        rowptr, colidx, nlocal, left, right,
    )

    # ── S2: lower_bound = max(max_i diag_nnz[i]), reduced across ranks ──
    local_max = int(diag.max()) if nlocal > 0 else 0
    lower_bound = comm.allreduce(local_max, op=_mpi_max_op(comm))
    lower_bound = min(lower_bound, ncols)
    logger.info(
        "Algorithm 1 init: left=%d right=%d local_max_diag=%d lower_bound=%d",
        left, right, local_max, lower_bound,
    )

    # ── S3: Greedy boundary expansion ──
    # Pre-build column→rows index so _update_diag_left/right and
    # _count_rows_with_col are O(rows_with_col) instead of O(nlocal).
    col_to_rows = _build_col_to_rows(rowptr, colidx, nlocal)

    max_iter = ncols  # safety cap
    for iteration in range(max_iter):
        # Identify rows still below the lower bound.
        # Cap the per-row target at total_nnz_per_row[i] —
        # a row cannot gain more diag nonzeros than it has nonzeros.
        target = np.minimum(lower_bound, total_nnz_per_row)
        deficit = np.maximum(target - diag, 0)
        deficient_rows = np.where(deficit > 0)[0]

        if len(deficient_rows) == 0:
            logger.info("Algorithm 1 converged after %d iterations", iteration)
            break

        deficient_set = set(deficient_rows.tolist())

        # ── Count how many deficient rows would benefit from expanding
        #     left by one column or right by one column ──
        left_gain = 0
        if left > 0:
            left_gain = _count_rows_with_col(
                col_to_rows, deficient_set, left - 1,
            )
        right_gain = 0
        if right < ncols:
            right_gain = _count_rows_with_col(
                col_to_rows, deficient_set, right,
            )

        if left_gain == 0 and right_gain == 0:
            logger.info(
                "Algorithm 1 stalled at iter %d: %d rows still deficient",
                iteration, len(deficient_rows),
            )
            break

        # Expand in the direction that helps more deficient rows
        if left_gain >= right_gain and left > 0:
            left -= 1
            _update_diag_left(col_to_rows, left, diag, off_left)
        elif right < ncols:
            _update_diag_right(col_to_rows, right, diag, off_right)
            right += 1
        else:
            break  # cannot expand either direction

    else:
        logger.warning("Algorithm 1 hit iteration limit (%d)", max_iter)

    logger.info(
        "Algorithm 1 result: left=%d right=%d  (initial=[%d,%d))",
        left, right, r_start, min(r_end, ncols),
    )
    return left, right


# ──────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────

def _count_blocks(
    rowptr: np.ndarray, colidx: np.ndarray,
    nlocal: int, left: int, right: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Count diag, off_left, off_right nonzeros per local row.

    - diag[i]:     nonzeros with left <= col < right
    - off_left[i]: nonzeros with col < left
    - off_right[i]:nonzeros with col >= right
    """
    diag = np.zeros(nlocal, dtype=np.int32)
    off_left = np.zeros(nlocal, dtype=np.int32)
    off_right = np.zeros(nlocal, dtype=np.int32)

    for i in range(nlocal):
        for j in range(rowptr[i], rowptr[i + 1]):
            c = colidx[j]
            if c < left:
                off_left[i] += 1
            elif c >= right:
                off_right[i] += 1
            else:
                diag[i] += 1

    return diag, off_left, off_right


def _build_col_to_rows(
    rowptr: np.ndarray, colidx: np.ndarray,
    nlocal: int,
) -> dict:
    """Build a mapping from column index to list of local rows containing it.

    Returns a dict {col: np.ndarray of local row indices}.
    Only columns that appear in the local partition get an entry.
    """
    from collections import defaultdict

    col_buckets: defaultdict[int, list[int]] = defaultdict(list)
    for i in range(nlocal):
        for j in range(rowptr[i], rowptr[i + 1]):
            col_buckets[colidx[j]].append(i)

    return {c: np.array(rows, dtype=np.int32) for c, rows in col_buckets.items()}


def _count_rows_with_col(
    col_to_rows: dict,
    deficient_set: set,
    col: int,
) -> int:
    """Count how many deficient rows have a nonzero in column *col*."""
    row_list = col_to_rows.get(col)
    if row_list is None:
        return 0
    return sum(1 for r in row_list if r in deficient_set)


def _update_diag_left(
    col_to_rows: dict,
    col: int,
    diag: np.ndarray,
    off_left: np.ndarray,
) -> None:
    """Column *col* just moved from off_left into diag.  Update counters."""
    row_list = col_to_rows.get(col)
    if row_list is not None:
        for i in row_list:
            off_left[i] -= 1
            diag[i] += 1


def _update_diag_right(
    col_to_rows: dict,
    col: int,
    diag: np.ndarray,
    off_right: np.ndarray,
) -> None:
    """Column *col* just moved from off_right into diag.  Update counters."""
    row_list = col_to_rows.get(col)
    if row_list is not None:
        for i in row_list:
            off_right[i] -= 1
            diag[i] += 1


def _mpi_max_op(comm):
    """Return MPI.MAX for the given communicator.

    If MPI is not available (e.g. in unit-test mocks), returns None;
    the mock's allreduce should ignore the op parameter.
    """
    try:
        from mpi4py import MPI
        return MPI.MAX
    except (ImportError, RuntimeError):
        return None  # for mock comms in unit tests
