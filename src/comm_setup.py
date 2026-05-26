"""
Algorithm 2: Communication Information Collection
==================================================

Paper reference: Section III.C, Algorithm 2.

Given the diagonal block boundaries from Algorithm 1, this module builds
the communication schedule: which vector elements each process must send
to / receive from every other process.

Key data structures built here:

  sendid[p] : np.ndarray[int32]
      Column indices that *this* process must send to process p.
      These are columns we own (our initial row range) that p needs.

  recvid[p] : np.ndarray[int32]
      Column indices that *this* process must receive from process p.
      These are columns p owns that appear in our off-diagonal block.

  recv_idx_map[p] : Dict[int, int]
      Maps global column → local position in the per-process recv buffer.
      Used during unpacking (Algorithm 3).

Column ownership is determined by the **initial row distribution**:
column c is owned by the process whose r_start ≤ c < r_end.  This is
deterministic and invariant under boundary expansion.

Communication pattern (implemented inside ``build_schedule``):

  1. MPI_Allgather — share all processes' (left, right, r_start, r_end).
  2. Local scan — build recvid per remote process, deduplicating columns.
  3. MPI_Alltoall — exchange recv *counts* (how many columns we need from
     each peer).  This tells each process how many columns it must *send*.
  4. MPI_Isend / MPI_Irecv — exchange the actual column-index lists.
  5. Reconstruct sendid from received lists.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────

def build_schedule(
    rowptr: np.ndarray,        # int32, shape (nlocal+1,)
    colidx: np.ndarray,        # int32, shape (nnz_local,)
    r_start: int,
    r_end: int,
    left: int,
    right: int,
    comm,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], np.ndarray]:
    """Build communication schedule (Algorithm 2).

    Parameters
    ----------
    rowptr, colidx : np.ndarray
        CSR data for locally-owned rows.
    r_start, r_end : int
        This process's initial global row range (determines column ownership).
    left, right : int
        Diagonal block boundaries from Algorithm 1.
    comm :
        MPI communicator.

    Returns
    -------
    sendid : Dict[int, np.ndarray]
        sendid[q] = columns this rank must send to rank q (int32).
    recvid : Dict[int, np.ndarray]
        recvid[q] = columns this rank must receive from rank q (int32).
    recv_idx_map : np.ndarray of dict
        recv_idx_map[q] maps global col → local position in recv buffer for q.
    """
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    nlocal = r_end - r_start

    # ── Step 1: Allgather boundaries & initial row ranges ──
    my_info = np.array([left, right, r_start, r_end], dtype=np.int64)
    all_info = np.empty((nprocs, 4), dtype=np.int64)
    comm.Allgather(my_info, all_info)

    all_left = all_info[:, 0]
    all_right = all_info[:, 1]
    all_r_start = all_info[:, 2]
    all_r_end = all_info[:, 3]

    # ── Step 2: Build recvid (what we need from each remote process) ──
    # recv_sets[q] = set of global columns we need from process q
    recv_sets: Dict[int, set] = {}
    for q in range(nprocs):
        if q != rank:
            recv_sets[q] = set()

    for i in range(nlocal):
        for j in range(rowptr[i], rowptr[i + 1]):
            c = int(colidx[j])
            if left <= c < right:
                continue  # diagonal — locally available, no comm needed
            owner = _find_owner(c, all_r_start, all_r_end)
            if owner == rank:
                # We own this column even though it's outside our diagonal
                # block — this can happen due to boundary expansion.
                continue
            if owner not in recv_sets:
                recv_sets[owner] = set()
            recv_sets[owner].add(c)

    # Convert sets to sorted int32 arrays; build index map
    recvid: Dict[int, np.ndarray] = {}
    recv_idx_map: Dict[int, Dict[int, int]] = {}
    for q, cols in recv_sets.items():
        if len(cols) == 0:
            continue
        sorted_cols = np.array(sorted(cols), dtype=np.int32)
        recvid[q] = sorted_cols
        # Map col → position in this per-peer buffer
        recv_idx_map[q] = {int(c): idx for idx, c in enumerate(sorted_cols)}

    # ── Step 3: Alltoall to exchange counts ──
    # send_counts[q] = |recvid[q]| (how many columns we need from q)
    send_counts = np.zeros(nprocs, dtype=np.int32)
    for q, cols in recvid.items():
        send_counts[q] = len(cols)
    recv_counts = np.zeros(nprocs, dtype=np.int32)
    comm.Alltoall(send_counts, recv_counts)
    # recv_counts[q] = how many columns process q needs from US
    # (i.e. |sendid[q]| we need to build)

    # ── Step 4: Exchange column-index lists via point-to-point ──
    # We send recvid[q] to process q; we receive sendid[q] from process q.
    sendid: Dict[int, np.ndarray] = {}

    from mpi4py import MPI

    requests: list = []

    # Post receives for incoming column lists (what others need from us)
    recv_buffers: Dict[int, np.ndarray] = {}
    for q in range(nprocs):
        if q == rank:
            continue
        cnt = int(recv_counts[q])
        if cnt > 0:
            buf = np.empty(cnt, dtype=np.int32)
            recv_buffers[q] = buf
            req = comm.Irecv(buf, source=q, tag=100 + q)
            requests.append(req)

    # Post sends of our recvid lists to each peer
    for q, cols in recvid.items():
        req = comm.Isend(cols, dest=q, tag=100 + rank)
        requests.append(req)

    # Wait for all exchanges to complete
    MPI.Request.Waitall(requests)

    # Convert received buffers into sendid
    for q, buf in recv_buffers.items():
        sendid[q] = buf

    # ── Statistics ──
    total_recv = sum(len(v) for v in recvid.values())
    total_send = sum(len(v) for v in sendid.values())
    logger.info(
        "Algorithm 2: send to %d peers (%d elems), recv from %d peers (%d elems)",
        len(sendid), total_send, len(recvid), total_recv,
    )

    return sendid, recvid, recv_idx_map


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

def _find_owner(col: int, all_r_start: np.ndarray, all_r_end: np.ndarray) -> int:
    """Return the rank that owns global column *col*.

    Ownership follows the initial row distribution: process q owns
    columns [all_r_start[q], all_r_end[q]).
    """
    for q in range(len(all_r_start)):
        if all_r_start[q] <= col < all_r_end[q]:
            return q
    # Fallback: column outside any range (shouldn't happen for square matrices)
    # Assign to last process
    return len(all_r_start) - 1
