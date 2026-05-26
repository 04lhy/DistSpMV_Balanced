"""
Algorithms 3 & 4: MPI Communication + Multi-threaded SpMV
===========================================================

Paper reference: Section III.D (Algorithm 3) and Section III.E (Algorithm 4).

Algorithm 3 — MPI Communication Phase
--------------------------------------
Exchanges remote vector elements between processes using point-to-point
messages.  The schedule was precomputed by Algorithm 2.

Communication pattern (per SpMV call)::

  1. Pack send buffers:  for each peer q, gather x[sendid[q][k]].
  2. Post MPI_Irecv for incoming data from all peers.
  3. Post MPI_Isend for outgoing data to all peers.
  4. WaitAll — unpack received data into the unified x-buffer.

Algorithm 4 — OpenMP Multi-threaded Local SpMV
-----------------------------------------------
Divides local rows among threads so that each thread processes
approximately the same number of nonzeros (nnz-balanced scheduling).

Each thread accumulates into a **private** y array; a final reduction
sums thread-local contributions.  This avoids locks / atomics inside
the inner loop.

Unified x-buffer layout
-----------------------
To avoid per-nonzero conditional branches in the hot loop, we use a
single flat ``x_buf``::

  x_buf[0 .. diag_len)              — locally available x elements
  x_buf[diag_len .. diag_len+R)     — remote x elements, packed by peer

A precomputed ``local_pos`` array (length = nnz_local) maps each
nonzero column to its position in ``x_buf``, so the SpMV kernel is
simply::

  for j in range(rowptr[i], rowptr[i+1]):
      acc += val[j] * x_buf[local_pos[j]]
"""

from __future__ import annotations

import logging
import time
from typing import Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# DistSpMV_Balanced solver class
# ──────────────────────────────────────────────────────────────────────

class DistSpMVSolver:
    """Holds precomputed state for repeated distributed SpMV operations.

    After calling ``setup()`` once, ``multiply(x_global)`` can be called
    many times (e.g. for iterative solvers or benchmarking).

    Parameters
    ----------
    rowptr, colidx, val : np.ndarray
        CSR data for locally-owned rows.
    r_start, r_end : int
        This rank's global row range.
    left, right : int
        Diagonal block boundaries from Algorithm 1.
    sendid, recvid, recv_idx_map :
        Communication schedule from Algorithm 2.
    n_threads : int
        Number of OpenMP / Python threads for local SpMV.
    """

    def __init__(
        self,
        rowptr: np.ndarray,
        colidx: np.ndarray,
        val: np.ndarray,
        r_start: int,
        r_end: int,
        left: int,
        right: int,
        sendid: Dict[int, np.ndarray],
        recvid: Dict[int, np.ndarray],
        recv_idx_map: Dict[int, Dict[int, int]],
        comm,
        n_threads: int = 1,
    ):
        self.rowptr = rowptr
        self.colidx = colidx
        self.val = val
        self.r_start = r_start
        self.r_end = r_end
        self.left = left
        self.right = right
        self.sendid = sendid
        self.recvid = recvid
        self.recv_idx_map = recv_idx_map
        self.comm = comm
        self.rank = comm.Get_rank()
        self.nprocs = comm.Get_size()
        self.n_threads = n_threads

        self.nlocal = r_end - r_start
        self.nnz_local = len(val)
        self.diag_len = right - left

        # Gather all ranks' initial row ranges for column ownership lookup
        self._owner_bounds = comm.allgather((r_start, r_end))

        # ── Build unified x-buffer layout ──
        # remote_offset[q] = start position of peer q's data in x_buf
        self.remote_offset: Dict[int, int] = {}
        offset = self.diag_len
        for q in sorted(recvid.keys()):
            self.remote_offset[q] = offset
            offset += len(recvid[q])
        self.total_remote = offset - self.diag_len
        self.x_buf_len = self.diag_len + self.total_remote

        # ── Build local_pos array (Algorithm 4 preparation) ──
        self.local_pos = np.zeros(self.nnz_local, dtype=np.int32)
        for j in range(self.nnz_local):
            c = int(self.colidx[j])
            if left <= c < right:
                self.local_pos[j] = c - left
            else:
                owner = self._find_owner_fast(c)
                if owner == self.rank:
                    # We own this column — put in local part
                    self.local_pos[j] = c - left
                elif owner in self.recv_idx_map and c in self.recv_idx_map[owner]:
                    rpos = self.recv_idx_map[owner][c]
                    self.local_pos[j] = self.remote_offset[owner] + rpos
                else:
                    # Column not needed (shouldn't happen with correct setup)
                    self.local_pos[j] = 0

        # x_buf allocated once, reused for every multiply call
        self.x_buf = np.zeros(self.x_buf_len, dtype=np.float64)

        # y buffer — per-thread accumulators
        self.y = np.zeros(self.nlocal, dtype=np.float64)

        logger.info(
            "DistSpMVSolver: nlocal=%d nnz_local=%d diag_len=%d "
            "total_remote=%d n_threads=%d",
            self.nlocal, self.nnz_local, self.diag_len,
            self.total_remote, n_threads,
        )

    # ── Public API ──────────────────────────────────────────────────

    def multiply(self, x_global: np.ndarray) -> np.ndarray:
        """Compute y_local = A_local @ x_global (distributed SpMV).

        Parameters
        ----------
        x_global : np.ndarray
            Full global x vector (length N).  Only the portion relevant
            to this process is accessed.

        Returns
        -------
        y_local : np.ndarray
            Local portion of y = A @ x (length = nlocal).
        """
        # Algorithm 3: Communication
        self._exchange_remote(x_global)

        # Algorithm 4: Local SpMV
        self._local_spmv()

        return self.y.copy()

    def multiply_benchmark(self, x_global: np.ndarray, n_warmup: int = 5,
                           n_repeat: int = 50) -> Tuple[float, float]:
        """Run multiply with warm-up and return (avg_time, std_time) in seconds."""
        # Warm-up
        for _ in range(n_warmup):
            self.multiply(x_global)

        times = np.empty(n_repeat, dtype=np.float64)
        for i in range(n_repeat):
            # Need fresh x_buf state but same x_global
            t0 = time.perf_counter()
            self.multiply(x_global)
            times[i] = time.perf_counter() - t0

        return float(np.mean(times)), float(np.std(times))

    # ── Algorithm 3: MPI Communication ──────────────────────────────

    def _exchange_remote(self, x_global: np.ndarray) -> None:
        """MPI communication phase: fetch remote x elements (Algorithm 3).

        Steps:
          1. Copy local x elements into x_buf.
          2. For each peer q: pack x_global[sendid[q][k]] into send buffer.
          3. Post non-blocking receives for incoming data.
          4. Post non-blocking sends.
          5. Wait for all transfers, then copy received data into x_buf.
        """
        from mpi4py import MPI

        # 1. Copy local x part
        self.x_buf[:self.diag_len] = x_global[self.left:self.right]

        # 2. Pack send buffers
        send_bufs: Dict[int, np.ndarray] = {}
        for q, cols in self.sendid.items():
            if len(cols) == 0:
                continue
            # Map global column → position in our local x part
            # col is in [left, right) since WE own it
            buf = np.empty(len(cols), dtype=np.float64)
            for k, c in enumerate(cols):
                buf[k] = x_global[int(c)]
            send_bufs[q] = buf

        # 3. Post receives for incoming remote x elements
        recv_bufs: Dict[int, np.ndarray] = {}
        recv_reqs: list = []
        for q, cols in self.recvid.items():
            if len(cols) == 0:
                continue
            buf = np.empty(len(cols), dtype=np.float64)
            recv_bufs[q] = buf
            req = self.comm.Irecv(buf, source=q, tag=200 + q)
            recv_reqs.append(req)

        # 4. Post sends
        send_reqs: list = []
        for q, buf in send_bufs.items():
            req = self.comm.Isend(buf, dest=q, tag=200 + self.rank)
            send_reqs.append(req)

        # 5. Wait for completion
        MPI.Request.Waitall(recv_reqs + send_reqs)

        # 6. Unpack received data into x_buf
        for q, buf in recv_bufs.items():
            offset = self.remote_offset[q]
            self.x_buf[offset:offset + len(buf)] = buf

    # ── Algorithm 4: Multi-threaded Local SpMV ─────────────────────

    def _local_spmv(self) -> None:
        """Local SpMV using thread-level load balancing (Algorithm 4).

        Rows are partitioned so that each thread handles ~equal nnz.
        Each thread writes to a private y buffer; results are summed
        at the end.

        Python note: due to the GIL, true multi-threading requires
        releasing the GIL (e.g. via numba or C extension).  This
        implementation provides the *algorithm* structure; for real
        OpenMP performance use the C++ backend or ``numba.jit(nopython=True)``.
        """
        # Zero out y
        self.y.fill(0.0)

        if self.n_threads <= 1:
            # Single-threaded path (no GIL contention)
            self._spmv_range(0, self.nlocal, self.y)
            return

        # Multi-threaded: partition rows by nnz count
        row_nnz = np.diff(self.rowptr).astype(np.int32)
        total_nnz = int(row_nnz.sum())
        target_per_thread = max(1, total_nnz // self.n_threads)

        # Build thread boundaries
        thread_bounds = [0]  # start row for each thread
        acc = 0
        for i in range(self.nlocal):
            acc += row_nnz[i]
            if acc >= target_per_thread and len(thread_bounds) < self.n_threads:
                thread_bounds.append(i + 1)
                acc = 0
        thread_bounds.append(self.nlocal)

        # Allocate per-thread y buffers
        thread_ys = [np.zeros(self.nlocal, dtype=np.float64)
                     for _ in range(len(thread_bounds) - 1)]

        # Launch threads
        import threading
        threads = []
        for t in range(len(thread_bounds) - 1):
            start_row = thread_bounds[t]
            end_row = thread_bounds[t + 1]
            y_priv = thread_ys[t]
            th = threading.Thread(
                target=self._spmv_range,
                args=(start_row, end_row, y_priv),
            )
            threads.append(th)
            th.start()

        for th in threads:
            th.join()

        # Reduction: sum per-thread contributions
        for yt in thread_ys:
            self.y += yt

    def _spmv_range(self, start_row: int, end_row: int,
                    y_out: np.ndarray) -> None:
        """Compute y_out[i] += sum_j A[i,j] * x_buf[local_pos[j]]
        for rows in [start_row, end_row)."""
        rowptr = self.rowptr
        val = self.val
        local_pos = self.local_pos
        x_buf = self.x_buf

        for i in range(start_row, end_row):
            acc = 0.0
            r_start = int(rowptr[i])
            r_end = int(rowptr[i + 1])
            for j in range(r_start, r_end):
                acc += val[j] * x_buf[local_pos[j]]
            y_out[i] += acc

    # ── Helpers ─────────────────────────────────────────────────────

    def _find_owner_fast(self, col: int) -> int:
        """Quick owner lookup using initial row distribution."""
        for q, (r0, r1) in enumerate(self._owner_bounds):
            if r0 <= col < r1:
                return q
        return self.rank

    @property
    def nnz_diag(self) -> int:
        """Number of nonzeros in the diagonal block."""
        return int(np.sum(
            (self.colidx >= self.left) & (self.colidx < self.right)
        ))

    @property
    def nnz_offdiag(self) -> int:
        """Number of nonzeros outside the diagonal block."""
        return self.nnz_local - self.nnz_diag
